using LinearAlgebra
using JuMP
using Hypatia
import Hypatia.Solvers
using Random

# See example
# https://github.com/chriscoey/Hypatia.jl/blob/master/examples/relentrentanglement/JuMP.jl

function heisenberg(delta,L)
    sx = [0 1; 1 0]
    sy = [0 -1im; 1im 0]
    sz = [1 0; 0 -1]
    id = [1 0; 0 1]
    h = -(kron(sx,sx) + kron(sy,sy) + delta*kron(sz,sz))
    return real( kron(h,I(2^(L-2))) )
end

function hXYPlain(beta,L)
    sx = [0 1.0; 1.0 0]/2.0
    sy = [0 1im; -1im 0]/2.0
    return -beta*real(kron(sx,sx,I(2^(L-2)))+kron(sy,sy,I(2^(L-2))))
end

function TrX(p,sys,dim)
    # Partial trace
    n = length(dim)

    #if length(sys) == 0
        # No system to trace out
    #    return p
    #end

    if any(i -> i < 1 || i > n, sys)
        println("Argument sys is invalid")
        return
    end

    if prod(dim) != size(p,1)
        println("Argument dim is invalid: prod(dim) != len(p)")
        return
    end


    keep = setdiff(1:n,sys) # systems to keep = {1,...,n} \ sys
    dimkeep = prod(dim[keep]) # dimension of the output system
    dimtrace = prod(dim[sys])

    # dim2 is the concatenation [dim,dim]
    # note: reshape takes tuples. Converting from arrays to tuples is done with the splat operator ...
    dim2 = [dim; dim]
    rdim = reverse(dim)
    rkeep = reverse(keep)
    # Adapted from Matlab's quantinf
    perm = n+1 .- [rkeep; rkeep.-n; sys; sys.-n];
    x = reshape(permutedims(reshape(p,[rdim; rdim]...),perm),
                [dimkeep; dimkeep; dimtrace; dimtrace]...)
    x = [sum(x[i,j,k,k] for k=1:dimtrace) for i=1:dimkeep, j=1:dimkeep]
    if dimkeep == 1
        # Scalar
        x = x[1,1]
    end
    return x
end

function randZ(n,cplx=true,rng=nothing)
    if rng == nothing
        rng = Random.default_rng()
    end
    if cplx
        return randn(rng,n,n) + 1im*randn(rng,n,n)
    else
        return randn(rng,n,n)
    end
end

function randH(n,cplx=true,rng=nothing)
    H = randZ(n,cplx,rng)
    return H+H'
end

function assertAlmostEqual(a,b)
    @assert maximum(abs.(a-b)) < 1e-12 "Not almost equal"
end

function test_TrX()

    # Bipartite system
    dim = [2,2]
    p1 = randH(dim[1])
    p2 = randH(dim[2])
    p = kron(p1,p2)
    assertAlmostEqual(TrX(p,[],dim),p)
    assertAlmostEqual(TrX(p,[1],dim),tr(p1)*p2)
    assertAlmostEqual(TrX(p,[2],dim),tr(p2)*p1)
    assertAlmostEqual(TrX(p,[1,2],dim),tr(p))

    # Tripartite system
    dim = [3,2,5]
    p1 = randH(dim[1])
    p2 = randH(dim[2])
    p3 = randH(dim[3])
    p = kron(p1,p2,p3)
    assertAlmostEqual(TrX(p,[],dim),p)
    assertAlmostEqual(TrX(p,[1],dim),tr(p1)*kron(p2,p3))
    assertAlmostEqual(TrX(p,[2],dim),tr(p2)*kron(p1,p3))
    assertAlmostEqual(TrX(p,[3],dim),tr(p3)*kron(p1,p2))
    assertAlmostEqual(TrX(p,[1,2],dim),tr(p1)*tr(p2)*p3)
    assertAlmostEqual(TrX(p,[1,3],dim),tr(p1)*tr(p3)*p2)
    assertAlmostEqual(TrX(p,[2,3],dim),tr(p2)*tr(p3)*p1)
    assertAlmostEqual(TrX(p,[1,2,3],dim),tr(p))

end

function quantum_rel_entr(X,Y)

    # In Julia, log(X) is the matrix log of X [elementwise log is log.(X)]
    return tr(X*(log(X)-log(Y)))

end

function quantum_entr(X)
    return tr(X*log(X))
end

# Build MED problem
#   min tr(h*rho)
#   s.t. rho >= 0, trace(rho) == 1
#        Tr_1 rho == Tr_L rho
#        S(L|1...L-1) >= 0

L = 4;
H = heisenberg(-1,L); # Hamiltonian
dims = 2*ones(Int,L);

# Parameters for Hypatia modeling
rho_vec_dim = Hypatia.Cones.svec_length(2^L) # = (2^L + 1 choose 2) = dimension of space where rho lives (assuming rho is real symmetric)
rt2 = sqrt(2)

opt = Hypatia.Optimizer(verbose = true)
model = Model(() -> opt)
@variable(model, rho_vec[1:rho_vec_dim])
@variable(model, rho2_vec[1:rho_vec_dim])
rho = zeros(AffExpr, 2^L, 2^L)
Hypatia.Cones.svec_to_smat!(rho, one(Float64)*rho_vec,rt2)
rho = Symmetric(rho)

rho2 = kron(TrX(rho,L,dims),I(2)) # rho2 = Tr_L(rho) \ox I_2
rho2_vec = zeros(AffExpr, rho_vec_dim)  # rho2_vec holds rho_{1...L-1} \ox id_L in vectorized form
Hypatia.Cones.smat_to_svec!(rho2_vec, rho2, rt2)

@constraint(model, tr(rho) == 1)
@constraint(model, TrX(rho,L,dims) .== TrX(rho,1,dims)) # Consistency constraint
# @constraint(model, rho in PSDCone())
@constraint(model, vcat(0, rho2_vec, rho_vec) in Hypatia.EpiTrRelEntropyTriCone{Float64, Float64}(1 + 2*rho_vec_dim))

medobj = sum(H.*rho);

@objective(model,Min,medobj)

println("Now solving")

optimize!(model)

println("L = ", L, ", MED value = ", value.(medobj))
println("Preprocessing time: ", opt.solver.time_rescale + opt.solver.time_initx + opt.solver.time_inity)
println("Solve time: ", Solvers.get_solve_time(opt.solver) - opt.solver.time_rescale - opt.solver.time_initx - opt.solver.time_inity)
println("Num iter: ", Solvers.get_num_iters(opt.solver))
println("Abs gap: ", Solvers.get_primal_obj(opt.solver) - Solvers.get_dual_obj(opt.solver))