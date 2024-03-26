using LinearAlgebra

import Hypatia
import Hypatia.Cones
import Hypatia.Solvers
using Random

include("cones/quantcondentr.jl")
include("systemsolvers/elim.jl")

function heisenberg(delta,L)
    sx = [0 1; 1 0]
    sy = [0 -1im; 1im 0]
    sz = [1 0; 0 -1]
    id = [1 0; 0 1]
    h = -(kron(sx,sx) + kron(sy,sy) + delta*kron(sz,sz))
    return real( kron(h,I(2^(L-2))) )
end

function get_ptr(n::Int, m::Int, sm::Int, sN::Int, sys::Int)
    ptr = zeros(sm, sN)
    k = 0
    for j = 1:m, i = 1:j
        k += 1
    
        H = zeros(m, m)
        if i == j
            H[i, j] = 1
        else
            H[i, j] = H[j, i] = 1/sqrt(2)
        end

        Id = Matrix{Float64}(I, n, n)
        
        @views ptr_k = ptr[k, :]
        if sys == 1
            I_H = kron(Id, H)
        else
            I_H = kron(H, Id)
        end
        Cones.smat_to_svec!(ptr_k, I_H, sqrt(2))
    end

    return ptr
end


# Build MED problem
#   min tr(h*rho)
#   s.t. rho >= 0, trace(rho) == 1
#        Tr_1 rho == Tr_L rho
#        S(L|1...L-1) >= 0

L = 2
H = heisenberg(-1,L) # Hamiltonian
dims = 2*ones(Int,L)

N = 2^L
sN = Cones.svec_length(N)

m = 2^(L-1)
sm = Cones.svec_length(m)

# Parameters for Hypatia modeling
rt2 = sqrt(2)


# Build problem model
# tr1 = lin_to_mat(T, x -> pTr!(zeros(T, m, m), x, 1, (2, m)), N, m)

tr1 = get_ptr(2, m, sm, sN, 1)
trend = get_ptr(2, m, sm, sN, 2)
Id = zeros(sN)
Cones.smat_to_svec!(Id, Matrix{Float64}(I, N, N), sqrt(2))

A1 = hcat(zeros(sm, 1), tr1 - trend)
A2 = hcat(0              , Id')
A3 = hcat(1              , zeros(1, sN))
A = vcat(A1[2:end, :], A2, A3)

b = zeros(sm + 1)
b[end-1] = 1



H = convert.(Float64, heisenberg(-1,L)); # Hamiltonian
c_temp = zeros(sN)
Cones.smat_to_svec!(c_temp, H, sqrt(2))
c = vcat(0, c_temp)

G = -1.0 * I
h = zeros(1 + sN)


cones = [QuantCondEntropy{Float64}(1 + sN, 2^(L-1), 2, 2)]
model = Hypatia.Models.Model{Float64}(c, A, b, G, h, cones)

solver = Solvers.Solver{Float64}(verbose = true, reduce = false, rescale = false, preprocess = false, syssolver = ElimSystemSolver{Float64}())
Solvers.load(solver, model)

Solvers.solve(solver)

println("Solve time: ", Solvers.get_solve_time(solver))
println("Num iter: ", Solvers.get_num_iters(solver))
println("Abs gap: ", Solvers.get_primal_obj(solver) - Solvers.get_dual_obj(solver))

cones = [QuantCondEntropy{Float64}(1 + sN, 2^(L-1), 2, 2)]
model = Hypatia.Models.Model{Float64}(c, A, b, G, h, cones)

solver = Solvers.Solver{Float64}(verbose = true, reduce = false, rescale = false, preprocess = false, syssolver = ElimSystemSolver{Float64}())
Solvers.load(solver, model)

Solvers.solve(solver)

println("Solve time: ", Solvers.get_solve_time(solver))
println("Num iter: ", Solvers.get_num_iters(solver))
println("Abs gap: ", Solvers.get_primal_obj(solver) - Solvers.get_dual_obj(solver))