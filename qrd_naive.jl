using LinearAlgebra

import Hypatia
import Hypatia.Cones
import Hypatia.Solvers

function randPSD(n::Int)
    X = randn(n, n)
    X = X*X';
    return Symmetric(X / (tr(X)))
end

function purify(λ::Vector{T}) where {T <: Real}
    n = length(λ)
    vec = sparse(collect(1:n+1:n^2), ones(n), sqrt.(λ))
    vec = collect(vec)
    return vec * vec'
end

function get_ptr(n::Int, sn::Int, sN::Int, sys::Int)
    ptr = zeros(sn, sN)
    k = 0
    for j = 1:n, i = 1:j
        k += 1
    
        H = zeros(T, n, n)
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

function get_kron_ptr(N::Int, sn::Int, sN::Int)
    ptr = zeros(sN, sN)
    k = 0
    for j = 1:N, i = 1:j
        k += 1
    
        H = zeros(T, N, N)
        if i == j
            H[i, j] = 1
        else
            H[i, j] = H[j, i] = 1/sqrt(2)
        end

        Id = Matrix{Float64}(I, isqrt(N), isqrt(N))
        
        @views ptr_k = ptr[:, k]
        ptr_H = pTr(H, 1)
        I_H = kron(Id, ptr_H)
        Cones.smat_to_svec!(ptr_k, I_H, sqrt(2))
    end

    return ptr
end

function pTr(X::Matrix{T}, sys::Int = 2, dim::Union{Tuple{Int, Int}, Nothing} = nothing) where {T <: Real}

    if dim === nothing
        n1 = n2 = isqrt(size(X, 1))
    else
        n1 = dim[1]
        n2 = dim[2]
    end
    @assert n1*n2 == size(X, 1) == size(X, 2)

    if sys == 2
        ptrX = zeros(T, n1, n1)
        @inbounds for i = 1:n1, j = 1:n1
            ptrX[i, j] = tr( X[(i-1)*n2+1 : i*n2, (j-1)*n2+1 : j*n2] )
        end
    else
        ptrX = zeros(T, n2, n2)
        ptrX .= 0
        @inbounds for i = 1:n1
            @. ptrX += X[(i-1)*n2+1 : i*n2, (i-1)*n2+1 : i*n2]
        end
    end

    return ptrX
end



import Random
Random.seed!(1)

T = Float64
n = 4
N = n^2
sn = Cones.svec_length(n)
sN = Cones.svec_length(N)
dim = 2*sN + 1

# Rate distortion problem data
λ_A  = ones(T, n)
λ_A /= sum(λ_A)
ρ_A  = diagm(λ_A)
ρ_AR = purify(λ_A)

entr_A = -sum(λ_A .* log.(λ_A))

Δ = zeros(T, sN)
Cones.smat_to_svec!(Δ, I - ρ_AR, sqrt(2))

D = 0.25

kron_ptr = get_kron_ptr(N, sn, sN)
tr1 = get_ptr(n, sn, sN, 1)
tr2 = get_ptr(n, sn, sN, 2)
Id = Matrix{T}(I, sN, sN)

# Build problem model
A = hcat(zeros(T, sn, 1), tr2)
b = zeros(T, sn)
Cones.smat_to_svec!(b, ρ_A, sqrt(2))


G1 = hcat(-1, zeros(T, 1, sN))
G2 = hcat(zeros(T, sN, 1), -kron_ptr)
G3 = hcat(zeros(T, sN, 1), -Id)
G4 = hcat(0, Δ')
G = vcat(G1, G2, G3, G4)

h = zeros(T, 2*sN + 2)
h[end] = D


c = vcat(one(T), zeros(T, sN))

cones = [Cones.EpiTrRelEntropyTri{T}(dim), Cones.Nonnegative{T}(1)]
model = Hypatia.Models.Model{T}(c, A, b, G, h, cones, obj_offset=entr_A)

solver = Solvers.Solver{T}(verbose = true)
# solver = Solvers.Solver{T}(verbose = true, reduce = false, syssolver = Solvers.IdConeSystemSolver{T}())
Solvers.load(solver, model)
# VSCodeServer.@profview Solvers.solve(solver)
Solvers.solve(solver)

println("Solve time: ", Solvers.get_solve_time(solver) - solver.time_rescale - solver.time_initx - solver.time_inity)
println("Num iter: ", Solvers.get_num_iters(solver))
println("Abs gap: ", Solvers.get_primal_obj(solver) - Solvers.get_dual_obj(solver))