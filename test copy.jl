using LinearAlgebra
using SparseArrays

import Hypatia
import Hypatia.Cones
import Hypatia.Solvers

include("cones/condentropy.jl")
include("systemsolvers/elim.jl")

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

function get_tr2(n::Int, sn::Int, sN::Int, rt2::T) where {T <: Real}
    tr2 = zeros(sn, sN)
    k = 0
    for j = 1:n, i = 1:j
        k += 1
    
        H = zeros(T, n, n)
        if i == j
            H[i, j] = 1
        else
            H[i, j] = H[j, i] = 1/rt2
        end
        
        @views tr2_k = tr2[k, :]
        I_H = kron(H, I(n))
        Cones.smat_to_svec!(tr2_k, I_H, rt2)
    end

    return tr2
end

import Random
Random.seed!(1)

T = Float64
n = 10
N = n^2
sn = Cones.svec_length(n)
sN = Cones.svec_length(N)
dim = sN + 1

# Rate distortion problem data
λ_A  = rand(T, n)
λ_A /= sum(λ_A)
ρ_A  = diagm(λ_A)
ρ_AR = purify(λ_A)

Δ = zeros(T, sN)
Cones.smat_to_svec!(Δ, I - ρ_AR, sqrt(2))

D = 0.4

tr2 = get_tr2(n, sn, sN, sqrt(2))
Id = zeros(T, sn)
Cones.smat_to_svec!(Id, Matrix{T}(I, n, n), sqrt(2))

# Build problem model
A1 = hcat(zeros(T, sn, 1), tr2, zeros(T, sn, 1))
A2 = hcat(0              , Δ' , 1)
A = vcat(A1, A2)

b = zeros(T, sn + 1)
@views b1 = b[1:sn]
Cones.smat_to_svec!(b1, ρ_A, sqrt(2))
b[end] = D

c = vcat(one(T), zeros(T, dim-1), 0)
# A = rand(T, 2, dim)
# b = rand(T, 2)
G = -one(T) * I
h = zeros(T, dim + 1)
cones = [EpiCondEntropyTri{T}(dim, n, n, 1), Cones.Nonnegative{T}(1)]
model = Hypatia.Models.Model{T}(c, A, b, G, h, cones)

solver = Solvers.Solver{T}(verbose = true, reduce = false, preprocess = false, syssolver = ElimSystemSolver{T}())
# solver = Solvers.Solver{T}(verbose = true, reduce = false, preprocess = false)
# solver = Solvers.Solver{T}(verbose = true)
Solvers.load(solver, model)

Solvers.solve(solver)

println("Solve time: ", Solvers.get_solve_time(solver) - solver.time_rescale - solver.time_initx - solver.time_inity)
println("Num iter: ", Solvers.get_num_iters(solver))
println("Abs gap: ", Solvers.get_primal_obj(solver) - Solvers.get_dual_obj(solver))