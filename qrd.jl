using LinearAlgebra
using SparseArrays

import Hypatia
import Hypatia.Cones
import Hypatia.Solvers

include("cones/quantcondentr.jl")
include("systemsolvers/elim.jl")
include("utils/helper.jl")

import Random
Random.seed!(1)

T = Float64

# Rate distortion problem data
n = 6
N = n^2
sn = Cones.svec_length(n)
sN = Cones.svec_length(N)
dim = sN + 1

λ_A  = eigvals(randDensityMatrix(T, n))
ρ_A  = diagm(λ_A)
ρ_AR = purify(λ_A)

D = 0.5
Δ = zeros(T, sN)
Cones.smat_to_svec!(Δ, I - ρ_AR, sqrt(2))

tr2 = lin_to_mat(T, x -> pTr!(zeros(T, n, n), x, 2, (n, n)), N, n)


# Build problem model
A1 = hcat(zeros(T, sn, 1), tr2, zeros(T, sn, 1))
A2 = hcat(0              , Δ' , 1)
A = vcat(A1, A2)

b = zeros(T, sn + 1)
@views b1 = b[1:sn]
Cones.smat_to_svec!(b1, ρ_A, sqrt(2))
b[end] = D

c = vcat(one(T), zeros(T, dim-1), 0)
G = -one(T) * I
h = zeros(T, dim + 1)


# Solve problem
cones = [QuantCondEntropy{T}(dim, n, n, 1), Cones.Nonnegative{T}(1)]
model = Hypatia.Models.Model{T}(c, A, b, G, h, cones)

solver = Solvers.Solver{T}(verbose = true, reduce = false, preprocess = false, syssolver = ElimSystemSolver{T}())
Solvers.load(solver, model)

Solvers.solve(solver)

println("Solve time: ", Solvers.get_solve_time(solver) - solver.time_rescale - solver.time_initx - solver.time_inity)
println("Num iter: ", Solvers.get_num_iters(solver))
println("Abs gap: ", Solvers.get_primal_obj(solver) - Solvers.get_dual_obj(solver))