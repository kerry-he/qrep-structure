using LinearAlgebra

import Hypatia
import Hypatia.Cones
import Hypatia.Solvers
using Random

include("cones/quantcondentr.jl")
include("systemsolvers/elim.jl")

T = Float64

function heisenberg(delta::Float64, L::Int)
    sx = [0 1; 1 0]
    sy = [0 -1im; 1im 0]
    sz = [1 0; 0 -1]
    h = -(kron(sx, sx) + kron(sy, sy) + delta*kron(sz, sz))
    return real( kron(h, I(2^(L-2))) )
end

function med_problem(L::Int)
    # Build MED problem
    #   min  <H,X>
    #   s.t. X ⪰ 0, tr(X) = 1
    #        Tr_1[X] == Tr_L[X]
    #        S(L|1...L-1)_X ≥ 0

    N = 2^L
    sN = Cones.svec_length(N)

    m = 2^(L-1)
    sm = Cones.svec_length(m)

    tr1 = lin_to_mat(T, x -> pTr!(zeros(T, m, m), x, 1, (2, m)), N, m)
    trn = lin_to_mat(T, x -> pTr!(zeros(T, m, m), x, 2, (m, 2)), N, m)
    Id  = Cones.smat_to_svec!(zeros(sN), Matrix{T}(I, N, N), sqrt(2.))

    # Build problem model
    A1 = hcat(zeros(sm, 1), tr1 - trn)          # Tr_1[X] == Tr_L[X]
    A2 = hcat(0           , Id')                # tr[X] == 1
    A3 = hcat(1           , zeros(1, sN))       # t == 0
    A  = vcat(A1[2:end, :], A2, A3)

    b = zeros(sm + 1)
    b[end-1] = 1

    H = convert.(T, heisenberg(-1.0, L))
    c = zeros(T, 1 + sN)
    @views Cones.smat_to_svec!(c[2:end], H, sqrt(2.))

    G = -one(T) * I
    h = zeros(1 + sN)

    cones = [QuantCondEntropy{T}(m, 2, 2)]

    return Hypatia.Models.Model{T}(c, A, b, G, h, cones)
end

function med_naive_problem(L::Int)
    # Build MED problem
    #   min  <H,X>
    #   s.t. X ⪰ 0, tr(X) = 1
    #        Tr_1[X] == Tr_L[X]
    #        S(L|1...L-1)_X ≥ 0

    N = 2^L
    sN = Cones.svec_length(N)

    m = 2^(L-1)
    sm = Cones.svec_length(m)

    tr1     = lin_to_mat(T, x -> pTr!(zeros(T, m, m), x, 1, (2, m)), N, m)
    trn     = lin_to_mat(T, x -> pTr!(zeros(T, m, m), x, 2, (m, 2)), N, m)
    ikr_tr1 = lin_to_mat(T, x -> idKron!(zeros(T, N, N), pTr!(zeros(T, m, m), x, 1, (2, m)), 1, (2, m)), N, N)
    Id      = Cones.smat_to_svec!(zeros(sN), Matrix{T}(I, N, N), sqrt(2.))

    # Build problem model
    A  = vcat((tr1 - trn)[2:end, :], Id')

    b = zeros(sm)
    b[end] = 1

    H = convert.(T, heisenberg(-1.0, L))
    c = Cones.smat_to_svec!(zeros(T, sN), H, sqrt(2.))

    G = -vcat(zeros(T, 1, sN), ikr_tr1, one(T)*I(sN))

    h = zeros(1 + 2*sN)

    cones = [Cones.EpiTrRelEntropyTri{T, T}(1 + 2*sN)]

    return Hypatia.Models.Model{T}(c, A, b, G, h, cones)
end

function main()
    L = 6

    model = med_problem(L)
    solver = Solvers.Solver{T}(verbose = true, reduce = false, rescale = false, preprocess = false, syssolver = ElimSystemSolver{T}())
    Solvers.load(solver, model)
    Solvers.solve(solver)
    
    println("Solve time: ", Solvers.get_solve_time(solver) - solver.time_rescale - solver.time_initx - solver.time_inity)
    println("Num iter: ", Solvers.get_num_iters(solver))
    println("Abs gap: ", Solvers.get_primal_obj(solver) - Solvers.get_dual_obj(solver))


    model = med_naive_problem(L)
    solver = Solvers.Solver{T}(verbose = true)
    Solvers.load(solver, model)
    Solvers.solve(solver)
    
    println("Solve time: ", Solvers.get_solve_time(solver) - solver.time_rescale - solver.time_initx - solver.time_inity)
    println("Num iter: ", Solvers.get_num_iters(solver))
    println("Abs gap: ", Solvers.get_primal_obj(solver) - Solvers.get_dual_obj(solver))
end

main()