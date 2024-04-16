using LinearAlgebra

import Hypatia
import Hypatia.Cones
import Hypatia.Solvers

include("../cones/quantcondentr.jl")
include("../systemsolvers/elim.jl")
include("../utils/linear.jl")
include("../utils/quantum.jl")
include("../utils/helper.jl")

import Random
Random.seed!(1)

T = Float64

function qrd_problem(n::Int, m::Int, W::Matrix{T}, Δ::Matrix{T}, D::Float64) where {T <: Real}
    # Build quantum rate distortion problem
    #   min  t
    #   s.t. Tr_2[X] = W
    #        (t, X) ∈ K_qce
    #        ⟨Δ, X⟩ ≤ D

    N = n * m
    sn = Cones.svec_length(n)
    sN = Cones.svec_length(N)

    trB = lin_to_mat(T, x -> pTr!(zeros(T, m, m), x, 1, (n, m)), N, n)
    Δ_vec = Cones.smat_to_svec!(zeros(T, sN), Δ, sqrt(2.))
    
    # Build problem model
    A1 = hcat(zeros(T, sn, 1), trB    , zeros(T, sn, 1))        # Tr_B[X] = W
    A2 = hcat(zero(T)        , Δ_vec' , one(T))                 # D = ⟨Δ, X⟩
    A  = vcat(A1, A2)

    b = zeros(T, sn + 1)
    @views Cones.smat_to_svec!(b[1:sn], W, sqrt(2.))
    b[end] = D

    c = vcat(one(T), zeros(T, sN + 1))

    G = -one(T) * I
    h = zeros(T, sN + 2)

    cones = [QuantCondEntropy{T}(n, m, 2), Cones.Nonnegative{T}(1)]

    return Hypatia.Models.Model{T}(c, A, b, G, h, cones, obj_offset=entr(W))
end

function qrd_naive_problem(n::Int, m::Int, W::Matrix{T}, Δ::Matrix{T}, D::Float64) where {T <: Real}
    # Build quantum rate distortion problem
    #   min  t
    #   s.t. Tr_2[X] = W
    #        (t, X, I⊗Tr_1[X]) ∈ K_qre
    #        ⟨Δ, X⟩ ≤ D

    N = n * m
    sn = Cones.svec_length(n)
    sN = Cones.svec_length(N)

    ikr_trR = lin_to_mat(T, x -> idKron!(zeros(T, N, N), pTr!(zeros(T, n, n), x, 2, (n, m)), 2, (n, m)), N, N)
    trB     = lin_to_mat(T, x -> pTr!(zeros(T, m, m), x, 1, (n, m)), N, n)
    Δ_vec   = Cones.smat_to_svec!(zeros(T, sN), Δ, sqrt(2.))
    
    # Build problem model
    A = hcat(zeros(T, sn, 1), trB)        # Tr_B[X] = W
    b = Cones.smat_to_svec!(zeros(T, sn), W, sqrt(2.))
    
    G1 = hcat(one(T)         , zeros(T, 1, sN))
    G2 = hcat(zeros(T, sN, 1), ikr_trR)
    G3 = hcat(zeros(T, sN, 1), one(T)*I(sN))
    G4 = hcat(zero(T)        , -Δ_vec')
    G = -vcat(G1, G2, G3, G4)

    h = zeros(T, 2 + 2*sN)
    h[end] = D

    c = vcat(one(T), zeros(T, sN))

    cones = [Cones.EpiTrRelEntropyTri{T, T}(1 + 2*sN), Cones.Nonnegative{T}(1)]

    return Hypatia.Models.Model{T}(c, A, b, G, h, cones, obj_offset=entr(W))
end

function precompile()
    n = 2
    W = randDensityMatrix(T, n)
    Δ = I - purify(W)
    D = 0.5

    # Use quantum conditional entropy cone
    model = qrd_problem(n, n, W, Δ, D)
    solver = Solvers.Solver{T}(verbose = false, iter_limit = 2, reduce = false, syssolver = ElimSystemSolver{T}())
    Solvers.load(solver, model)
    Solvers.solve(solver)

    # Use quantum relative entropy cone
    model = qrd_naive_problem(n, n, W, Δ, D)
    solver = Solvers.Solver{T}(verbose = false, iter_limit = 2)
    Solvers.load(solver, model)
    Solvers.solve(solver)
end

function main(csv_name::String, all_tests::Bool)
    # Solve the quantum rate distortion problem
    #   min  S(B|BR)_X + S(W)
    #   s.t. Tr_B[X] = W
    #        ⟨Δ, X⟩ ≤ D
    #        X ⪰ 0
    
    test_set = [2; 4; 6]
    if all_tests
        test_set = [test_set; 8; 10; 12; 14; 16]
    end

    problem = "qrd"
    
    # Precompile with small problem
    precompile()
    
    # Loop through all the problems
    for test in test_set
        n = test
        description = string(n)

        # Generate problem data
        Random.seed!(1)
        W = randDensityMatrix(T, n)
        Δ = I - purify(W)
        D = 0.5

        # Use quantum conditional entropy cone
        model = qrd_problem(n, n, W, Δ, D)
        solver = Solvers.Solver{T}(verbose = true, reduce = false, syssolver = ElimSystemSolver{T}())
        try_solve(model, solver, problem, description, "QCE", csv_name)

        # Use quantum relative entropy cone
        model = qrd_naive_problem(n, n, W, Δ, D)
        solver = Solvers.Solver{T}(verbose = true)
        try_solve(model, solver, problem, description, "QRE", csv_name)
    end

end