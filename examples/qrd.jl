using LinearAlgebra

import Hypatia
import Hypatia.Cones
import Hypatia.Solvers

include("../cones/quantcondentr.jl")
include("../systemsolvers/elim.jl")
include("../utils/helper.jl")
include("../utils/quantum.jl")

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


function main()
    # Solve the quantum rate distortion problem
    #   min  S(B|BR)_X + S(W)
    #   s.t. Tr_B[X] = W
    #        ⟨Δ, X⟩ ≤ D
    #        X ⪰ 0

    # Define rate distortion problem with entanglement fidelity distortion
    n = 4
    W = randDensityMatrix(T, n)
    Δ = I - purify(W)
    D = 0.5
    
    # Use quantum conditional entropy cone
    model = qrd_problem(n, n, W, Δ, D)
    solver = Solvers.Solver{T}(reduce = false, syssolver = ElimSystemSolver{T}())
    Solvers.load(solver, model)
    Solvers.solve(solver)
    print_statistics(solver, "Quantum rate distortion", "QCE")

    # Use quantum relative entropy cone
    model = qrd_naive_problem(n, n, W, Δ, D)
    solver = Solvers.Solver{T}()
    Solvers.load(solver, model)
    Solvers.solve(solver)
    println("Solved using QRE cone")
    print_statistics(solver, "Quantum rate distortion", "QRE")
end

main()