using LinearAlgebra

import Hypatia
import Hypatia.Cones
import Hypatia.Solvers

include("../cones/quantcondentr.jl")
include("../systemsolvers/elim.jl")
include("../utils/helper.jl")

import Random
Random.seed!(1)

T = Float64

function qrd_problem(n::Int, m::Int, Z::Matrix{T}, Δ::Matrix{T}, D::Float64) where {T <: Real}
    # Build quantum rate distortion problem
    #   min  S(B|BR)_X + S(Z)
    #   s.t. Tr_B[X] = Z
    #        <Δ,X> ≤ D
    #        X ⪰ 0

    N = n * m
    sn = Cones.svec_length(n)
    sN = Cones.svec_length(N)

    trB = lin_to_mat(T, x -> pTr!(zeros(T, m, m), x, 1, (n, m)), N, n)
    Δ_vec = Cones.smat_to_svec!(zeros(T, sN), Δ, sqrt(2.))
    
    # Build problem model
    A1 = hcat(zeros(T, sn, 1), trB    , zeros(T, sn, 1))        # Tr_B[X] = Z
    A2 = hcat(zero(T)        , Δ_vec' , one(T))                 # D = <Δ,X>
    A  = vcat(A1, A2)

    b = zeros(T, sn + 1)
    @views Cones.smat_to_svec!(b[1:sn], Z, sqrt(2.))
    b[end] = D

    c = vcat(one(T), zeros(T, sN + 1))

    G = -one(T) * I
    h = zeros(T, sN + 2)

    cones = [QuantCondEntropy{T}(n, m, 2), Cones.Nonnegative{T}(1)]

    return Hypatia.Models.Model{T}(c, A, b, G, h, cones, obj_offset=entr(Z))
end

function qrd_naive_problem(n::Int, m::Int, Z::Matrix{T}, Δ::Matrix{T}, D::Float64) where {T <: Real}
    # Build quantum rate distortion problem
    #   min  S(B|BR)_X + S(Z)
    #   s.t. Tr_B[X] = Z
    #        <Δ,X> ≤ D
    #        X ⪰ 0

    N = n * m
    sn = Cones.svec_length(n)
    sN = Cones.svec_length(N)

    ikr_trR = lin_to_mat(T, x -> idKron!(zeros(T, N, N), pTr!(zeros(T, n, n), x, 2, (n, m)), 2, (n, m)), N, N)
    trB     = lin_to_mat(T, x -> pTr!(zeros(T, m, m), x, 1, (n, m)), N, n)
    Δ_vec   = Cones.smat_to_svec!(zeros(T, sN), Δ, sqrt(2.))
    
    # Build problem model
    A = hcat(zeros(T, sn, 1), trB)        # Tr_B[X] = Z
    b = Cones.smat_to_svec!(zeros(T, sn), Z, sqrt(2.))
    
    G1 = hcat(one(T)         , zeros(T, 1, sN))
    G2 = hcat(zeros(T, sN, 1), ikr_trR)
    G3 = hcat(zeros(T, sN, 1), one(T)*I(sN))
    G4 = hcat(zero(T)        , -Δ_vec')
    G = -vcat(G1, G2, G3, G4)

    h = zeros(T, 2 + 2*sN)
    h[end] = D

    c = vcat(one(T), zeros(T, sN))

    cones = [Cones.EpiTrRelEntropyTri{T, T}(1 + 2*sN), Cones.Nonnegative{T}(1)]

    return Hypatia.Models.Model{T}(c, A, b, G, h, cones, obj_offset=entr(Z))
end


function main()
    # Define rate distortion problem with entanglement fidelity distortion
    n = 4
    λ = eigvals(randDensityMatrix(T, n))
    Z = diagm(λ)    
    Δ = I - purify(λ)
    D = 0.5
    
    model = qrd_problem(n, n, Z, Δ, D)
    solver = Solvers.Solver{T}(verbose = true, reduce = false, syssolver = ElimSystemSolver{T}())
    Solvers.load(solver, model)
    Solvers.solve(solver)
    print_statistics(solver)


    # model = qrd_naive_problem(n, n, Z, Δ, D)
    # solver = Solvers.Solver{T}(verbose = true)
    # Solvers.load(solver, model)
    # Solvers.solve(solver)
    # print_statistics(solver)
end

main()