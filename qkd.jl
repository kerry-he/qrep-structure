using LinearAlgebra
using SparseArrays
import MAT

import Hypatia
import Hypatia.Cones
import Hypatia.Solvers

include("cones/quantkeyrate.jl")
include("cones/quantcondentr.jl")
include("systemsolvers/elim.jl")
include("utils/helper.jl")

import Random
# Random.seed!(1)

T = Float64

function qkd_problem(
    K_list::Vector{Matrix{R}},
    Z_list::Vector{Matrix{T}},
    Γ::Vector{Matrix{R}},
    γ::Vector{T}
) where {T <: Real, R <: Hypatia.RealOrComplex{T}}
    # Build quantum rate distortion problem
    #   min  S( G(X) || Z(G(X)) )
    #   s.t. A(X) = b
    #        X ⪰ 0

    ni  = size(K_list[1], 2)    # Input dimension
    nc  = size(γ, 1)
    vni = Cones.svec_length(R, ni)

    Γ_op = reduce(hcat, [Hypatia.Cones.smat_to_svec!(zeros(T, vni), convert(Matrix{R}, G), sqrt(2.)) for G in Γ])'

    # Build problem model
    A = hcat(zeros(T, nc, 1), Γ_op)
    b = γ

    c = vcat(one(T), zeros(T, vni))

    G = -one(T) * I
    h = zeros(T, 1 + vni)

    cones = [QuantKeyRate{T, R}(K_list, Z_list)]

    return Hypatia.Models.Model{T}(c, A, b, G, h, cones)
end

function qkd_naive_problem(
    K_list::Vector{Matrix{R}},
    ZK_list::Vector{Matrix{R}},
    Γ::Vector{Matrix{R}},
    γ::Vector{T}
) where {T <: Real, R <: Hypatia.RealOrComplex{T}}
    # Build quantum rate distortion problem
    #   min  S( G(X) || Z(G(X)) )
    #   s.t. A(X) = b
    #        X ⪰ 0

    (no, ni)  = size(K_list[1])    # Input dimension
    nc  = size(γ, 1)
    vni = Cones.svec_length(R, ni)
    vno = Cones.svec_length(R, no)

    K_op  = lin_to_mat(R, x -> sum([K * x * K' for K in K_list]), ni, no)
    ZK_op = lin_to_mat(R, x -> sum([ZK * x * ZK' for ZK in ZK_list]), ni, no)
    Γ_op  = reduce(hcat, [Hypatia.Cones.smat_to_svec!(zeros(T, vni), convert(Matrix{R}, G), sqrt(2.)) for G in Γ])'

    # Build problem model
    A = hcat(zeros(T, nc, 1), Γ_op)
    b = γ

    c = vcat(one(T), zeros(T, vni))

    G1 = hcat(one(T), zeros(T, 1, vni))
    G2 = hcat(zeros(T, vno), ZK_op)
    G3 = hcat(zeros(T, vno), K_op)
    G4 = hcat(zeros(T, vni), 1.0I)
    G = -vcat(G1, G2, G3, G4)
    h = zeros(T, 1 + 2 * vno + vni)

    cones = [Cones.EpiTrRelEntropyTri{T, R}(1 + 2*vno), Cones.PosSemidefTri{T, R}(vni)]

    return Hypatia.Models.Model{T}(c, A, b, G, h, cones)
end


function main()
    # Define rate distortion problem with entanglement fidelity distortion
    f = MAT.matopen("data/DMCV_11_60_05_35.mat")
    data = MAT.read(f, "Data")

    if all(imag(data["Klist"][:]) == 0) && all(imag(data["Gamma_fr"][:]) == 0)
        R = T
    else
        R = Complex{T}
    end
    
    K_list = convert(Vector{Matrix{R}}, data["Klist"][:])
    Z_list = convert(Vector{Matrix{T}}, data["Zlist"][:])
    Γ = convert(Vector{Matrix{R}}, data["Gamma_fr"][:])
    γ = convert(Vector{T}, data["gamma_fr"][:])

    model = qkd_problem(K_list, Z_list, Γ, γ)
    solver = Solvers.Solver{T}(verbose = true, reduce = false, rescale = false, preprocess = true, syssolver = ElimSystemSolver{T}())
    Solvers.load(solver, model)
    Solvers.solve(solver)
    
    println("Solve time: ", Solvers.get_solve_time(solver) - solver.time_rescale - solver.time_initx - solver.time_inity)
    println("Num iter: ", Solvers.get_num_iters(solver))
    println("Abs gap: ", Solvers.get_primal_obj(solver) - Solvers.get_dual_obj(solver))


    K_list = convert(Vector{Matrix{R}}, data["Klist_fr"][:])
    ZK_list = convert(Vector{Matrix{R}}, data["ZKlist_fr"][:])
    Γ = convert(Vector{Matrix{R}}, data["Gamma_fr"][:])
    γ = convert(Vector{T}, data["gamma_fr"][:])

    model = qkd_naive_problem(K_list, ZK_list, Γ, γ)
    solver = Solvers.Solver{T}(verbose = true)
    Solvers.load(solver, model)
    Solvers.solve(solver)
    
    println("Solve time: ", Solvers.get_solve_time(solver) - solver.time_rescale - solver.time_initx - solver.time_inity)
    println("Num iter: ", Solvers.get_num_iters(solver))
    println("Abs gap: ", Solvers.get_primal_obj(solver) - Solvers.get_dual_obj(solver))    
end

main()