using LinearAlgebra
using SparseArrays

import Hypatia
import Hypatia.Cones
import Hypatia.Solvers

include("cones/quantcoherentinf.jl")
include("cones/quantcondentr.jl")
include("systemsolvers/elim.jl")
include("utils/helper.jl")

import Random
# Random.seed!(1)

T = Float64

function qqcc_problem(ni::Int, no::Int, ne::Int, N::Function, Nc::Function)
    # Build quantum-quantum channel capacity problem
    #   max  -S( WN(X)W || I x Tr_2[WN(X)W'] )
    #   s.t. tr[X] = 1
    #        X ⪰ 0

    vni = Cones.svec_length(ni)

    tr = Cones.smat_to_svec!(zeros(T, vni), Matrix{T}(I, ni, ni), sqrt(2.))
    
    # Build problem model
    A = vcat(hcat(zero(T), tr'))
    b = ones(T, 1) 

    c = vcat(one(T), zeros(T, vni))

    G = -one(T) * I
    h = zeros(T, 1 + vni)

    cones = [QuantCoherentInformation{T}(ni, no, ne, N, Nc)]

    return Hypatia.Models.Model{T}(c, A, b, G, h, cones)
end

function qqcc_qce_problem(ni::Int, no::Int, ne::Int, N::Function, W::Matrix{T}) where {T <: Real}
    # Build quantum-quantum channel capacity problem
    #   max  -S( WN(X)W || I x Tr_2[WN(X)W'] )
    #   s.t. tr[X] = 1
    #        X ⪰ 0

    vni  = Cones.svec_length(ni)
    vnei = Cones.svec_length(ne*ni)

    tr  = Cones.smat_to_svec!(zeros(T, vni), Matrix{T}(I, ni, ni), sqrt(2.))
    WNW = lin_to_mat(T, x -> W*N(x)*W', ni, ni*ne)
    
    # Build problem model
    A = vcat(hcat(zeros(T, 1, 1), tr'))
    b = ones(T, 1) 

    c = vcat(ones(T, 1), zeros(T, vni))

    G1 = hcat(one(T), zeros(T, 1, vni))      # t_cond
    G2 = hcat(zeros(T, vnei, 1), WNW)        # X_cond
    G3 = hcat(zeros(T, vni, 1), 1.0I)        # PSD
    G = -vcat(G1, G2, G3)

    h = zeros(T, 1 + vnei + vni)

    cones = [QuantCondEntropy{T}(ne, ni, 2), Cones.PosSemidefTri{T, T}(vni)]

    return Hypatia.Models.Model{T}(c, A, b, G, h, cones)
end

function qqcc_qre_problem(ni::Int, no::Int, ne::Int, N::Function, W::Matrix{T}) where {T <: Real}
    # Build quantum-quantum channel capacity problem
    #   max  -S( WN(X)W || I x Tr_2[WN(X)W'] )
    #   s.t. tr[X] = 1
    #        X ⪰ 0

    nei  = ne*ni
    vni  = Cones.svec_length(ni)
    vnei = Cones.svec_length(nei)

    tr          = Cones.smat_to_svec!(zeros(T, vni), Matrix{T}(I, ni, ni), sqrt(2.))
    WNW         = lin_to_mat(T, x -> W*N(x)*W', ni, ni*ne)
    ikr_trB_WNW = lin_to_mat(T, x -> idKron!(zeros(T, nei, nei), pTr!(zeros(T, ne, ne), W*N(x)*W', 2, (ne, ni)), 2, (ne, ni)), ni, ni*ne)
    
    # Build problem model
    A = vcat(hcat(zeros(T, 1, 1), tr'))
    b = ones(T, 1) 

    c = vcat(ones(T, 1), zeros(T, vni))

    G1 = hcat(one(T), zeros(T, 1, vni))             # t_cond
    G2 = hcat(zeros(T, vnei, 1), ikr_trB_WNW)       # Y_cond
    G3 = hcat(zeros(T, vnei, 1), WNW)               # X_cond
    G4 = hcat(zeros(T, vni, 1), 1.0I)               # PSD
    G = -vcat(G1, G2, G3, G4)

    h = zeros(T, 1 + 2*vnei + vni)

    cones = [Cones.EpiTrRelEntropyTri{T, T}(1 + 2*vnei), Cones.PosSemidefTri{T, T}(vni)]

    return Hypatia.Models.Model{T}(c, A, b, G, h, cones)
end


function main()
    # Define random instance of ea channel capacity problem
    (ni, no, ne) = (4, 4, 4)
    V, W = randDegradableChannel(T, ni, no, ne)
    N(x)  = pTr!(zeros(T, no, no), V*x*V', 2, (no, ne))
    Nc(x) = pTr!(zeros(T, ne, ne), V*x*V', 1, (no, ne))

    model = qqcc_problem(ni, no, ne, N, Nc)
    solver = Solvers.Solver{T}(verbose = true, reduce = false, rescale = false, preprocess = true, syssolver = ElimSystemSolver{T}())
    Solvers.load(solver, model)
    Solvers.solve(solver)
    
    println("Solve time: ", Solvers.get_solve_time(solver) - solver.time_rescale - solver.time_initx - solver.time_inity)
    println("Num iter: ", Solvers.get_num_iters(solver))
    println("Abs gap: ", Solvers.get_primal_obj(solver) - Solvers.get_dual_obj(solver))


    model = qqcc_qce_problem(ni, no, ne, N, W)
    solver = Solvers.Solver{T}(verbose = true)
    Solvers.load(solver, model)
    Solvers.solve(solver)
    
    println("Solve time: ", Solvers.get_solve_time(solver) - solver.time_rescale - solver.time_initx - solver.time_inity)
    println("Num iter: ", Solvers.get_num_iters(solver))
    println("Abs gap: ", Solvers.get_primal_obj(solver) - Solvers.get_dual_obj(solver))


    model = qqcc_qre_problem(ni, no, ne, N, W)
    solver = Solvers.Solver{T}(verbose = true)
    Solvers.load(solver, model)
    Solvers.solve(solver)
    
    println("Solve time: ", Solvers.get_solve_time(solver) - solver.time_rescale - solver.time_initx - solver.time_inity)
    println("Num iter: ", Solvers.get_num_iters(solver))
    println("Abs gap: ", Solvers.get_primal_obj(solver) - Solvers.get_dual_obj(solver))        
end

main()