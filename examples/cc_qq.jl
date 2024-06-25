using LinearAlgebra
import Random

import Hypatia
import Hypatia.Cones
import Hypatia.Solvers

include("../cones/quantcoherentinf.jl")
include("../cones/quantcondentr.jl")
include("../systemsolvers/elim.jl")
include("../utils/linear.jl")
include("../utils/quantum.jl")
include("../utils/helper.jl")



T = Float64

function qqcc_problem(ni::Int, no::Int, ne::Int, N::Function, Nc::Function)
    # Build quantum-quantum channel capacity problem
    #   min  t
    #   s.t. tr[X] = 1
    #        (t, X) ∈ K_qci

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
    #   min  t
    #   s.t. tr[X] = 1
    #        (t, WN(X)W') ∈ K_qce
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
    #   min  t
    #   s.t. tr[X] = 1
    #        (t, WN(X)W', I⊗Tr_1[WN(X)W']) ∈ K_qre
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

function precompile_cc_qq()
    V, W = randDegradableChannel(T, 2, 2, 2)
    N(x)  = pTr!(zeros(T, 2, 2), V*x*V', 2, (2, 2))
    Nc(x) = pTr!(zeros(T, 2, 2), V*x*V', 1, (2, 2))

    # Use quantum mutual information cone
    model = qqcc_problem(2, 2, 2, N, Nc)
    solver = Solvers.Solver{T}(verbose = false, iter_limit = 2, reduce = false, syssolver = ElimSystemSolver{T}())
    Solvers.load(solver, model)
    Solvers.solve(solver)

    # Use quantum conditional entropy cone
    model = qqcc_qce_problem(2, 2, 2, N, W)
    solver = Solvers.Solver{T}(verbose = false, iter_limit = 2)
    Solvers.load(solver, model)
    Solvers.solve(solver)

    # Use quantum relative entropy cone
    model = qqcc_qre_problem(2, 2, 2, N, W)
    solver = Solvers.Solver{T}(verbose = false, iter_limit = 2)
    Solvers.load(solver, model)
    Solvers.solve(solver)
end

function main_cc_qq(csv_name::String, all_tests::Bool)
    # Solve for quantum-quantum channel capacity
    #   max  -S( WN(X)W' || I x Tr_2[WN(X)W'] )
    #   s.t. tr[X] = 1
    #        X ⪰ 0

    test_set = [2; 4]
    if all_tests
        test_set = [test_set; 8; 16; 32; 64]
    end

    problem = "cc_qq"
    
    # Precompile with small problem
    precompile_cc_qq()
    
    # Loop through all the problems
    for test in test_set
        ni = no = ne = test
        description = string(ni)

        # Define random instance of qq channel capacity problem
        Random.seed!(1)
        V, W = randDegradableChannel(T, ni, no, ne)
        N(x)  = pTr!(zeros(T, no, no), V*x*V', 2, (no, ne))
        Nc(x) = pTr!(zeros(T, ne, ne), V*x*V', 1, (no, ne))

        # Use quantum conditional information cone
        model = qqcc_problem(ni, no, ne, N, Nc)
        solver = Solvers.Solver{T}(verbose = true, reduce = false, syssolver = ElimSystemSolver{T}())
        try_solve(model, solver, problem, description, "QCI", csv_name)

        # Use quantum conditional entropy cone
        model = qqcc_qce_problem(ni, no, ne, N, W)
        solver = Solvers.Solver{T}(verbose = true)
        try_solve(model, solver, problem, description, "QCE", csv_name)

        # Use quantum relative entropy cone
        model = qqcc_qre_problem(ni, no, ne, N, W)
        solver = Solvers.Solver{T}(verbose = true)
        try_solve(model, solver, problem, description, "QRE", csv_name)
    end

end