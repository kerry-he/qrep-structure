using LinearAlgebra
import Random

import Hypatia
import Hypatia.Cones
import Hypatia.Solvers

include("../cones/quantcondentr.jl")
include("../cones/quantmutualinf.jl")
include("../systemsolvers/elim.jl")
include("../utils/linear.jl")
include("../utils/quantum.jl")
include("../utils/helper.jl")

T = Float64

function eacc_problem(ni::Int, no::Int, ne::Int, V::Matrix{T}) where {T <: Real}
    # Build entanglement assisted channel capacity problem
    #   min  t
    #   s.t. tr[X] = 1
    #        (t, X) ∈ K_qmi

    vni = Cones.svec_length(ni)

    tr = Cones.smat_to_svec!(zeros(T, vni), Matrix{T}(I, ni, ni), sqrt(2.))
    
    # Build problem model
    A = vcat(hcat(zero(T), tr'))
    b = ones(T, 1) 

    c = vcat(one(T), zeros(T, vni))

    G = -one(T) * I
    h = zeros(T, 1 + vni)

    cones = [QuantMutualInformation{T}(ni, no, ne, V)]

    return Hypatia.Models.Model{T}(c, A, b, G, h, cones)
end

function eacc_qce_problem(ni::Int, no::Int, ne::Int, V::Matrix{T}) where {T <: Real}
    # Build entanglement assisted channel capacity problem
    #   min  t1 + t2
    #   s.t. tr[X] = 1
    #        (t1, VXV') ∈ K_qce
    #        (t2, Tr_2[VXV'], tr[X]) ∈ K_qe

    vni  = Cones.svec_length(ni)
    vno  = Cones.svec_length(no)
    vnoe = Cones.svec_length(no*ne)

    tr      = Cones.smat_to_svec!(zeros(T, vni), Matrix{T}(I, ni, ni), sqrt(2.))
    VV      = lin_to_mat(T, x -> V*x*V', ni, no*ne)
    trE_VV  = lin_to_mat(T, x -> pTr!(zeros(T, no, no), V*x*V', 2, (no, ne)), ni, no)
    
    # Build problem model
    A = vcat(hcat(zeros(T, 1, 2), tr'))
    b = ones(T, 1) 

    c = vcat(ones(T, 2), zeros(T, vni))

    G1 = hcat(one(T), zero(T)  , zeros(T, 1, vni))      # t_cond
    G2 = hcat(zeros(T, vnoe, 2), VV)                    # X_cond
    G3 = hcat(zero(T), one(T)  , zeros(T, 1, vni))      # t_entr
    G4 = hcat(zero(T), zero(T) , zeros(T, 1, vni))      # y_entr
    G5 = hcat(zeros(T, vno, 2) , trE_VV)                # X_entr
    G = -vcat(G1, G2, G3, G4, G5)

    h = zeros(T, 1 + vnoe + 1 + 1 + vno)
    h[1 + vnoe + 1 + 1] = 1.

    cones = [
        QuantCondEntropy{T}(no, ne, 1), 
        Cones.EpiPerSepSpectral{Cones.MatrixCSqr{T, T}, T}(
            Cones.NegEntropySSF(),
            no,
        ),
    ]

    return Hypatia.Models.Model{T}(c, A, b, G, h, cones)
end

function eacc_qre_problem(ni::Int, no::Int, ne::Int, V::Matrix{T}) where {T <: Real}
    # Build entanglement assisted channel capacity problem
    #   min  t1 + t2
    #   s.t. tr[X] = 1
    #        (t1, VXV', I⊗Tr_1[VXV']) ∈ K_qre
    #        (t2, Tr_2[VXV'], tr[X]) ∈ K_qe

    noe = no*ne
    vni  = Cones.svec_length(ni)
    vno  = Cones.svec_length(no)
    vnoe = Cones.svec_length(noe)

    tr      = Cones.smat_to_svec!(zeros(T, vni), Matrix{T}(I, ni, ni), sqrt(2.))
    VV      = lin_to_mat(T, x -> V*x*V', ni, noe)
    trE_VV  = lin_to_mat(T, x -> pTr!(zeros(T, no, no), V*x*V', 2, (no, ne)), ni, no)
    ikr_trB_VV = lin_to_mat(T, x -> idKron!(zeros(T, noe, noe), pTr!(zeros(T, ne, ne), V*x*V', 1, (no, ne)), 1, (no, ne)), ni, noe)
    
    # Build problem model
    A = vcat(hcat(zeros(T, 1, 2), tr'))
    b = ones(T, 1) 

    c = vcat(ones(T, 2), zeros(T, vni))

    G1 = hcat(one(T), zero(T)  , zeros(T, 1, vni))      # t_cond
    G2 = hcat(zeros(T, vnoe, 2), ikr_trB_VV)            # Y_qre
    G3 = hcat(zeros(T, vnoe, 2), VV)                    # X_qre
    G4 = hcat(zero(T), one(T)  , zeros(T, 1, vni))      # t_entr
    G5 = hcat(zero(T), zero(T) , zeros(T, 1, vni))      # y_entr
    G6 = hcat(zeros(T, vno, 2) , trE_VV)                # X_entr
    G = -vcat(G1, G2, G3, G4, G5, G6)

    h = zeros(T, 1 + 2*vnoe + 1 + 1 + vno)
    h[1 + 2*vnoe + 1 + 1] = 1.

    cones = [
        Cones.EpiTrRelEntropyTri{T, T}(1 + 2*vnoe), 
        Cones.EpiPerSepSpectral{Cones.MatrixCSqr{T, T}, T}(
            Cones.NegEntropySSF(),
            no,
        ),
    ]

    return Hypatia.Models.Model{T}(c, A, b, G, h, cones)
end

function precompile()
    V = randStinespringOperator(T, 2, 2, 2)

    # Use quantum mutual information cone
    model = eacc_problem(2, 2, 2, V)
    solver = Solvers.Solver{T}(verbose = true, iter_limit = 2, reduce = false, syssolver = ElimSystemSolver{T}())
    Solvers.load(solver, model)
    Solvers.solve(solver)

    # Use quantum conditional entropy cone
    model = eacc_qce_problem(2, 2, 2, V)
    solver = Solvers.Solver{T}(verbose = true, iter_limit = 2)
    Solvers.load(solver, model)
    Solvers.solve(solver)

    # Use quantum relative entropy cone
    model = eacc_qre_problem(2, 2, 2, V)
    solver = Solvers.Solver{T}(verbose = true, iter_limit = 2)
    Solvers.load(solver, model)
    Solvers.solve(solver)
end

function main(csv_name::String, all_tests::Bool)
    # Solve entanglement assisted channel capacity
    #   max  -S(B|BE)_VXV' + S(B)_VXV'
    #   s.t. tr[X] = 1
    #        X ⪰ 0
    
    test_set = [2; 4]
    if all_tests
        test_set = [test_set; 8; 16; 32; 64]
    end

    problem = "cc_ea"
    
    # Precompile with small problem
    precompile()
    
    # Loop through all the problems
    for test in test_set
        (ni, no, ne) = test
        description = string(ni) * "_" * string(no) * "_" * string(ne)

        # Generate random problem data
        Random.seed!(1)
        V = randStinespringOperator(T, ni, no, ne)

        # Use quantum mutual information cone
        model = eacc_problem(ni, no, ne, V)
        solver = Solvers.Solver{T}(verbose = true, reduce = false, syssolver = ElimSystemSolver{T}())
        try_solve(model, solver, problem, description, "QMI", csv_name)

        # Use quantum conditional entropy cone
        model = eacc_qce_problem(ni, no, ne, V)
        solver = Solvers.Solver{T}(verbose = true)
        try_solve(model, solver, problem, description, "QCE", csv_name)

        # Use quantum relative entropy cone
        model = eacc_qre_problem(ni, no, ne, V)
        solver = Solvers.Solver{T}(verbose = true)
        try_solve(model, solver, problem, description, "QRE", csv_name)
    end

end