using LinearAlgebra
import MAT

import Hypatia
import Hypatia.Cones
import Hypatia.Solvers

include("../cones/quantkeyrate.jl")
include("../cones/quantcondentr.jl")
include("../systemsolvers/elim.jl")
include("../utils/linear.jl")
include("../utils/helper.jl")

T = Float64

function qkd_problem(
    K_list::Vector{Matrix{R}},
    Z_list::Vector{Matrix{T}},
    Γ::Vector{Matrix{R}},
    γ::Vector{T},
    protocol::Union{String, Nothing} = nothing
) where {T <: Real, R <: Hypatia.RealOrComplex{T}}
    # Build quantum rate distortion problem
    #   min  t
    #   s.t. A(X) = b
    #        (t, X) ∈ K_qkd ≔ { (t, X) : t > S( G(X) || Z(G(X)) ), X⪰0 }

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

    cones = [QuantKeyRate{T, R}(K_list, Z_list, protocol)]

    return Hypatia.Models.Model{T}(c, A, b, G, h, cones)
end

function qkd_naive_problem(
    K_list::Vector{Matrix{R}},
    Z_list::Vector{Matrix{T}},
    Γ::Vector{Matrix{R}},
    γ::Vector{T}
) where {T <: Real, R <: Hypatia.RealOrComplex{T}}
    # Build quantum rate distortion problem
    #   min  t
    #   s.t. A(X) = b
    #        (t, G(X), Z(G(X))) ∈ K_qre
    #        X ⪰ 0

    ni  = size(K_list[1], 2)    # Input dimension
    nc  = size(γ, 1)
    vni = Cones.svec_length(R, ni)

    ZK_list = [Z * K for K in K_list for Z in Z_list]
    ZK_list_fr, K_list_fr = facial_reduction(ZK_list, K2_list=K_list)
    no = size(K_list_fr[1], 1)
    vno = Cones.svec_length(R, no)

    K_op  = lin_to_mat(R, x -> sum([K * x * K' for K in K_list_fr]), ni, no)
    ZK_op = lin_to_mat(R, x -> sum([ZK * x * ZK' for ZK in ZK_list_fr]), ni, no)
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


function precompile()
    # Get problem data for dprBB84 quantum key rate protocol
    f_name = "dprBB84_02_14_30"
    f = MAT.matopen("data/" * f_name * ".mat")
    data = MAT.read(f, "Data")

    if all(imag(data["Klist"][:]) == 0) && all(imag(data["Gamma"][:]) == 0)
        R = T
    else
        R = Complex{T}
    end
    
    K_list = convert(Vector{Matrix{R}}, data["Klist"][:])
    Z_list = convert(Vector{Matrix{T}}, data["Zlist"][:])
    Γ = convert(Vector{Matrix{R}}, data["Gamma"][:])
    γ = convert(Vector{T}, data["gamma"][:])

    # Use specialized dprBB4 oracle
    models = [
        qkd_problem(K_list, Z_list, Γ, γ, "dprBB84");
        qkd_problem(K_list, Z_list, Γ, γ, "dprBB84_naive");
        qkd_problem(K_list, Z_list, Γ, γ)
    ]
    for model in models
        solver = Solvers.Solver{T}(verbose = false, iter_limit = 2, reduce = false, syssolver = ElimSystemSolver{T}())
        Solvers.load(solver, model)
        Solvers.solve(solver)
    end

    # Use generic QRD oracle
    model = qkd_naive_problem(K_list, Z_list, Γ, γ)
    solver = Solvers.Solver{T}(verbose = false, iter_limit = 2)
    Solvers.load(solver, model)
    Solvers.solve(solver)
end

function main(csv_name::String, all_tests::Bool)
    # Solve for quantum key rate
    #   min  S( G(X) || Z(G(X)) )
    #   s.t. A(X) = b
    #        X ⪰ 0
    
    # Precompile with small problem
    precompile()
    
    main_dpr(csv_name, all_tests)
    main_dmcv(csv_name, all_tests)
end


function main_dpr(csv_name::String, all_tests::Bool)
    # dprBB84 examples
    test_set = [
        "dprBB84_02_14_30";
        "dprBB84_04_14_30"
    ]
    if all_tests
        test_set = [
            test_set;         
            "dprBB84_06_14_30";
            "dprBB84_08_14_30";
            "dprBB84_10_14_30";
        ]
    end

    problem = "qkd_dpr"
    
    # Loop through all the problems
    for test in test_set
        f_name = test
        description = parse(Int, split(f_name, "_")[2])

        # Generate problem data
        f = MAT.matopen("data/" * f_name * ".mat")
        data = MAT.read(f, "Data")
    
        if all(imag(data["Klist"][:]) == 0) && all(imag(data["Gamma"][:]) == 0)
            R = T
        else
            R = Complex{T}
        end

        K_list = convert(Vector{Matrix{R}}, data["Klist"][:])
        Z_list = convert(Vector{Matrix{T}}, data["Zlist"][:])
        Γ = convert(Vector{Matrix{R}}, data["Gamma"][:])
        γ = convert(Vector{T}, data["gamma"][:])        

        # Use specialized dprBB4 oracle
        model = qkd_problem(K_list, Z_list, Γ, γ, "dprBB84")
        solver = Solvers.Solver{T}(verbose = true, reduce = false, syssolver = ElimSystemSolver{T}())
        try_solve(model, solver, problem, description, "DPR", csv_name)

        # Use specialized QKD oracle with block diagonalization
        model = qkd_problem(K_list, Z_list, Γ, γ, "dprBB84_naive")
        solver = Solvers.Solver{T}(verbose = true, reduce = false, syssolver = ElimSystemSolver{T}())
        try_solve(model, solver, problem, description, "QKD", csv_name)

        # Use generic QRD oracle
        model = qkd_naive_problem(K_list, Z_list, Γ, γ)
        solver = Solvers.Solver{T}(verbose = true)
        try_solve(model, solver, problem, description, "QRE", csv_name)
    end

end

function main_dmcv(csv_name::String, all_tests::Bool)
    # DMCV examples
    test_set = ["DMCV_04_60_05_35"]
    if all_tests
        test_set = [
            test_set;  
            "DMCV_08_60_05_35";
            "DMCV_12_60_05_35";
            "DMCV_16_60_05_35";
            "DMCV_20_60_05_35";
        ]
    end

    problem = "qkd_dmcv"
    
    # Loop through all the problems
    for test in test_set
        f_name = test
        description = parse(Int, split(f_name, "_")[2])

        # Generate problem data
        f = MAT.matopen("data/" * f_name * ".mat")
        data = MAT.read(f, "Data")
    
        if all(imag(data["Klist"][:]) == 0) && all(imag(data["Gamma"][:]) == 0)
            R = T
        else
            R = Complex{T}
        end

        K_list = convert(Vector{Matrix{R}}, data["Klist"][:])
        Z_list = convert(Vector{Matrix{T}}, data["Zlist"][:])
        Γ = convert(Vector{Matrix{R}}, data["Gamma"][:])
        γ = convert(Vector{T}, data["gamma"][:])        

        # Use specialized QKD oracle with block diagonalization
        model = qkd_problem(K_list, Z_list, Γ, γ)
        solver = Solvers.Solver{T}(verbose = true, reduce = false, syssolver = ElimSystemSolver{T}())
        try_solve(model, solver, problem, description, "QKD", csv_name)

        # Use generic QRD oracle
        model = qkd_naive_problem(K_list, Z_list, Γ, γ)
        solver = Solvers.Solver{T}(verbose = true)
        try_solve(model, solver, problem, description, "QRE", csv_name)
    end

end
