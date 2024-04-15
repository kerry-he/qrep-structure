using LinearAlgebra

import Hypatia
import Hypatia.Cones
import Hypatia.Solvers

include("../cones/quantratedist.jl")
include("../cones/quantcondentr.jl")
include("../systemsolvers/elim.jl")
include("../utils/linear.jl")
include("../utils/quantum.jl")

import Random
Random.seed!(1)

T = Float64

function qrd_ef_problem(n::Int, λ::Vector{T}, D::Float64) where {T <: Real}
    # Build quantum rate distortion problem
    #   min  t
    #   s.t. Tr_2[G(y, Z)] = W
    #        (t, y, Z) ∈ K_qce
    #        ⟨Δ, G(y, Z)⟩ ≤ D

    m = n * (n - 1)
    vn = Cones.svec_length(n)

    # Construct matrix of indices
    indices = zeros(Int, n, n)
    temp = reshape(range(1, n*(n-1)), (n-1, n))
    indices[2:end, :] += tril(temp)
    indices[1:end-1, :] += triu(temp, 1)

    pTr_y = zeros(T, n, n*(n-1))
    pTr_X = zeros(T, n, vn)
    for i in 1:n
        idx = indices[i, :]
        deleteat!(idx, i)
        pTr_y[i, idx] .= 1.

        temp = zeros(T, n, n)
        temp[i, i] = 1.
        @views Hypatia.Cones.smat_to_svec!(pTr_X[i, :], temp, sqrt(2.))
    end

    Δ_vec = Cones.smat_to_svec!(zeros(T, vn), 1.0I - sqrt.(λ * λ'), sqrt(2.))


    # Build problem model
    A1 = hcat(zeros(T, n, 1), pTr_y        , pTr_X  , zeros(T, n, 1))        # Tr_B[X] = W
    A2 = hcat(zero(T)       , ones(T, 1, m), Δ_vec' , one(T))                # D = <Δ,X>
    A  = vcat(A1, A2)

    b = zeros(T, n + 1)
    b[1:n] .= λ
    b[end] = D

    c = vcat(one(T), zeros(T, m + vn + 1))

    G = -one(T) * I
    h = zeros(T, 2 + m + vn)

    cones = [QuantRateDistortion{T}(n), Cones.Nonnegative{T}(1)]

    return Hypatia.Models.Model{T}(c, A, b, G, h, cones, obj_offset=entr(λ))
end

function qrd_ef_qre_problem(n::Int, λ::Vector{T}, D::Float64) where {T <: Real}
    # Build quantum rate distortion problem
    #   min  t1 + t2
    #   s.t. Tr_2[G(y, Z)] = W
    #        (t1, y, G1(y, Z)) ∈ K_cre
    #        (t2, Z, G2(y, Z)) ∈ K_qre
    #        ⟨Δ, G(y, Z)⟩ ≤ D

    m = n * (n - 1)
    N = n * n
    vn = Cones.svec_length(n)
    vN = Cones.svec_length(N)

    # Construct matrix of indices
    indices = zeros(Int, n, n)
    temp = reshape(range(1, n*(n-1)), (n-1, n))
    indices[2:end, :] += tril(temp)
    indices[1:end-1, :] += triu(temp, 1)

    pTr_y = zeros(T, n, n*(n-1))
    pTr_X = zeros(T, n, vn)
    for i in 1:n
        idx = indices[i, :]
        deleteat!(idx, i)
        pTr_y[i, idx] .= 1.

        temp = zeros(T, n, n)
        temp[i, i] = 1.
        @views Hypatia.Cones.smat_to_svec!(pTr_X[i, :], temp, sqrt(2.))
    end

    Δ_vec = Cones.smat_to_svec!(zeros(T, vn), 1.0I - sqrt.(λ * λ'), sqrt(2.))

    G1_y = zeros(T, m, m)
    G1_X = zeros(T, m, vn)
    k = 1
    for i in 1:n, j in 1:n-1
        idx = indices[:, i]
        deleteat!(idx, i)
        G1_y[k, idx] .= 1.

        temp = zeros(T, n, n)
        temp[i, i] = 1.
        @views Hypatia.Cones.smat_to_svec!(G1_X[k, :], temp, sqrt(2.))
        
        k += 1
    end   

    G4_y = zeros(T, vn, m)
    G4_X = zeros(T, vn, vn)
    k = 1
    for j in 1:n
        for i in 1:j-1
            k += 1
        end
        idx = indices[:, j]
        deleteat!(idx, j)
        G4_y[k, idx] .= 1.

        temp = zeros(T, n, n)
        temp[j, j] = 1.
        @views Hypatia.Cones.smat_to_svec!(G4_X[k, :], temp, sqrt(2.))

        k += 1
    end

    k = 1
    p_k = 1
    blkdiag = zeros(T, vN, m + vn)
    for j in 1:N, i in 1:j
        G_y = zeros(T, 1, m)
        G_X = zeros(T, 1, vn)

        if ((i - 1) % (n + 1) == 0) && ((j - 1) % (n + 1) == 0)
            I = div(i - 1, n + 1) + 1
            J = div(j - 1, n + 1) + 1

            temp = zeros(T, n, n)
            temp[I, J] = temp[J, I] = if (I == J) 1.0 else sqrt(0.5) end
            @views Hypatia.Cones.smat_to_svec!(G_X[1, :], temp, sqrt(2.))
        elseif i == j
            G_y[1, p_k] = 1.
            p_k += 1
        end

        @views blkdiag[k, :] = hcat(G_y, G_X)
        k += 1
    end


    # Build problem model
    A = hcat(zeros(T, n, 2), pTr_y, pTr_X)        # Tr_B[X] = W
    b = λ

    c = vcat(ones(T, 2), zeros(T, m + vn))

    G0 = -hcat(one(T)      , zero(T)     , zeros(T, 1, m + vn))           # t_cre
    G1 = -hcat(zeros(T, m) , zeros(T, m) , G1_y, G1_X)                    # q_cre
    G2 = -hcat(zeros(T, m) , zeros(T, m) , 1.0I, zeros(T, m, vn))         # p_cre
    G3 = -hcat(zero(T)     , one(T)      , zeros(T, 1, m + vn))           # t_qre
    G4 = -hcat(zeros(T, vn), zeros(T, vn), G4_y, G4_X)                    # Y_qre
    G5 = -hcat(zeros(T, vn), zeros(T, vn), zeros(T, vn, m), 1.0I)         # X_qre
    G6 =  hcat(zero(T)     , zeros(T)    , ones(T, 1, m), Δ_vec')         # nn
    G  =  vcat(G0, G1, G2, G3, G4, G5, G6)

    h = zeros(T, 3 + 2*m + 2*vn)
    h[end] = D

    cones = [Cones.EpiRelEntropy{T}(1 + 2*m), Cones.EpiTrRelEntropyTri{T, T}(1 + 2*vn), Cones.Nonnegative{T}(1)]

    return Hypatia.Models.Model{T}(c, A, b, G, h, cones, obj_offset=entr(λ))
end


function qrd_ef_qce_problem(n::Int, λ::Vector{T}, D::Float64) where {T <: Real}
    # Build quantum rate distortion problem
    #   min  t
    #   s.t. Tr_2[G(y, Z)] = W
    #        (t, G(y, Z)) ∈ K_qce
    #        ⟨Δ, G(y, Z)⟩ ≤ D

    m = n * (n - 1)
    N = n * n
    vn = Cones.svec_length(n)
    vN = Cones.svec_length(N)

    # Construct matrix of indices
    indices = zeros(Int, n, n)
    temp = reshape(range(1, n*(n-1)), (n-1, n))
    indices[2:end, :] += tril(temp)
    indices[1:end-1, :] += triu(temp, 1)

    pTr_y = zeros(T, n, n*(n-1))
    pTr_X = zeros(T, n, vn)
    for i in 1:n
        idx = indices[i, :]
        deleteat!(idx, i)
        pTr_y[i, idx] .= 1.

        temp = zeros(T, n, n)
        temp[i, i] = 1.
        @views Hypatia.Cones.smat_to_svec!(pTr_X[i, :], temp, sqrt(2.))
    end

    Δ_vec = Cones.smat_to_svec!(zeros(T, vn), 1.0I - sqrt.(λ * λ'), sqrt(2.))

    k = 1
    p_k = 1
    blkdiag = zeros(T, vN, m + vn)
    for j in 1:N, i in 1:j
        G_y = zeros(T, 1, m)
        G_X = zeros(T, 1, vn)

        if ((i - 1) % (n + 1) == 0) && ((j - 1) % (n + 1) == 0)
            I = div(i - 1, n + 1) + 1
            J = div(j - 1, n + 1) + 1

            temp = zeros(T, n, n)
            temp[I, J] = temp[J, I] = if (I == J) 1.0 else sqrt(0.5) end
            @views Hypatia.Cones.smat_to_svec!(G_X[1, :], temp, sqrt(2.))
        elseif i == j
            G_y[1, p_k] = 1.
            p_k += 1
        end

        @views blkdiag[k, :] = hcat(G_y, G_X)
        k += 1
    end

    # Build problem model
    A = hcat(zeros(T, n, 1), pTr_y, pTr_X)        # Tr_B[X] = W
    b = λ

    c = vcat(one(T), zeros(T, m + vn))

    G0 = -hcat(one(T)      , zeros(T, 1, m + vn))        # t_qce
    G1 = -hcat(zeros(T, vN), blkdiag)                    # X_qce
    G2 =  hcat(zero(T)     , ones(T, 1, m), Δ_vec')      # nn
    G  =  vcat(G0, G1, G2)

    h = zeros(T, 2 + vN)
    h[end] = D

    cones = [QuantCondEntropy{T}(n, n, 2), Cones.Nonnegative{T}(1)]

    return Hypatia.Models.Model{T}(c, A, b, G, h, cones, obj_offset=entr(λ))
end


function main()
    # Solve quantum rate distortion problem with entaglement fidelity distortion
    #   min  S(B|BR)_G(y, Z) + S(W)
    #   s.t. Tr_B[G(y, Z)] = W
    #        ⟨Δ, G(y, Z)⟩ ≤ D
    #        y ≥ 0, Z ⪰ 0

    # Define rate distortion problem with entanglement fidelity distortion
    n = 4
    λ = eigvals(randDensityMatrix(T, n))
    D = 0.5
    
    # Use restriction of quantum conditional entropy cone to fixed point subspace
    model = qrd_ef_problem(n, λ, D)
    solver = Solvers.Solver{T}(reduce = false, syssolver = ElimSystemSolver{T}())
    Solvers.load(solver, model)
    Solvers.solve(solver)
    print_statistics(solver, "Quantum rate distortion w/ entanglement fidelity distortion", "QRD")

    # Use decomposition of relative entropy cones
    model = qrd_ef_qre_problem(n, λ, D)
    solver = Solvers.Solver{T}()
    Solvers.load(solver, model)
    Solvers.solve(solver)
    print_statistics(solver, "Quantum rate distortion w/ entanglement fidelity distortion", "QRE*")

    # Use quantum conditional entropy cone with linear constraints
    model = qrd_ef_qce_problem(n, λ, D)
    solver = Solvers.Solver{T}()
    Solvers.load(solver, model)
    Solvers.solve(solver)
    print_statistics(solver, "Quantum rate distortion w/ entanglement fidelity distortion", "QCE*")
end

main()