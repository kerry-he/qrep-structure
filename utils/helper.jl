using LinearAlgebra

import Hypatia
import Hypatia.Cones
import Hypatia.Solvers

function lin_to_mat(
    R::Type,
    A, 
    ni::Int, 
    no::Int,
)
    # Construct the matrix representation for a given linear 
    # operator X->A(X) acting on symmetric or Hermitian matrices
    vni = Hypatia.Cones.svec_length(R, ni)
    vno = Hypatia.Cones.svec_length(R, no)
    mat = zeros(vno, vni)

    H_mat = zeros(R, ni, ni)
    rt2  = sqrt(2.0)

    for k in 1:vni
        # Get directional vector
        H = zeros(vni)
        H[k] = 1.
        Hypatia.Cones.svec_to_smat!(H_mat, H, rt2)
        LinearAlgebra.copytri!(H_mat, 'U', true)

        # Get column of matrix
        A_H = A(H_mat)
        @views Hypatia.Cones.smat_to_svec!(mat[:, k], A_H, rt2)
    end

    return mat
end

function pTr!(
    ptrX::Matrix{T}, 
    X::Matrix{T}, 
    sys::Int = 2, 
    dim::Union{Tuple{Int, Int}, Nothing} = nothing
) where {T <: Real}
    # Inplace partial trace operator
    if dim === nothing
        n1 = n2 = isqrt(size(X, 1))
    else
        n1 = dim[1]
        n2 = dim[2]
    end
    @assert n1*n2 == size(X, 1) == size(X, 2)

    if sys == 2
        @assert size(ptrX, 1) == size(ptrX, 2) == n1
        ptrX .= 0
        @inbounds for i = 1:n1, j = 1:n1
            @views X_views = X[(i-1)*n2+1 : i*n2, (j-1)*n2+1 : j*n2]
            @inbounds for k = 1:n2
                ptrX[i, j] += X_views[k, k]
            end
            
        end
    else
        @assert size(ptrX, 1) == size(ptrX, 2) == n2
        ptrX .= 0
        @inbounds for i = 1:n1
            @. ptrX += X[(i-1)*n2+1 : i*n2, (i-1)*n2+1 : i*n2]
        end
    end

    return ptrX
end

function pTr(
    X::Matrix{T}, 
    sys::Int = 2, 
    dim::Union{Tuple{Int, Int}, Nothing} = nothing
) where {T <: Real}
    # Partial trace operator
    if sys == 2
        return pTr!(zeros(T, dim[1], dim[1]), X, sys, dim)
    else
        return pTr!(zeros(T, dim[2], dim[2]), X, sys, dim)
    end
end

function idKron!(
    kronI::Matrix{T}, 
    X::Matrix{T}, 
    sys::Int = 2, 
    dim::Union{Tuple{Int, Int}, Nothing} = nothing
) where {T <: Real}
    # Inplace Kroneker product input matrix X with the identity matrix
    if dim === nothing
        n1 = n2 = isqrt(size(kronI, 1))
    else
        n1 = dim[1]
        n2 = dim[2]
    end
    @assert n1*n2 == size(kronI, 1) == size(kronI, 2)

    if sys == 2
        @assert size(X, 1) == size(X, 2) == n1
        kron!(kronI, X, Matrix{T}(I, n2, n2))
    else
        @assert size(X, 1) == size(X, 2) == n2
        kron!(kronI, Matrix{T}(I, n1, n1), X)
    end

    return kronI
end

function idKron(
    X::Matrix{T}, 
    sys::Int = 2, 
    dim::Union{Tuple{Int, Int}, Nothing} = nothing
) where {T <: Real}
    # Kroneker product input matrix X with the identity matrix
    n = dim[1] * dim[2]
    return idKron!(zeros(T, n, n), X, sys, dim)
end

function randDensityMatrix(R::Type, n::Int)
    # Generate random density matrix on Haar measure
    X = randn(R, n, n)
    rho = X * X'
    return rho / tr(rho)
end

function randUnitary(R::Type, n::Int)
    # Random unitary uniformly distributed on Haar measure
    # See https://case.edu/artsci/math/esmeckes/Meckes_SAMSI_Lecture2.pdf
    X = randn(R, n, n)
    U, _ = qr(X)
    return U
end

function randStinespringOperator(R::Type, nin::Int, nout::Int, nenv::Int)
    # Random Stinespring operator uniformly distributed on Hilbert-Schmidt measure
    # See https://arxiv.org/abs/2011.02994
    U = randUnitary(R, nout * nenv)
    return U[:, 1:nin]
end

function randDegradableChannel(R::Type, nin::Int, nout::Int, nenv::Int)
    # Random degradable channel, represented as a Stinespring isometry
    # Returns both Stinespring isometry V such that
    #     N(X)  = Tr_2[VXV']
    #     Nc(X) = Tr_1[VXV']
    # Also returns Stinespring isometry W such that
    #     Nc(X) = Tr_2[WN(X)W']
    # See https://arxiv.org/abs/0802.1360

    @assert nenv <= nin

    V = zeros(T, nout*nenv, nin)    # N Stinespring isometry
    W = zeros(T, nin*nenv, nout)    # Ξ Stinespring isometry

    U = randUnitary(R, nin)
    for k in 1:nout
        # Generate random vector
        v = randn(R, nenv)
        v /= norm(v)

        # Make Kraus operator and insert into N Stinespring isometry
        K = v * U[k, :]'
        @views V[(k - 1)*nenv + 1:k*nenv, :] = K

        # Make Kraus operator and insert into Ξ Stinespring isometry
        @views W[k:nin:end, k] = v
    end

    return V, W
end

function purify(rho::Matrix{Hypatia.RealOrComplex{T}}) where {T <: Real}
    # Returns a purification of a quantum state
    n = size(X, 1)
    λ, U = eigen(rho)

    vec = zeros(R, n*n, 1)
    for i in 1:n
        vec += sqrt(λ[i]) * kron(U[:, i], U[:, i])
    end

    return vec * vec'
end

function purify(λ::Vector{T}) where {T <: Real}
    # Returns a purification of a quantum state
    n = length(λ)
    vec = zeros(T, n*n)
    vec[collect(1:n+1:n^2)] .= sqrt.(λ)
    return vec * vec'
end

function entr(X::Matrix)
    λ = eigvals(Hermitian(X, :U))
    λ = λ[λ .> 0]
    return sum(λ .* log.(λ))
end

function entr(λ::Vector{T}) where {T <: Real}
    λ = λ[λ .> 0]
    return sum(λ .* log.(λ))
end

function facial_reduction(
    K_list::Vector{Matrix{R}};
    K2_list::Union{Vector{Matrix{R}}, Nothing} = nothing,
) where {T <: Real, R <: Hypatia.RealOrComplex{T}}
    # For a set of Kraus operators i.e., SUM_i K_i @ X @ K_i.T, returns a set of
    # Kraus operators which preserves positive definiteness
    nk = size(K_list[1], 1)

    # Pass identity matrix (maximally mixed state) through the Kraus operators
    KK = sum([K * K' for K in K_list])

    # Determine if output is low rank, in which case we need to perform facial reduction
    Dkk, Ukk = eigen(Hermitian(KK, :U))
    KKnzidx = Dkk .> 1e-12
    nk_fr = sum(KKnzidx)

    if nk == nk_fr
        if isnothing(K2_list)
            return K_list
        else
            return K_list, K2_list
        end
    end
    
    # Perform facial reduction
    Qkk = Ukk[:, KKnzidx]
    K_list_fr = [Qkk' * K for K in K_list]
    
    if isnothing(K2_list)
        return K_list_fr
    else
        # Do simultaneous facial reduction if second argument is provided
        K2_list_fr = [Qkk' * K for K in K2_list]
        return K_list_fr, K2_list_fr
    end
end

function print_statistics(solver)
    worst_gap = min(solver.gap / solver.point.tau[], abs(solver.primal_obj_t - solver.dual_obj_t))
    max_tau_obj = max(solver.point.tau[], min(abs(solver.primal_obj_t), abs(solver.dual_obj_t)))
    println("solve_time: \t", Solvers.get_solve_time(solver) - solver.time_rescale - solver.time_initx - solver.time_inity)
    println("no_iter: \t", Solvers.get_num_iters(solver))
    println("abs_gap: \t", solver.gap)
    println("rel_gap: \t", worst_gap / max_tau_obj)
    println("worst_feas: \t", max(solver.x_feas, solver.y_feas, solver.z_feas))

    print()
end