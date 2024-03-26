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
        @views mat_k = mat[:, k]
        H = zeros(vni)
        H[k] = 1.
        Hypatia.Cones.svec_to_smat!(H_mat, H, rt2)

        # Get column of matrix
        A_H = A(H_mat)
        Hypatia.Cones.smat_to_svec!(mat_k, A_H, rt2)
    end

    return mat
end

function pTr!(
    ptrX::Matrix{T}, 
    X::Matrix{T}, 
    sys::Int = 2, 
    dim::Union{Tuple{Int, Int}, Nothing} = nothing
) where {T <: Real}
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

function idKron!(
    kronI::Matrix{T}, 
    X::Matrix{T}, 
    sys::Int = 2, 
    dim::Union{Tuple{Int, Int}, Nothing} = nothing
) where {T <: Real}
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

function randDensityMatrix(R::Type, n::Int)
    # Generate random density matrix on Haar measure
    X = randn(R, n, n)
    rho = X * X'
    return rho / tr(rho)
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
    λ = eigvals(X)
    λ = λ[λ .> 0]
    return sum(λ .* log.(λ))
end

function entr(λ::Vector{T}) where {T <: Real}
    λ = λ[λ .> 0]
    return sum(λ .* log.(λ))
end