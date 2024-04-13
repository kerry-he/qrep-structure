using LinearAlgebra

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

function purify(ρ::Matrix{T}) where {T <: Real}
    # Returns a purification of a quantum state
    n = size(ρ, 1)
    λ, U = eigen(ρ)

    vec = zeros(T, n*n, 1)
    for i in 1:n
        vec += sqrt(λ[i]) * kron(U[:, i], U[:, i])
    end

    return vec * vec'
end

function purify(λ::Vector{T}) where {T <: Real}
    # Returns a purification of a diagonal quantum state given its diagonal elements
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