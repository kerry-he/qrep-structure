using LinearAlgebra

function Δ2_log!(Δ2::Matrix{T}, λ::Vector{T}, log_λ::Vector{T}) where {T <: Real}
    # Create first divided differences matrix of f(x)=log(x)
    rteps = sqrt(eps(T))
    d = length(λ)

    @inbounds for j in 1:d
        λ_j = λ[j]
        lλ_j = log_λ[j]
        for i in 1:(j - 1)
            λ_i = λ[i]
            λ_ij = λ_i - λ_j
            if abs(λ_ij) < rteps
                Δ2[i, j] = 2 / (λ_i + λ_j)
            elseif (λ_i < λ_j / 2) || (λ_j < λ_i / 2)
                Δ2[i, j] = (log_λ[i] - lλ_j) / λ_ij
            else
                z = λ_ij / (λ_i + λ_j)
                Δ2[i, j] = 2 * atanh(z) / λ_ij
            end
        end
        Δ2[j, j] = inv(λ_j)
    end

    # make symmetric
    LinearAlgebra.copytri!(Δ2, 'U', true)
    return Δ2
end

function Δ3_log!(Δ3::Array{T, 3}, Δ2::Matrix{T}, λ::Vector{T}) where {T <: Real}
    # Create second divided differences matrix of f(x)=log(x)
    @assert issymmetric(Δ2) # must be symmetric (wrapper is less efficient)
    rteps = sqrt(eps(T))
    d = length(λ)

    @inbounds for k in 1:d, j in 1:k, i in 1:j
        λ_j = λ[j]
        λ_k = λ[k]
        λ_jk = λ_j - λ_k
        if abs(λ_jk) < rteps
            λ_i = λ[i]
            λ_ij = λ_i - λ_j
            if abs(λ_ij) < rteps
                t = abs2(3 / (λ_i + λ_j + λ_k)) / -2
            else
                t = (Δ2[i, j] - Δ2[j, k]) / λ_ij
            end
        else
            t = (Δ2[i, j] - Δ2[i, k]) / λ_jk
        end

        Δ3[i, j, k] =
            Δ3[i, k, j] = Δ3[j, i, k] = Δ3[j, k, i] = Δ3[k, i, j] = Δ3[k, j, i] = t
    end

    return Δ3
end

function Δ2_frechet!(
    Δ2_F::Matrix{T}, 
    S_Δ3::Array{T, 3}, 
    U::Matrix{T}, 
    UHU::Matrix{T}, 
    mat::Matrix{T}, 
    mat2::Matrix{T}
) where {T <: Real}
    n = size(U, 1);

    @inbounds for k = 1:n
        @views mat[:, k] .= S_Δ3[:, :, k] * UHU[k, :];
    end
    mat .*= 2;
    spectral_outer!(Δ2_F, U, mat, mat2)

    return Δ2_F
end

function spectral_outer!(
    mat::AbstractMatrix{T},
    vecs::Union{Matrix{T}, Adjoint{T, Matrix{T}}},
    symm::Matrix{T},
    temp::Matrix{T},
) where {T <: Real}
    mul!(temp, vecs, symm)
    mul!(mat, temp, vecs')
    return mat
end

function spectral_outer!(
    mat::AbstractMatrix{T},
    vecs::Union{Matrix{T}, Adjoint{T, Matrix{T}}},
    diag::AbstractVector{T},
    temp::Matrix{T},
) where {T <: Real}
    mul!(temp, vecs, Diagonal(diag))
    mul!(mat, temp, vecs')
    return mat
end

function Δ2_frechet(
    S_Δ3::Array{T, 3}, 
    U::Matrix{T}, 
    UHU::Matrix{T}, 
) where {T <: Real}
    out = zeros(T, size(U))

    @inbounds for k in axes(S_Δ3, 3)
        @views out[:, k] .= S_Δ3[:, :, k] * UHU[k, :];
    end
    out .*= 2;

    return U * out * U'
end


function frechet_matrix!(
    out::AbstractMatrix{T}, 
    U::Matrix{R}, 
    D1::Matrix{T}, 
    temp::Matrix{T}, 
    c::Union{T, Bool},
    K_list=nothing
) where {T <: Real, R <: Hypatia.RealOrComplex{T}}
    # Build matrix corresponding to linear map H -> U @ [D1 * (U' @ H @ U)] @ U'
    KU_list = isnothing(K_list) ? [U] : [K' * U for K in K_list]
    D1_rt2 = sqrt.(D1)
    
    (n, m) = size(KU_list[1])
    rt2 = sqrt(2.)

    UHU = zeros(R, m, m)
    temp2 = zeros(R, m, m)

    k = 1
    @inbounds for j in 1:n
        @inbounds for i in 1:j-1
            UHU .= 0
            for KU in KU_list
                mul!(UHU, KU'[:, i], transpose(KU[j, :]), true, true)
            end
            UHU .*= D1_rt2 ./ rt2
            @. temp2 = UHU + UHU'
            @views Hypatia.Cones.smat_to_svec!(temp[k, :], temp2, rt2)
            k += 1

            if R == Complex{T}
                @. UHU *= -1im
                @. temp2 = UHU + UHU'          
                @views Hypatia.Cones.smat_to_svec!(temp[k, :], temp2, rt2)
                k += 1
            end
        end

        UHU .= 0
        for KU in KU_list
            mul!(UHU, KU'[:, j], transpose(KU[j, :]), true, true)
        end
        UHU .*= D1_rt2
        @views Hypatia.Cones.smat_to_svec!(temp[k, :], UHU, rt2)
        k += 1
    end

    return syrk_wrapper!(out, temp, c)
end

function syrk_wrapper!(C::Matrix{T}, A::Matrix{T}, alpha::Union{T, Bool}) where {T <: Real}
    return LinearAlgebra.BLAS.syrk!('U', 'N', alpha, A, true, C)
end

function syrk_wrapper!(C::AbstractMatrix{T}, A::Matrix{T}, alpha::Union{T, Bool}) where {T <: Real}
    # BLAS doesn't support views of matrices, so need to work around this
    temp = LinearAlgebra.BLAS.syrk('U', 'N', alpha, A)
    @inbounds for j in axes(C, 1), i in 1:j
        C[i, j] += temp[i, j]
    end
    return C
end

function nonsymm_kron!(
    out::AbstractMatrix{T},
    X::Matrix{R},
    rt2::T
) where {T <: Real, R <: Hypatia.RealOrComplex{T}}
    # Build matrix corresponding to linear map H -> X H X'
    n   = size(X, 1)

    temp = zeros(R, n, n)

    k = 1
    @inbounds for j in 1:n
        @inbounds for i in 1:j-1
            mul!(temp, X[:, i] ./ rt2, transpose(X'[j, :]))
            @views Hypatia.Cones.smat_to_svec!(out[:, k], temp .+ temp', rt2)
            k += 1

            if R == Complex{T}
                @. temp *= -1im
                @views Hypatia.Cones.smat_to_svec!(out[:, k], temp .+ temp', rt2)
                k += 1
            end
        end

        mul!(temp, X[:, j], transpose(X'[j, :]))
        @views Hypatia.Cones.smat_to_svec!(out[:, k], temp, rt2)
        k += 1
    end

    return out
end