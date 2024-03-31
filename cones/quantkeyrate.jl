using LinearAlgebra
import Hypatia.Cones

include("../utils/helper.jl")

mutable struct QuantKeyRate{T <: Real, R <: Hypatia.RealOrComplex{T}} <: Hypatia.Cones.Cone{T}
    use_dual_barrier::Bool
    dim::Int

    ni::Int
    vni::Int
    X_idxs::UnitRange{Int}
    K_list_blk::Vector{Vector{Matrix{R}}}
    ZK_list_blk::Vector{Vector{Matrix{R}}}
    protocol::Union{String, Nothing}

    N::Matrix{T}
    Nc::Matrix{T}
    tr::Vector{T}

    point::Vector{T}
    dual_point::Vector{T}
    fval::T
    grad::Vector{T}
    dder3::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool

    rt2::T
    X::Matrix{R}
    GX_blk::Vector{Matrix{R}}
    ZGX_blk::Vector{Matrix{R}}
    trX::T
    X_fact::Eigen{R}
    GX_fact_blk::Vector{Eigen{R}}
    ZGX_fact_blk::Vector{Eigen{R}}
    Xi::Matrix{R}
    Λx_log::Vector{T}
    Λgx_log_blk::Vector{Vector{T}}
    Λzgx_log_blk::Vector{Vector{T}}

    Λnx_log::Vector{T}
    Λncx_log::Vector{T}
    X_log::Matrix{R}
    GX_log_blk::Vector{Matrix{R}}
    ZGX_log_blk::Vector{Matrix{R}}
    z::T

    Δ2x_log::Matrix{T}
    Δ2gx_log_blk::Vector{Matrix{T}}
    Δ2zgx_log_blk::Vector{Matrix{T}}
    Δ2x_comb::Matrix{T}
    Δ3x_log::Array{T, 3}
    Δ3gx_log_blk::Vector{Array{T, 3}}
    Δ3zgx_log_blk::Vector{Array{T, 3}}

    DPhi::Vector{T}
    hess::Matrix{T}
    hess_fact::Cholesky{T, Matrix{T}}

    Hx::Matrix{R}

    hessprod_aux_updated::Bool
    invhessprod_aux_updated::Bool
    dder3_aux_updated::Bool

    vec1::Vector{T}
    vec2::Vector{T}

    function QuantKeyRate{T, R}(
        K_list::Vector{Matrix{R}},
        Z_list::Vector{Matrix{T}};
        protocol::Union{String, Nothing} = nothing,
        use_dual::Bool = false,
    ) where {T <: Real, R <: Hypatia.RealOrComplex{T}}
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual

        # Get dimensions of input, output, and environment
        cone.ni = size(K_list[1], 2)    # Input dimension
        cone.protocol = protocol        # Environment dimension
        
        cone.vni = Hypatia.Cones.svec_length(R, cone.ni)
        cone.dim = 1 + cone.vni

        cone.K_list_blk  = [facial_reduction(K_list)]
        cone.ZK_list_blk = [facial_reduction([K[first.(Tuple.(findall(!iszero, Z)[:, 1])), :] for K in K_list]) for Z in Z_list]

        return cone
    end
end

function Hypatia.Cones.reset_data(cone::QuantKeyRate)
    return (
        cone.feas_updated =
            cone.grad_updated =
                cone.hess_updated =
                    cone.inv_hess_updated =
                        cone.hess_fact_updated = 
                            cone.hessprod_aux_updated = 
                                cone.invhessprod_aux_updated = cone.dder3_aux_updated = false
    )
end

function Hypatia.Cones.setup_extra_data!(
    cone::QuantKeyRate{T, R},
) where {T <: Real, R <: Hypatia.RealOrComplex{T}}
    ni = cone.ni
    dim = cone.dim

    cone.rt2 = sqrt(T(2))
    cone.X_idxs = 2:dim

    cone.X = zeros(T, ni, ni)
    cone.Xi = zeros(T, ni, ni)
    cone.Λx_log = zeros(T, ni)

    cone.DPhi = zeros(T, cone.vni)
    cone.hess = zeros(T, cone.vni, cone.vni)

    cone.Hx = zeros(T, ni, ni)

    return cone
end

Hypatia.Cones.get_nu(cone::QuantKeyRate) = 1 + cone.ni

function Hypatia.Cones.set_initial_point!(
    arr::AbstractVector{T},
    cone::QuantKeyRate{T, R},
) where {T <: Real, R <: Hypatia.RealOrComplex{T}}
    GG_blk  = [congr(1.0I, K_list)  for K_list  in cone.K_list_blk]
    ZGGZ_blk = [congr(1.0I, ZK_list) for ZK_list in cone.ZK_list_blk]

    entr_GX  = sum([entr(GG) for GG in GG_blk])
    entr_ZGX = sum([entr(ZGGZ) for ZGGZ in ZGGZ_blk])

    arr[1] = 1.0 + (entr_GX - entr_ZGX)
    X0 = Matrix{R}(I, cone.ni, cone.ni)
    @views Hypatia.Cones.smat_to_svec!(arr[cone.X_idxs], X0, cone.rt2)
    return arr
end

function Hypatia.Cones.update_feas(cone::QuantKeyRate{T, R}) where {T <: Real, R <: Hypatia.RealOrComplex{T}}
    @assert !cone.feas_updated
    cone.is_feas = false

    # Compute required maps
    point = cone.point
    @views Hypatia.Cones.svec_to_smat!(cone.X, point[cone.X_idxs], cone.rt2)
    LinearAlgebra.copytri!(cone.X, 'U', true)
    cone.GX_blk  = [congr(cone.X, K_list)  for K_list  in cone.K_list_blk]
    cone.ZGX_blk = [congr(cone.X, ZK_list) for ZK_list in cone.ZK_list_blk]

    XH = Hermitian(cone.X, :U)

    if isposdef(XH)
        X_fact = cone.X_fact = eigen(XH)
        GX_fact_blk = cone.GX_fact_blk = [eigen(Hermitian(X, :U)) for X in cone.GX_blk]
        ZGX_fact_blk = cone.ZGX_fact_blk = [eigen(Hermitian(X, :U)) for X in cone.ZGX_blk]

        if isposdef(X_fact) && all(isposdef.(GX_fact_blk)) && all(isposdef.(ZGX_fact_blk))
            Λgx_blk  = [fact.values for fact in GX_fact_blk]
            Ugx_blk  = [fact.vectors for fact in GX_fact_blk]
            cone.Λgx_log_blk = [log.(λgx) for λgx in Λgx_blk]
            cone.GX_log_blk  = [U * diagm(Λ) * U' for (U, Λ) in zip(Ugx_blk, cone.Λgx_log_blk)]

            Λzgx_blk = [fact.values for fact in ZGX_fact_blk]
            Uzgx_blk = [fact.vectors for fact in ZGX_fact_blk] 
            cone.Λzgx_log_blk = [log.(λgx) for λgx in Λzgx_blk]
            cone.ZGX_log_blk  = [U * diagm(Λ) * U' for (U, Λ) in zip(Uzgx_blk, cone.Λzgx_log_blk)]            

            entr_GX  = sum([dot(λ, λ_log) for (λ, λ_log) in zip(Λgx_blk, cone.Λgx_log_blk)])
            entr_ZGX = sum([dot(λ, λ_log) for (λ, λ_log) in zip(Λzgx_blk, cone.Λzgx_log_blk)])

            cone.z = point[1] - (entr_GX - entr_ZGX)

            cone.is_feas = (cone.z > 0)

            if cone.is_feas
                cone.fval = -log(cone.z) - sum(log.(X_fact.values))
            end
        end
    end
    
    cone.feas_updated = true
    return cone.is_feas
end

function Hypatia.Cones.update_grad(cone::QuantKeyRate)
    @assert cone.is_feas
    rt2 = cone.rt2
    (Λx, Ux) = cone.X_fact

    cone.Xi .= Ux * diagm(inv.(Λx)) * Ux'

    G_log_GX   = sum([congr(X_log, Klist, true) for (X_log, Klist) in zip(cone.GX_log_blk, cone.K_list_blk)])
    ZG_log_ZGX = sum([congr(X_log, Klist, true) for (X_log, Klist) in zip(cone.ZGX_log_blk, cone.ZK_list_blk)])

    zi = inv(cone.z)
    Hypatia.Cones.smat_to_svec!(cone.DPhi, G_log_GX - ZG_log_ZGX, rt2)

    g = cone.grad
    g[1] = -zi
    @views Hypatia.Cones.smat_to_svec!(g[cone.X_idxs], -cone.Xi, rt2)
    g[cone.X_idxs] += zi * cone.DPhi

    cone.grad_updated = true
    return cone.grad
end

function update_hessprod_aux(cone::QuantKeyRate)
    @assert !cone.hessprod_aux_updated
    @assert cone.grad_updated

    Λgx_blk  = [fact.values for fact in cone.GX_fact_blk]
    Λzgx_blk = [fact.values for fact in cone.ZGX_fact_blk]

    # Compute first divideded differences matrix
    cone.Δ2gx_log_blk  = [Δ2_log!(zeros(T, length(D), length(D)), D, D_log) for (D, D_log) in zip(Λgx_blk,  cone.Λgx_log_blk)]
    cone.Δ2zgx_log_blk = [Δ2_log!(zeros(T, length(D), length(D)), D, D_log) for (D, D_log) in zip(Λzgx_blk,  cone.Λzgx_log_blk)]
 
    cone.hessprod_aux_updated = true
    return
end


function Hypatia.Cones.hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::QuantKeyRate{T, R},
) where {T <: Real, R <: Hypatia.RealOrComplex{T}}
    @assert cone.grad_updated
    cone.hessprod_aux_updated || update_hessprod_aux(cone)

    rt2 = cone.rt2
    zi = 1 / cone.z

    Ugx_blk  = [fact.vectors for fact in cone.GX_fact_blk]
    Uzgx_blk = [fact.vectors for fact in cone.ZGX_fact_blk] 

    Hx = cone.Hx

    @inbounds for j in axes(arr, 2)

        # Get input direction
        Ht = arr[1, j]
        @views Hx_vec = arr[cone.X_idxs, j]
        @views Hypatia.Cones.svec_to_smat!(Hx, Hx_vec, rt2)
        LinearAlgebra.copytri!(Hx, 'U', true)

        KH_blk  = [congr(Hx, K_list)  for K_list  in cone.K_list_blk]
        ZKH_blk = [congr(Hx, ZK_list) for ZK_list in cone.ZK_list_blk]

        UkKHUk_blk    = [U' * H * U for (H, U) in zip(KH_blk, Ugx_blk)]
        UkzZKHUkz_blk = [U' * H * U for (H, U) in zip(ZKH_blk, Uzgx_blk)]

        # Hessian product of quantum entropies
        D2PhiH  = sum([congr(U * (D1 .* UHU) * U', K_list, true) for (U, D1, UHU, K_list)
                 in zip(Ugx_blk, cone.Δ2gx_log_blk, UkKHUk_blk, cone.K_list_blk)])
        D2PhiH -= sum([congr(U * (D1 .* UHU) * U', K_list, true) for (U, D1, UHU, K_list)
                 in zip(Uzgx_blk, cone.Δ2zgx_log_blk, UkzZKHUkz_blk, cone.ZK_list_blk)])

        prodt = zi * zi * (Ht - dot(Hx_vec, cone.DPhi))
        prodX = -prodt * cone.DPhi + Hypatia.Cones.smat_to_svec!(zeros(T, cone.vni), zi * D2PhiH + cone.Xi * Hx * cone.Xi, cone.rt2)

        prod[1, j] = prodt
        prod[cone.X_idxs, j] = prodX

    end

    return prod
end

function update_invhessprod_aux(cone::QuantKeyRate)
    @assert !cone.invhessprod_aux_updated
    @assert cone.grad_updated
    @assert cone.hessprod_aux_updated

    rt2 = cone.rt2
    zi = 1 / cone.z

    Ugx_blk  = [fact.vectors for fact in cone.GX_fact_blk]
    Uzgx_blk = [fact.vectors for fact in cone.ZGX_fact_blk] 

    # Default computation of QKD Hessian
    cone.hess  = kronecker_matrix(cone.Xi)
    cone.hess += sum([frechet_matrix(U, D1, K_list) * zi
                    for (U, D1, K_list) in zip(Ugx_blk, cone.Δ2gx_log_blk, cone.K_list_blk)])
    cone.hess -= sum([frechet_matrix(U, D1, K_list) * zi
                    for (U, D1, K_list) in zip(Uzgx_blk, cone.Δ2zgx_log_blk, cone.ZK_list_blk)])

    # Rescale and factor Hessian
    cone.hess_fact = cholesky(Hermitian(cone.hess))

    cone.invhessprod_aux_updated = true
    return
end

function Hypatia.Cones.inv_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::QuantKeyRate{T, R},
) where {T <: Real, R <: Hypatia.RealOrComplex{T}}
    @assert cone.grad_updated
    cone.hessprod_aux_updated || update_hessprod_aux(cone)
    cone.invhessprod_aux_updated || update_invhessprod_aux(cone)

    @inbounds for j in axes(arr, 2)

        # Get input direction
        Ht = arr[1, j]
        @views Hx = arr[cone.X_idxs, j]

        # Solve linear system
        @views prod[cone.X_idxs, j] = cone.hess_fact \ (Hx + Ht*cone.DPhi)
        prod[1, j] = cone.z * cone.z * Ht + dot(prod[cone.X_idxs, j], cone.DPhi)

    end
        
    return prod
end

function update_dder3_aux(cone::QuantKeyRate)
    @assert !cone.dder3_aux_updated
    @assert cone.hessprod_aux_updated

    Λgx_blk  = [fact.values for fact in cone.GX_fact_blk]
    Λzgx_blk = [fact.values for fact in cone.ZGX_fact_blk]

    # Compute first divideded differences matrix
    cone.Δ3gx_log_blk  = [Δ3_log!(zeros(T, length(D), length(D), length(D)), D1, D) for (D, D1) in zip(Λgx_blk,  cone.Δ2gx_log_blk)]
    cone.Δ3zgx_log_blk = [Δ3_log!(zeros(T, length(D), length(D), length(D)), D1, D) for (D, D1) in zip(Λzgx_blk, cone.Δ2zgx_log_blk)]

    cone.dder3_aux_updated = true
    return
end

function Hypatia.Cones.dder3(cone::QuantKeyRate{T, R}, dir::AbstractVector{T}) where {T <: Real, R <: Hypatia.RealOrComplex{T}}
    @assert cone.grad_updated
    cone.hessprod_aux_updated || update_hessprod_aux(cone)
    cone.dder3_aux_updated || update_dder3_aux(cone)

    rt2 = cone.rt2
    zi = 1 / cone.z

    Ugx_blk  = [fact.vectors for fact in cone.GX_fact_blk]
    Uzgx_blk = [fact.vectors for fact in cone.ZGX_fact_blk] 

    Hx = cone.Hx

    dder3 = cone.dder3

    # Get input direction
    Ht = dir[1]
    @views Hx_vec = dir[cone.X_idxs]
    @views Hypatia.Cones.svec_to_smat!(Hx, Hx_vec, rt2)
    LinearAlgebra.copytri!(Hx, 'U', true)

    KH_blk  = [congr(Hx, K_list)  for K_list  in cone.K_list_blk]
    ZKH_blk = [congr(Hx, ZK_list) for ZK_list in cone.ZK_list_blk]

    UkKHUk_blk    = [U' * H * U for (H, U) in zip(KH_blk, Ugx_blk)]
    UkzZKHUkz_blk = [U' * H * U for (H, U) in zip(ZKH_blk, Uzgx_blk)]


    # Derivatives of objective function
    D2PhiH  = sum([congr(U * (D1 .* UHU) * U', K_list, true) for (U, D1, UHU, K_list)
            in zip(Ugx_blk, cone.Δ2gx_log_blk, UkKHUk_blk, cone.K_list_blk)])
    D2PhiH -= sum([congr(U * (D1 .* UHU) * U', K_list, true) for (U, D1, UHU, K_list)
            in zip(Uzgx_blk, cone.Δ2zgx_log_blk, UkzZKHUkz_blk, cone.ZK_list_blk)])


    D3PhiHH  = sum([congr(Δ2_frechet(UHU .* D2, U, UHU), K_list, true) for (U, D2, UHU, K_list)
            in zip(Ugx_blk, cone.Δ3gx_log_blk, UkKHUk_blk, cone.K_list_blk)])
    D3PhiHH -= sum([congr(Δ2_frechet(UHU .* D2, U, UHU), K_list, true) for (U, D2, UHU, K_list)
            in zip(Uzgx_blk, cone.Δ3zgx_log_blk, UkzZKHUkz_blk, cone.ZK_list_blk)])


    # Third directional derivatives of the barrier function
    DPhiH = dot(cone.DPhi, Hx_vec)
    D2PhiHH = real(dot(D2PhiH, Hx))
    χ = Ht - DPhiH
    zi = 1 / cone.z

    @views dder3_X = dder3[cone.X_idxs]
    dder3[1]  = -(2 * zi^3 * χ^2) - (zi^2 * D2PhiHH)
    dder3_X  .= -dder3[1] * cone.DPhi
    temp      = -2 * zi^2 * χ * D2PhiH
    temp    .+= zi * D3PhiHH
    temp    .-= 2 * cone.Xi * Hx * cone.Xi * Hx * cone.Xi
    dder3_X .+= Hypatia.Cones.smat_to_svec!(zeros(T, cone.vni), temp, cone.rt2)

    # @. dder3 *= -0.5    

    return dder3
end

#-----------------------------------------------------------------------------------------------------

function Δ2_log!(Δ2::Matrix{T}, λ::Vector{T}, log_λ::Vector{T}) where {T <: Real}
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
            else
                Δ2[i, j] = (log_λ[i] - lλ_j) / λ_ij
            end
        end
        Δ2[j, j] = inv(λ_j)
    end

    # make symmetric
    LinearAlgebra.LinearAlgebra.LinearAlgebra.copytri!(Δ2, 'U', true)
    return Δ2
end

function Δ3_log!(Δ3::Array{T, 3}, Δ2::Matrix{T}, λ::Vector{T}) where {T <: Real}
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

function Δ2_frechet(
    S_Δ3::Array{R, 3}, 
    U::Matrix{R}, 
    UHU::Matrix{R}, 
) where {T <: Real, R <: Hypatia.RealOrComplex{T}}
    out = zeros(R, size(U))

    @inbounds for k in axes(S_Δ3, 3)
        @views out[:, k] .= S_Δ3[:, :, k] * UHU[k, :];
    end
    out .*= 2;

    return U * out * U'
end


function facial_reduction(K_list::Vector{Matrix{R}}) where {T <: Real, R <: Hypatia.RealOrComplex{T}}
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
        return K_list
    end
    
    # Perform facial reduction
    Qkk = Ukk[:, KKnzidx]
    K_list_fr = [Qkk' * K for K in K_list]

    return K_list_fr
end

function congr(x, K_list, adjoint = false)
    if adjoint
        return sum([K' * x * K for K in K_list])
    else
        return sum([K * x * K' for K in K_list])
    end
end

function frechet_matrix(U::Matrix{R}, D1::Matrix{T}, K_list=nothing) where {T <: Real, R <: Hypatia.RealOrComplex{T}}
    # Build matrix corresponding to linear map H -> U @ [D1 * (U' @ H @ U)] @ U'
    KU_list = isnothing(K_list) ? [U] : [K' * U for K in K_list]
    
    n   = size(KU_list[1], 1)
    vn  = Hypatia.Cones.svec_length(R, n)
    rt2 = sqrt(2.)
    out = zeros(T, vn, vn)

    k = 1
    for j in 1:n
        for i in 1:j-1
            UHU = sum([KU'[:, i] * transpose(KU[j, :]) for KU in KU_list]) / rt2
            D_H = congr(D1 .* UHU, KU_list)
            @views Hypatia.Cones.smat_to_svec!(out[:, k], D_H + D_H', rt2)
            k += 1

            if R == Complex{T}
                D_H *= -1im
                @views Hypatia.Cones.smat_to_svec!(out[:, k], D_H + D_H', rt2)
                k += 1
            end
        end

        UHU = sum([KU'[:, j] * transpose(KU[j, :]) for KU in KU_list])
        D_H = congr(D1 .* UHU, KU_list)
        @views Hypatia.Cones.smat_to_svec!(out[:, k], D_H, rt2)
        k += 1
    end

    return out
end

function kronecker_matrix(X::Matrix{R}) where {T <: Real, R <: Hypatia.RealOrComplex{T}}
    # Build matrix corresponding to linear map H -> X H X'
    n   = size(X, 1)
    vn  = Hypatia.Cones.svec_length(R, n)
    rt2 = sqrt(2.)
    out = zeros(T, vn, vn)

    k = 1
    for j in 1:n
        for i in 1:j-1
            temp = X[:, i] * transpose(X'[j, :]) / rt2
            @views Hypatia.Cones.smat_to_svec!(out[:, k], temp + temp', rt2)
            k += 1

            if R == Complex{T}
                temp *= -1im
                @views Hypatia.Cones.smat_to_svec!(out[:, k], temp + temp', rt2)
                k += 1
            end
        end

        temp = temp = X[:, j] * transpose(X'[j, :])
        @views Hypatia.Cones.smat_to_svec!(out[:, k], temp, rt2)
        k += 1
    end

    return out
end