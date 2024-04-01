using LinearAlgebra
import Hypatia.Cones

include("../utils/helper.jl")

mutable struct QuantMutualInformation{T <: Real} <: Hypatia.Cones.Cone{T}
    use_dual_barrier::Bool
    dim::Int

    ni::Int
    no::Int
    ne::Int
    vni::Int
    vno::Int
    vne::Int
    X_idxs::UnitRange{Int}
    V::Matrix{T}

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
    X::Matrix{T}
    NX::Matrix{T}
    NcX::Matrix{T}
    trX::T
    X_fact::Eigen{T}
    NX_fact::Eigen{T}
    NcX_fact::Eigen{T}
    Xi::Matrix{T}
    Λx_log::Vector{T}
    Λnx_log::Vector{T}
    Λncx_log::Vector{T}
    X_log::Matrix{T}
    NX_log::Matrix{T}
    NcX_log::Matrix{T}
    trX_log::T
    z::T

    Δ2x_log::Matrix{T}
    Δ2nx_log::Matrix{T}
    Δ2ncx_log::Matrix{T}
    Δ2x_comb::Matrix{T}
    Δ3x_log::Array{T, 3}
    Δ3nx_log::Array{T, 3}
    Δ3ncx_log::Array{T, 3}

    DPhi::Vector{T}
    hess::Matrix{T}
    hess_fact

    Hx::Matrix{T}
    Hnx::Matrix{T}
    Hncx::Matrix{T}

    hessprod_aux_updated::Bool
    invhessprod_aux_updated::Bool
    dder3_aux_updated::Bool

    vec1::Vector{T}
    vec2::Vector{T}

    function QuantMutualInformation{T}(
        ni::Int,
        no::Int,
        ne::Int,
        V::Matrix{T};
        use_dual::Bool = false,
    ) where {T <: Real}
        cone = new{T}()
        cone.use_dual_barrier = use_dual

        # Get dimensions of input, output, and environment
        cone.ni = ni        # Input dimension
        cone.no = no        # Output dimension
        cone.ne = ne        # Environment dimension
        
        cone.vni = Hypatia.Cones.svec_length(ni)
        cone.vno = Hypatia.Cones.svec_length(no)
        cone.vne = Hypatia.Cones.svec_length(ne)
        cone.dim = 1 + cone.vni

        cone.V  = V         # Stinespring isometry
        @assert size(V) == (no*ne, ni)

        # Build linear maps of quantum channels
        cone.N  = lin_to_mat(T, x -> pTr!(zeros(T, no, no), V*x*V', 2, (no, ne)), ni, no)         # Quantum channel
        cone.Nc = lin_to_mat(T, x -> pTr!(zeros(T, ne, ne), V*x*V', 1, (no, ne)), ni, ne)         # Complementary channel
        cone.tr = Hypatia.Cones.smat_to_svec!(zeros(T, cone.vni), Matrix{T}(I, ni, ni), cone.rt2) # Trace operator

        return cone
    end
end

function Hypatia.Cones.reset_data(cone::QuantMutualInformation)
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
    cone::QuantMutualInformation{T},
) where {T <: Real}
    ni = cone.ni
    no = cone.no
    ne = cone.ne
    dim = cone.dim

    cone.rt2 = sqrt(T(2))
    cone.X_idxs = 2:dim

    cone.X = zeros(T, ni, ni)
    cone.NX = zeros(T, no, no)
    cone.NcX = zeros(T, ne, ne)
    cone.Xi = zeros(T, ni, ni)
    cone.Λx_log = zeros(T, ni)
    cone.Λnx_log = zeros(T, no)
    cone.Λncx_log = zeros(T, ne)
    cone.X_log = zeros(T, ni, ni)
    cone.NX_log = zeros(T, no, no)
    cone.NcX_log = zeros(T, ne, ne)

    cone.DPhi = zeros(T, cone.vni)
    cone.hess = zeros(T, cone.vni, cone.vni)

    cone.Δ2x_log = zeros(T, ni, ni)
    cone.Δ2nx_log = zeros(T, no, no)
    cone.Δ2ncx_log = zeros(T, ne, ne)
    cone.Δ2x_comb = zeros(T, ni, ni)
    cone.Δ3x_log = zeros(T, ni, ni, ni)
    cone.Δ3nx_log = zeros(T, no, no, no)
    cone.Δ3ncx_log = zeros(T, ne, ne, ne)

    cone.Hx = zeros(T, ni, ni)
    cone.Hnx = zeros(T, no, no)
    cone.Hncx = zeros(T, ne, ne)

    return cone
end

Hypatia.Cones.get_nu(cone::QuantMutualInformation) = 1 + cone.ni

function Hypatia.Cones.set_initial_point!(
    arr::AbstractVector{T},
    cone::QuantMutualInformation{T},
) where {T <: Real}
    arr[1] = 1.0
    X0 = Matrix{T}(I, cone.ni, cone.ni)
    @views Hypatia.Cones.smat_to_svec!(arr[cone.X_idxs], X0, cone.rt2)
    return arr
end

function Hypatia.Cones.update_feas(cone::QuantMutualInformation{T}) where {T <: Real}
    @assert !cone.feas_updated
    cone.is_feas = false

    # Compute required maps
    point = cone.point
    @views Hypatia.Cones.svec_to_smat!(cone.X, point[cone.X_idxs], cone.rt2)
    @views Hypatia.Cones.svec_to_smat!(cone.NX, cone.N * point[cone.X_idxs], cone.rt2)
    @views Hypatia.Cones.svec_to_smat!(cone.NcX, cone.Nc * point[cone.X_idxs], cone.rt2)
    cone.trX = tr(cone.X)

    LinearAlgebra.copytri!(cone.X, 'U')
    LinearAlgebra.copytri!(cone.NX, 'U')
    LinearAlgebra.copytri!(cone.NcX, 'U')

    XH   = Hermitian(cone.X, :U)
    NXH  = Hermitian(cone.NX, :U)
    NcXH = Hermitian(cone.NcX, :U)

    if isposdef(XH)
        X_fact = cone.X_fact = eigen(XH)
        NX_fact = cone.NX_fact = eigen(NXH)
        NcX_fact = cone.NcX_fact = eigen(NcXH)
        if isposdef(X_fact) && isposdef(NX_fact) && isposdef(NcX_fact)
            for (fact, λ_log, X_log) in zip(
                (X_fact, NX_fact, NcX_fact),
                (cone.Λx_log, cone.Λnx_log, cone.Λncx_log),
                (cone.X_log, cone.NX_log, cone.NcX_log)
            )
                (λ, U) = fact
                @. λ_log = log(λ)
                spectral_outer!(X_log, U, λ_log, zeros(T, size(X_log)))
            end

            cone.trX_log = log(cone.trX)
            
            entr_X   = dot(XH, Hermitian(cone.X_log, :U))
            entr_NX  = dot(NXH, Hermitian(cone.NX_log, :U))
            entr_NcX = dot(NcXH, Hermitian(cone.NcX_log, :U))
            entr_trX = cone.trX * cone.trX_log
            
            cone.z = point[1] - (entr_X + entr_NX - entr_NcX - entr_trX)

            cone.is_feas = (cone.z > 0)
        end
    end
    
    cone.feas_updated = true
    return cone.is_feas
end

function Hypatia.Cones.update_grad(cone::QuantMutualInformation)
    @assert cone.is_feas
    rt2 = cone.rt2
    (Λx, Ux) = cone.X_fact
    Xi = cone.Xi

    spectral_outer!(Xi, Ux, inv.(Λx), zeros(T, size(Xi)))

    log_X      = Hypatia.Cones.smat_to_svec!(zeros(T, cone.vni), cone.X_log, rt2)
    N_log_NX   = cone.N'  * Hypatia.Cones.smat_to_svec!(zeros(T, cone.vno), cone.NX_log, rt2)
    Nc_log_NcX = cone.Nc' * Hypatia.Cones.smat_to_svec!(zeros(T, cone.vne), cone.NcX_log, rt2)
    tr_log_trX = cone.tr  * cone.trX_log

    zi = inv(cone.z)
    cone.DPhi = log_X + N_log_NX - Nc_log_NcX - tr_log_trX

    g = cone.grad
    g[1] = -zi
    @views Hypatia.Cones.smat_to_svec!(g[cone.X_idxs], -Xi, rt2)
    g[cone.X_idxs] += zi * cone.DPhi

    cone.grad_updated = true
    return cone.grad
end

function update_hessprod_aux(cone::QuantMutualInformation)
    @assert !cone.hessprod_aux_updated
    @assert cone.grad_updated

    (Λx, Ux) = cone.X_fact
    (Λnx, Unx) = cone.NX_fact
    (Λncx, Uncx) = cone.NcX_fact

    # Compute first divideded differences matrix
    Δ2_log!(cone.Δ2x_log, Λx, cone.Λx_log)
    Δ2_log!(cone.Δ2nx_log, Λnx, cone.Λnx_log)
    Δ2_log!(cone.Δ2ncx_log, Λncx, cone.Λncx_log)

    @. cone.Δ2x_comb = cone.Δ2x_log + cone.z / (Λx' * Λx)

    cone.hessprod_aux_updated = true
    return
end


function Hypatia.Cones.hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::QuantMutualInformation{T},
) where {T <: Real}
    @assert cone.grad_updated
    cone.hessprod_aux_updated || update_hessprod_aux(cone)

    rt2 = cone.rt2
    zi = 1 / cone.z

    (Λx, Ux) = cone.X_fact
    (Λnx, Unx) = cone.NX_fact
    (Λncx, Uncx) = cone.NcX_fact

    Hx = cone.Hx
    Hnx = cone.Hnx
    Hncx = cone.Hncx

    @inbounds for j in axes(arr, 2)

        # Get input direction
        Ht = arr[1, j]
        @views Hx_vec = arr[cone.X_idxs, j]
        @views Hypatia.Cones.svec_to_smat!(Hx, Hx_vec, rt2)
        @views Hypatia.Cones.svec_to_smat!(Hnx, cone.N * Hx_vec, rt2)
        @views Hypatia.Cones.svec_to_smat!(Hncx, cone.Nc * Hx_vec, rt2)
        LinearAlgebra.copytri!(Hx, 'U')
        LinearAlgebra.copytri!(Hnx, 'U')
        LinearAlgebra.copytri!(Hncx, 'U')

        # Hessian product of quantum entropies
        D2PhiH =             Hypatia.Cones.smat_to_svec!(zeros(T, cone.vni), Ux * ( cone.Δ2x_comb .* (Ux' * Hx * Ux) ) * Ux', cone.rt2)
        D2PhiH += cone.N'  * Hypatia.Cones.smat_to_svec!(zeros(T, cone.vno), Unx * ( cone.Δ2nx_log .* (Unx' * Hnx * Unx) ) * Unx', cone.rt2)
        D2PhiH -= cone.Nc' * Hypatia.Cones.smat_to_svec!(zeros(T, cone.vne), Uncx * ( cone.Δ2ncx_log .* (Uncx' * Hncx * Uncx) ) * Uncx', cone.rt2)
        D2PhiH -= cone.tr  * tr(Hx) / cone.trX

        prodt = zi * zi * (Ht - dot(Hx_vec, cone.DPhi))
        prodX = -prodt * cone.DPhi + zi * D2PhiH

        prod[1, j] = prodt
        prod[cone.X_idxs, j] = prodX

    end

    return prod
end

function update_invhessprod_aux(cone::QuantMutualInformation)
    @assert !cone.invhessprod_aux_updated
    @assert cone.grad_updated
    @assert cone.hessprod_aux_updated

    rt2 = cone.rt2

    (Λx, Ux) = cone.X_fact
    (Λnx, Unx) = cone.NX_fact
    (Λncx, Uncx) = cone.NcX_fact    

    # Hessian for -S(X) component
    k = 1
    @inbounds for j in 1:cone.ni, i in 1:j
        UHU = Ux[i, :] * Ux[j, :]'
        if i != j
            UHU = (UHU + UHU') / rt2
        end
        temp = Ux * (cone.Δ2x_comb .* UHU) * Ux'
        @views Hypatia.Cones.smat_to_svec!(cone.hess[:, k], temp, cone.rt2)
        k += 1
    end

    # Hessian for -S(N(X)) component
    UUN = zeros(T, cone.vni, cone.vno)
    @inbounds for k in axes(cone.N, 2)
        H = Hypatia.Cones.svec_to_smat!(zeros(T, cone.no, cone.no), cone.N[:, k], cone.rt2)
        LinearAlgebra.copytri!(H, 'U')
        UHU = sqrt.(cone.Δ2nx_log) .* (Unx' * H * Unx)
        @views Hypatia.Cones.smat_to_svec!(UUN[k, :], UHU, cone.rt2)
    end
    cone.hess += UUN * UUN'

    # Hessian for +S(Nc(X)) component
    UUNc = zeros(T, cone.vni, cone.vne)
    @inbounds for k in axes(cone.Nc, 2)
        H = Hypatia.Cones.svec_to_smat!(zeros(T, cone.ne, cone.ne), cone.Nc[:, k], cone.rt2)
        LinearAlgebra.copytri!(H, 'U')
        UHU = sqrt.(cone.Δ2ncx_log) .* (Uncx' * H * Uncx)
        @views Hypatia.Cones.smat_to_svec!(UUNc[k, :], UHU, cone.rt2)
    end
    cone.hess -= UUNc * UUNc'

    # Hessian for +S(tr[X]) component
    cone.hess -= cone.tr * cone.tr' / cone.trX

    # Rescale and factor Hessian
    cone.hess /= cone.z
    sym_hess = Symmetric(cone.hess, :U)
    cone.hess_fact = Hypatia.Cones.posdef_fact_copy!(zero(sym_hess), sym_hess)    

    cone.invhessprod_aux_updated = true
    return
end

function Hypatia.Cones.inv_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::QuantMutualInformation{T},
) where {T <: Real}
    @assert cone.grad_updated
    cone.hessprod_aux_updated || update_hessprod_aux(cone)
    cone.invhessprod_aux_updated || update_invhessprod_aux(cone)

    Hx = cone.Hx

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

function update_dder3_aux(cone::QuantMutualInformation)
    @assert !cone.dder3_aux_updated
    @assert cone.hessprod_aux_updated

    Δ3_log!(cone.Δ3x_log, cone.Δ2x_log, cone.X_fact.values)
    Δ3_log!(cone.Δ3nx_log, cone.Δ2nx_log, cone.NX_fact.values)
    Δ3_log!(cone.Δ3ncx_log, cone.Δ2ncx_log, cone.NcX_fact.values)

    cone.dder3_aux_updated = true
    return
end

function Hypatia.Cones.dder3(cone::QuantMutualInformation{T}, dir::AbstractVector{T}) where {T <: Real}
    @assert cone.grad_updated
    cone.hessprod_aux_updated || update_hessprod_aux(cone)
    cone.dder3_aux_updated || update_dder3_aux(cone)

    rt2 = cone.rt2

    (Λx, Ux) = cone.X_fact
    (Λnx, Unx) = cone.NX_fact
    (Λncx, Uncx) = cone.NcX_fact

    Hx = cone.Hx
    Hnx = cone.Hnx
    Hncx = cone.Hncx

    dder3 = cone.dder3

    # Get input direction
    Ht = dir[1]
    @views Hx_vec = dir[cone.X_idxs]
    @views Hypatia.Cones.svec_to_smat!(Hx, Hx_vec, rt2)
    @views Hypatia.Cones.svec_to_smat!(Hnx, cone.N * Hx_vec, rt2)
    @views Hypatia.Cones.svec_to_smat!(Hncx, cone.Nc * Hx_vec, rt2)
    LinearAlgebra.copytri!(Hx, 'U')
    LinearAlgebra.copytri!(Hnx, 'U')
    LinearAlgebra.copytri!(Hncx, 'U')

    UHUx   = Ux'   * Hx   * Ux
    UHUnx  = Unx'  * Hnx  * Unx
    UHUncx = Uncx' * Hncx * Uncx


    # Derivatives of mutual information
    D2PhiH =             Hypatia.Cones.smat_to_svec!(zeros(T, cone.vni), Ux * ( cone.Δ2x_log .* UHUx ) * Ux', cone.rt2)
    D2PhiH += cone.N'  * Hypatia.Cones.smat_to_svec!(zeros(T, cone.vno), Unx * ( cone.Δ2nx_log .* UHUnx ) * Unx', cone.rt2)
    D2PhiH -= cone.Nc' * Hypatia.Cones.smat_to_svec!(zeros(T, cone.vne), Uncx * ( cone.Δ2ncx_log .* UHUncx ) * Uncx', cone.rt2)
    D2PhiH -= cone.tr  * tr(Hx) / cone.trX

    D3PhiHH =             Hypatia.Cones.smat_to_svec!(zeros(T, cone.vni), Δ2_frechet(UHUx .* cone.Δ3x_log, Ux, UHUx), cone.rt2)
    D3PhiHH += cone.N'  * Hypatia.Cones.smat_to_svec!(zeros(T, cone.vno), Δ2_frechet(UHUnx .* cone.Δ3nx_log, Unx, UHUnx), cone.rt2)
    D3PhiHH -= cone.Nc' * Hypatia.Cones.smat_to_svec!(zeros(T, cone.vne), Δ2_frechet(UHUncx .* cone.Δ3ncx_log, Uncx, UHUncx), cone.rt2)
    D3PhiHH += cone.tr  * (tr(Hx) / cone.trX)^2


    # Third directional derivatives of the barrier function
    DPhiH = dot(cone.DPhi, Hx_vec)
    D2PhiHH = dot(D2PhiH, Hx_vec)
    χ = Ht - DPhiH
    zi = 1 / cone.z

    @views dder3_X = dder3[cone.X_idxs]
    dder3[1]  = -(2 * zi^3 * χ^2) - (zi^2 * D2PhiHH)
    dder3_X  .= -dder3[1] * cone.DPhi
    dder3_X .-= 2 * zi^2 * χ * D2PhiH
    dder3_X .+= zi * D3PhiHH
    dder3_X .-= Hypatia.Cones.smat_to_svec!(zeros(T, cone.vni), 2 * cone.Xi * Hx * cone.Xi * Hx * cone.Xi, cone.rt2)

    @. dder3 *= -0.5    

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
    LinearAlgebra.LinearAlgebra.LinearAlgebra.copytri!(Δ2, 'U')
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

# function smat_to_svec(
#     mat::AbstractMatrix{Complex{T}},
#     rt2::Real,
# ) where {T <: Real}
#     return Hypatia.Cones.smat_to_svec!
# end