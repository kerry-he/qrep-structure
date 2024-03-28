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

    DPhi::Vector{T}

    Hx::Matrix{T}
    Hnx::Matrix{T}
    Hncx::Matrix{T}

    hessprod_aux_updated::Bool
    invhessprod_aux_updated::Bool
    dder3_aux_updated::Bool

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

    cone.Δ2x_log = zeros(T, ni, ni)
    cone.Δ2nx_log = zeros(T, no, no)
    cone.Δ2ncx_log = zeros(T, ne, ne)
    cone.Δ2x_comb = zeros(T, ni, ni)
    cone.Δ3x_log = zeros(T, ni, ni, ni)

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

            cone.fval = -log(cone.z) - logdet(cone.X)

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
    DPhi = cone.DPhi

    spectral_outer!(Xi, Ux, inv.(Λx), zeros(T, size(Xi)))

    log_X      = Hypatia.Cones.smat_to_svec!(zeros(T, cone.vni), cone.X_log, rt2)
    N_log_NX   = cone.N'  * Hypatia.Cones.smat_to_svec!(zeros(T, cone.vno), cone.NX_log, rt2)
    Nc_log_NcX = cone.Nc' * Hypatia.Cones.smat_to_svec!(zeros(T, cone.vne), cone.NcX_log, rt2)
    tr_log_trX = cone.tr  * cone.trX_log

    zi = inv(cone.z)
    DPhi = log_X + N_log_NX - Nc_log_NcX - tr_log_trX

    g = cone.grad
    g[1] = -zi
    @views Hypatia.Cones.smat_to_svec!(g[cone.X_idxs], -Xi, rt2)
    g[cone.X_idxs] += zi * DPhi

    cone.grad_updated = true
    return cone.grad
end

function update_hessprod_aux(cone::QuantMutualInformation)
    @assert !cone.hessprod_aux_updated
    @assert cone.grad_updated

    rt2 = cone.rt2
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

    @inbounds for j in 1:size(arr, 2)

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

        
    return prod
end

function update_dder3_aux(cone::QuantMutualInformation)
    @assert !cone.dder3_aux_updated
    @assert cone.hessprod_aux_updated

    cone.dder3_aux_updated = true
    return
end

function Hypatia.Cones.dder3(cone::QuantMutualInformation{T}, dir::AbstractVector{T}) where {T <: Real}
    @assert cone.grad_updated
    cone.hessprod_aux_updated || update_hessprod_aux(cone)
    cone.dder3_aux_updated || update_dder3_aux(cone)

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
