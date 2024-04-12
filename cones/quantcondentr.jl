using LinearAlgebra
import Hypatia.Cones

include("../utils/helper.jl")

mutable struct QuantCondEntropy{T <: Real} <: Hypatia.Cones.Cone{T}
    use_dual_barrier::Bool
    dim::Int

    n::Int
    n1::Int
    n2::Int
    m::Int
    N::Int
    sys::Int
    X_dim::Int
    Y_dim::Int
    X_idxs::UnitRange{Int}
    Y_idxs::UnitRange{Int}

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    dder3::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}

    rt2::T
    In::Matrix{T}
    IN::Matrix{T}
    X::Matrix{T}
    Y::Matrix{T}
    X_fact::Eigen{T}
    Y_fact::Eigen{T}
    Xi::Matrix{T}
    Yi::Matrix{T}
    Λx_log::Vector{T}
    Λy_log::Vector{T}
    X_log::Matrix{T}
    Y_log::Matrix{T}
    XY_log::Matrix{T}
    z::T

    Δ2x_log::Matrix{T}
    Δ2x_comb_inv::Matrix{T}
    Δ3x_log::Array{T, 3}
    Δ2y_log::Matrix{T} 
    Δ3y_log::Array{T, 3}

    UyXUy::Matrix{T}
    DPhi::Matrix{T}
    H_inv_g_x::Matrix{T}
    DPhi_H_DPhi::T
    UxK::Matrix{T}
    UxK_temp::Matrix{T}
    HYY::Matrix{T}
    KHxK::Matrix{T}
    HYY_KHxK::Matrix{T}
    HYY_KHxK_chol::Cholesky{T, Matrix{T}}
    schur_temp::Symmetric{T, Matrix{T}}

    Hx::Matrix{T}
    Hy::Matrix{T}

    matn::Matrix{T}
    matn2::Matrix{T}
    matn3::Matrix{T}
    matn4::Matrix{T}
    matm::Matrix{T}
    matN::Matrix{T}
    matN2::Matrix{T}
    matN3::Matrix{T}
    matN4::Matrix{T}
    vecm::Vector{T}
    vecm2::Vector{T}
    vecM::Vector{T}
    vecn::Vector{T}
    vecn2::Vector{T}
    vecN::Vector{T}
    vecN2::Vector{T}

    hessprod_aux_updated::Bool
    invhessprod_aux_updated::Bool
    dder3_aux_updated::Bool

    function QuantCondEntropy{T}(
        n1::Int,
        n2::Int,
        sys::Int;
        use_dual::Bool = false,
    ) where {T <: Real}
        cone = new{T}()
        cone.use_dual_barrier = use_dual

        cone.n1 = n1
        cone.n2 = n2
        cone.n = if (sys == 1) n2 else n1 end
        cone.N = n1*n2
        cone.sys = sys
        cone.X_dim = Hypatia.Cones.svec_length(cone.N)
        cone.Y_dim = Hypatia.Cones.svec_length(cone.n)
        cone.dim = 1 + cone.X_dim

        return cone
    end
end

function Hypatia.Cones.reset_data(cone::QuantCondEntropy)
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
    cone::QuantCondEntropy{T},
) where {T <: Real}
    n = cone.n
    N = cone.N
    dim = cone.dim
    X_dim = cone.X_dim
    Y_dim = cone.Y_dim

    cone.rt2 = sqrt(T(2))
    cone.In = Matrix{T}(I, n, n)
    cone.IN = Matrix{T}(I, N, N)
    cone.X_idxs = 2:dim

    cone.X = zeros(T, N, N)
    cone.Y = zeros(T, n, n)
    cone.Xi = zeros(T, N, N)
    cone.Yi = zeros(T, n, n)
    cone.Λx_log = zeros(T, N)
    cone.Λy_log = zeros(T, n)
    cone.X_log = zeros(T, N, N)
    cone.Y_log = zeros(T, n, n)
    cone.XY_log = zeros(T, N, N)

    cone.Δ2x_log = zeros(T, N, N)
    cone.Δ2x_comb_inv = zeros(T, N, N)
    cone.Δ3x_log = zeros(T, N, N, N)
    cone.Δ2y_log = zeros(T, n, n)
    cone.Δ3y_log = zeros(T, n, n, n)

    cone.UyXUy = zeros(T, n, n)
    cone.DPhi = zeros(T, N, N)
    cone.H_inv_g_x = zeros(T, N, N)
    cone.DPhi_H_DPhi = 0
    cone.UxK = zeros(T, X_dim, Y_dim)
    cone.UxK_temp = zeros(T, X_dim, Y_dim)
    cone.HYY = zeros(T, Y_dim, Y_dim)
    cone.KHxK = zeros(T, Y_dim, Y_dim)
    cone.HYY_KHxK = zeros(T, Y_dim, Y_dim)
    cone.schur_temp = Symmetric(zeros(T, Y_dim, Y_dim))

    cone.Hx = zeros(T, N, N)
    cone.Hy = zeros(T, n, n)

    cone.matn = zeros(T, n, n)
    cone.matn2 = zeros(T, n, n)
    cone.matn3 = zeros(T, n, n)
    cone.matn4 = zeros(T, n, n)
    cone.matm = zeros(T, Y_dim, Y_dim)
    cone.matN = zeros(T, N, N)
    cone.matN2 = zeros(T, N, N)
    cone.matN3 = zeros(T, N, N)
    cone.matN4 = zeros(T, N, N)
    cone.vecm = zeros(T, Y_dim)
    cone.vecm2 = zeros(T, Y_dim)
    cone.vecM = zeros(T, X_dim)
    cone.vecn = zeros(T, n)
    cone.vecn2 = zeros(T, n)
    cone.vecN = zeros(T, N)
    cone.vecN2 = zeros(T, N)

    return cone
end

Hypatia.Cones.get_nu(cone::QuantCondEntropy) = 1 + cone.N

function Hypatia.Cones.set_initial_point!(
    arr::AbstractVector{T},
    cone::QuantCondEntropy{T},
) where {T <: Real}
    arr[1] = 0.0
    X0 = Matrix{T}(I, cone.N, cone.N)
    @views Hypatia.Cones.smat_to_svec!(arr[cone.X_idxs], X0, cone.rt2)

    return arr
end

function Hypatia.Cones.update_feas(cone::QuantCondEntropy{T}) where {T <: Real}
    @assert !cone.feas_updated
    point = cone.point
    (n1, n2, sys) = (cone.n1, cone.n2, cone.sys)

    cone.is_feas = false
    @views Hypatia.Cones.svec_to_smat!(cone.X, point[cone.X_idxs], cone.rt2)
    LinearAlgebra.copytri!(cone.X, 'U')
    pTr!(cone.Y, cone.X, cone.sys, (cone.n1, cone.n2))

    XH = Hermitian(cone.X, :U)
    YH = Hermitian(cone.Y, :U)
    if isposdef(XH)
        X_fact = cone.X_fact = eigen(XH)
        Y_fact = cone.Y_fact = eigen(YH)
        if isposdef(X_fact) && isposdef(Y_fact)
            for (fact, λ_log, X_log, mat) in zip(
                (X_fact, Y_fact),
                (cone.Λx_log, cone.Λy_log),
                (cone.X_log, cone.Y_log),
                (cone.matN, cone.matn)
            )
                (λ, vecs) = fact
                @. λ_log = log(λ)
                spectral_outer!(X_log, vecs, λ_log, mat)
            end
            cone.XY_log .= cone.X_log .- idKron(cone.Y_log, sys, (n1, n2))
            cone.z = point[1] - dot(XH, Hermitian(cone.XY_log, :U))
            cone.is_feas = (cone.z > 0)
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

function Hypatia.Cones.update_grad(cone::QuantCondEntropy)
    @assert cone.is_feas
    rt2 = cone.rt2
    (Λx, Ux) = cone.X_fact
    Xi = cone.Xi

    spectral_outer!(Xi, Ux, inv.(Λx), zeros(T, size(Xi)))

    zi = inv(cone.z)
    @. cone.DPhi = cone.XY_log

    g = cone.grad
    g[1] = -zi;
    @views Hypatia.Cones.smat_to_svec!(g[cone.X_idxs], cone.DPhi * zi - Xi, rt2)

    cone.grad_updated = true
    return cone.grad
end

function update_hessprod_aux(cone::QuantCondEntropy)
    @assert !cone.hessprod_aux_updated
    @assert cone.grad_updated

    Λx = cone.X_fact.values
    Λy = cone.Y_fact.values

    Δ2_log!(cone.Δ2x_log, Λx, cone.Λx_log)
    Δ2_log!(cone.Δ2y_log, Λy, cone.Λy_log)

    cone.hessprod_aux_updated = true
    return
end


function Hypatia.Cones.hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::QuantCondEntropy{T},
) where {T <: Real}
    @assert cone.grad_updated
    cone.hessprod_aux_updated || update_hessprod_aux(cone)

    (n1, n2, sys) = (cone.n1, cone.n2, cone.sys)

    rt2 = cone.rt2
    Ux = cone.X_fact.vectors
    Uy = cone.Y_fact.vectors
    zi = inv(cone.z)

    Hx = cone.Hx
    Hy = cone.Hy

    @inbounds for j in axes(arr, 2)
        # Get input direction
        Ht = arr[1, j]
        @views Hypatia.Cones.svec_to_smat!(Hx, arr[cone.X_idxs, j], rt2)
        LinearAlgebra.copytri!(Hx, 'U')
        pTr!(Hy, Hx, cone.sys, (cone.n1, cone.n2))

        # Hessian product of quantum entropies
        D2PhiH = Ux * ( cone.Δ2x_log .* (Ux' * Hx * Ux) ) * Ux'
        D2PhiH -= idKron(Uy * ( cone.Δ2y_log .* (Uy' * Hy * Uy) ) * Uy', sys, (n1, n2))

        # Hessian product of barrier
        prodt = zi * zi * (Ht - dot(Hx, cone.DPhi))
        prodX = -prodt * cone.DPhi + zi * D2PhiH + cone.Xi * Hx * cone.Xi

        prod[1, j] = prodt
        @views Hypatia.Cones.smat_to_svec!(prod[cone.X_idxs, j], prodX, rt2)
    end

    return prod
end

function update_invhessprod_aux(cone::QuantCondEntropy)
    @assert !cone.invhessprod_aux_updated
    @assert cone.grad_updated
    @assert cone.hessprod_aux_updated

    rt2 = cone.rt2
    n = cone.n
    Δ2x_log = cone.Δ2x_log
    Δ2x_comb_inv = cone.Δ2x_comb_inv
    Δ2y_log = cone.Δ2y_log
    (Λx, Ux) = cone.X_fact
    (Λy, Uy) = cone.Y_fact
    zi = inv(cone.z)
    vecn = cone.vecn
    vecn2 = cone.vecn2
    vecN = cone.vecN
    vecN2 = cone.vecN2

    DPhi = cone.DPhi
    H_inv_g_x = cone.H_inv_g_x

    UxK = cone.UxK
    UxK_temp = cone.UxK_temp
    HYY = cone.HYY
    KHxK = cone.KHxK
    HYY_KHxK = cone.HYY_KHxK

    matn = cone.matn
    matn2 = cone.matn2
    matn3 = cone.matn3
    matN = cone.matN
    matN2 = cone.matN2
    matN3 = cone.matN3
    matN4 = cone.matN4
    vecm = cone.vecm
    vecm2 = cone.vecm2
    vecM = cone.vecM

    Hx = cone.Hx
    Hy = cone.Hy
    rt2i = 1 / rt2
    
    @. matN = 1 / (Λx' * Λx)
    @. Δ2x_comb_inv = 1 / (Δ2x_log*zi + matN)

    # Construct matrices
    k = 0
    @inbounds for j in 1:n, i in 1:j
        k += 1

        Hx .= 0

        if cone.sys == 1
            @inbounds for l = 0:cone.n1-1
                copyto!(vecN, Ux[cone.n2*l + i, :])
                copyto!(vecN2, Ux[cone.n2*l + j, :])
                mul!(Hx, vecN, vecN2', true, true)
            end
        else
            @inbounds for l = 1:cone.n2
                copyto!(vecN, Ux[cone.n2*(i - 1) + l, :])
                copyto!(vecN2, Ux[cone.n2*(j - 1) + l, :])
                mul!(Hx, vecN, vecN2', true, true)
            end         
        end
            
        if i != j
            @. matN = Hx + Hx'
            @. Hx = rt2i * matN          
        end
        @views UxK_k = UxK[:, k]
        Hypatia.Cones.smat_to_svec!(UxK_k, Hx, rt2)

        copyto!(vecn, Uy[i, :])
        copyto!(vecn2, Uy[j, :])
        mul!(Hy, vecn, vecn2')
        if i != j
            @. matn = Hy + Hy'
            @. Hy = rt2i * matn
        end

        
        # HYY
        @. matn = cone.z * Hy / Δ2y_log
        spectral_outer!(matn3, Uy, matn, matn2)
        @views HYY_k = HYY[:, k]
        Hypatia.Cones.smat_to_svec!(HYY_k, matn3, rt2)

    end

    Hypatia.Cones.smat_to_svec!(vecM, Δ2x_comb_inv, 1)
    @. UxK_temp = vecM * UxK
    mul!(KHxK, UxK', UxK_temp)
    @. HYY_KHxK = HYY - KHxK;

    sym_hess = Symmetric(HYY_KHxK, :U)
    cone.HYY_KHxK_chol = Hypatia.Cones.posdef_fact_copy!(cone.schur_temp, sym_hess)    

    cone.invhessprod_aux_updated = true
    return
end

function Hypatia.Cones.inv_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::QuantCondEntropy{T},
) where {T <: Real}
    @assert cone.grad_updated
    cone.hessprod_aux_updated || update_hessprod_aux(cone)
    cone.invhessprod_aux_updated || update_invhessprod_aux(cone)

    (n1, n2, sys) = (cone.n1, cone.n2, cone.sys)

    rt2 = cone.rt2
    Ux = cone.X_fact.vectors

    p = size(arr, 2)

    Wx_k = zeros(T, cone.N, cone.N)
    Wy_k = zeros(T, cone.n, cone.n)

    Hx = cone.Hx
    Wx = [zeros(T, cone.N, cone.N) for i in 1:p]
    Wy = zeros(T, cone.Y_dim, p)

    @inbounds for k in axes(arr, 2)
        
        # Get input direction
        Ht = arr[1, k]
        @views Hypatia.Cones.svec_to_smat!(Hx, arr[cone.X_idxs, k], rt2)
        LinearAlgebra.copytri!(Hx, 'U')
        @. Wx[k] = Hx + cone.DPhi * Ht

        Wx[k] = Ux * (cone.Δ2x_comb_inv .* (Ux' * Wx[k] * Ux)) * Ux'
        pTr!(Wy_k, Wx[k], cone.sys, (cone.n1, cone.n2))
        @views Hypatia.Cones.smat_to_svec!(Wy[:, k], -Wy_k, rt2)

    end

    Wy = cone.HYY_KHxK_chol \ Wy
        
    @inbounds for k in axes(arr, 2)

        # Get input direction
        Ht = arr[1, k]
        Hypatia.Cones.svec_to_smat!(Wy_k, Wy[:, k], rt2)
        LinearAlgebra.copytri!(Wy_k, 'U', true)

        idKron!(Wx_k, Wy_k, sys, (n1, n2))
        Wx[k] .-= Ux * ( cone.Δ2x_comb_inv .* (Ux' * Wx_k * Ux) ) * Ux'

        # Solve linear system
        @views Hypatia.Cones.smat_to_svec!(prod[cone.X_idxs, k], Wx[k], rt2)
        prod[1, k] = cone.z * cone.z * Ht[k] + dot(Wx[k], cone.DPhi)

    end    

    return prod
end

function update_dder3_aux(cone::QuantCondEntropy)
    @assert !cone.dder3_aux_updated
    @assert cone.hessprod_aux_updated

    Δ3!(cone.Δ3x_log, cone.Δ2x_log, cone.X_fact.values)
    Δ3!(cone.Δ3y_log, cone.Δ2y_log, cone.Y_fact.values)

    cone.dder3_aux_updated = true
    return
end

function Hypatia.Cones.dder3(cone::QuantCondEntropy{T}, dir::AbstractVector{T}) where {T <: Real}
    @assert cone.grad_updated
    cone.hessprod_aux_updated || update_hessprod_aux(cone)
    cone.dder3_aux_updated || update_dder3_aux(cone)

    (n1, n2, sys) = (cone.n1, cone.n2, cone.sys)

    rt2 = cone.rt2
    Ux = cone.X_fact.vectors
    Uy = cone.Y_fact.vectors
    Hx = cone.Hx
    Hy = cone.Hy

    dder3 = cone.dder3

    # Get input direction
    Ht = dir[1]
    @views Hypatia.Cones.svec_to_smat!(Hx, dir[cone.X_idxs], rt2)
    LinearAlgebra.copytri!(Hx, 'U')
    pTr!(Hy, Hx, cone.sys, (cone.n1, cone.n2))

    UHUx = Ux' * Hx * Ux
    UHUy = Uy' * Hy * Uy

    # Derivatives of mutual information
    D2PhiH  = Ux * ( cone.Δ2x_log .* (Ux' * Hx * Ux) ) * Ux'
    D2PhiH -= idKron(Uy * ( cone.Δ2y_log .* (Uy' * Hy * Uy) ) * Uy', sys, (n1, n2))

    D3PhiHH  = Δ2_frechet(UHUx .* cone.Δ3x_log, Ux, UHUx)
    D3PhiHH -= idKron(Δ2_frechet(UHUy .* cone.Δ3y_log, Uy, UHUy), sys, (n1, n2))

    # Third directional derivatives of the barrier function
    DPhiH = dot(cone.DPhi, Hx)
    D2PhiHH = dot(D2PhiH, Hx)
    χ = Ht - DPhiH
    zi = 1 / cone.z

    dder3[1]  = -(2 * zi^3 * χ^2) - (zi^2 * D2PhiHH)
    temp   = -dder3[1] * cone.DPhi
    temp .-= 2 * zi^2 * χ * D2PhiH
    temp .+= zi * D3PhiHH
    temp .-= 2 * cone.Xi * Hx * cone.Xi * Hx * cone.Xi
    @views Hypatia.Cones.smat_to_svec!(dder3[cone.X_idxs], temp, cone.rt2)

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

function Δ3!(Δ3::Array{T, 3}, Δ2::Matrix{T}, λ::Vector{T}) where {T <: Real}
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
    # spectral_outer!(Δ2_F, U, Symmetric(mat, :U), mat2)

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