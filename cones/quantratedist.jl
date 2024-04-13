using LinearAlgebra
import Hypatia.Cones

include("../utils/helper.jl")
include("../utils/spectral.jl")

mutable struct QuantRateDistortion{T <: Real} <: Hypatia.Cones.Cone{T}
    use_dual_barrier::Bool
    dim::Int

    n::Int
    vn::Int
    m::Int
    X_idxs::UnitRange{Int}
    y_idxs::UnitRange{Int}

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
    X::Matrix{T}
    y::Vector{T}
    w::Vector{T}
    X_fact::Eigen{T}
    Xi::Matrix{T}
    yi::Vector{T}
    Λx_log::Vector{T}
    X_log::Matrix{T}
    y_log::Vector{T}
    w_log::Vector{T}
    z::T

    Δ2x_log::Matrix{T}
    Δ2x_comb_inv::Matrix{T}
    Δ3x_log::Array{T, 3}
    Δ2y_comb_inv::Vector{T} 

    fval::T
    DPhiX::Matrix{T}
    DPhiy::Vector{T}
    schur::Matrix{T}
    schur_fact

    Hx::Matrix{T}
    Hy::Vector{T}
    Hw::Vector{T}

    hessprod_aux_updated::Bool
    invhessprod_aux_updated::Bool
    dder3_aux_updated::Bool

    function QuantRateDistortion{T}(
        n::Int,
        use_dual::Bool = false,
    ) where {T <: Real}
        cone = new{T}()
        cone.use_dual_barrier = use_dual

        # Get dimensions
        cone.n  = n                              # Dimension of input
        cone.vn = Hypatia.Cones.svec_length(n)   # Dimension of vectorized system being traced out
        cone.m  = n * (n - 1)                    # Dimension of diagonal component
        
        cone.dim = 1 + cone.m + cone.vn

        return cone
    end
end

function Hypatia.Cones.reset_data(cone::QuantRateDistortion)
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
    cone::QuantRateDistortion{T},
) where {T <: Real}
    n = cone.n
    m = cone.m
    dim = cone.dim

    cone.rt2 = sqrt(T(2))
    cone.y_idxs = 2:2+m-1
    cone.X_idxs = 2+m:dim

    cone.X = zeros(T, n, n)
    cone.y = zeros(T, m)
    cone.w = zeros(T, n)
    cone.Xi = zeros(T, n, n)
    cone.yi = zeros(T, m)
    cone.Λx_log = zeros(T, n)
    cone.X_log = zeros(T, n, n)
    cone.y_log = zeros(T, m)
    cone.w_log = zeros(T, n)

    cone.Δ2x_log = zeros(T, n, n)
    cone.Δ2x_comb_inv = zeros(T, n, n)
    cone.Δ3x_log = zeros(T, n, n, n)
    cone.Δ2y_comb_inv = zeros(T, m)

    cone.DPhiX = zeros(T, n, n)
    cone.DPhiy = zeros(T, m)
    cone.schur = zeros(T, n, n)

    cone.Hx = zeros(T, n, n)
    cone.Hy = zeros(T, m)
    cone.Hw = zeros(T, n)

    return cone
end

Hypatia.Cones.get_nu(cone::QuantRateDistortion) = 1 + cone.n^2

function Hypatia.Cones.set_initial_point!(
    arr::AbstractVector{T},
    cone::QuantRateDistortion{T},
) where {T <: Real}
    rt2 = cone.rt2

    arr[1] = 1.0
    @views arr[cone.y_idxs] .= 1.0
    @views Hypatia.Cones.smat_to_svec!(arr[cone.X_idxs], Matrix{T}(I, cone.n, cone.n), rt2)

    return arr
end

function Hypatia.Cones.update_feas(cone::QuantRateDistortion{T}) where {T <: Real}
    @assert !cone.feas_updated
    cone.is_feas = false

    # Compute required maps
    point = cone.point
    @views cone.y = point[cone.y_idxs]
    @views Hypatia.Cones.svec_to_smat!(cone.X, point[cone.X_idxs], cone.rt2)
    @views cone.w = diag(cone.X) .+ reshape(sum(reshape(cone.y', (cone.n, cone.n-1)), dims=2), (cone.n))

    LinearAlgebra.copytri!(cone.X, 'U')

    XH = Hermitian(cone.X, :U)

    if isposdef(XH)
        X_fact = cone.X_fact = eigen(XH)
        if isposdef(X_fact) && all(cone.y .> 0) && all(cone.w .> 0)
            (λ, U) = X_fact
            @. cone.Λx_log = log(λ)
            cone.X_log .= U * (cone.Λx_log .* U')

            cone.y_log = log.(cone.y)
            cone.w_log = log.(cone.w)
            
            entr_X = dot(XH, Hermitian(cone.X_log, :U))
            entr_y = dot(cone.y, cone.y_log)
            entr_w = dot(cone.w, cone.w_log)
            
            cone.z = point[1] - (entr_X + entr_y - entr_w)

            cone.is_feas = (cone.z > 0)
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

function Hypatia.Cones.update_grad(cone::QuantRateDistortion)
    @assert cone.is_feas
    rt2 = cone.rt2
    (Λx, Ux) = cone.X_fact

    cone.Xi = Ux * (inv.(Λx) .* Ux')
    cone.yi = inv.(cone.y)

    zi = 1 / cone.z
    cone.DPhiX = cone.X_log - diagm(cone.w_log)
    cone.DPhiy = cone.y_log - reshape(repeat(cone.w_log, cone.n-1), cone.m)

    g = cone.grad
    g[1] = -zi
    @views g[cone.y_idxs] = zi * cone.DPhiy - cone.yi
    @views Hypatia.Cones.smat_to_svec!(g[cone.X_idxs], zi * cone.DPhiX - cone.Xi, rt2)

    cone.grad_updated = true
    return cone.grad
end

function update_hessprod_aux(cone::QuantRateDistortion)
    @assert !cone.hessprod_aux_updated
    @assert cone.grad_updated

    (Λx, Ux) = cone.X_fact

    # Compute first divideded differences matrix
    Δ2_log!(cone.Δ2x_log, Λx, cone.Λx_log)

    cone.hessprod_aux_updated = true
    return
end


function Hypatia.Cones.hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::QuantRateDistortion{T},
) where {T <: Real}
    @assert cone.grad_updated
    cone.hessprod_aux_updated || update_hessprod_aux(cone)

    rt2 = cone.rt2
    zi = 1 / cone.z

    (Λx, Ux) = cone.X_fact
    Hx = cone.Hx
    Hy = cone.Hy
    Hw = cone.Hw

    @inbounds for j in axes(arr, 2)

        # Get input direction
        Ht = arr[1, j]
        @views Hx_vec = arr[cone.X_idxs, j]
        @views Hypatia.Cones.svec_to_smat!(Hx, Hx_vec, rt2)
        Hy = arr[cone.y_idxs, j]
        Hw = diag(Hx) .+ reshape(sum(reshape(Hy', (cone.n, cone.n-1)), dims=2), (cone.n))
        LinearAlgebra.copytri!(Hx, 'U')

        # Hessian product of quantum entropies
        D2PhiwH = Hw ./ cone.w

        D2PhiXH = Ux * ( cone.Δ2x_log .* (Ux' * Hx * Ux) ) * Ux' - diagm(D2PhiwH)
        D2PhiyH = Hy ./ cone.y - reshape(repeat(D2PhiwH, cone.n-1), cone.m)


        prodt = zi * zi * (Ht - dot(Hx, cone.DPhiX) - dot(Hy, cone.DPhiy))
        prodX = -prodt * cone.DPhiX + zi * D2PhiXH + cone.Xi * Hx * cone.Xi
        prody = -prodt * cone.DPhiy + zi * D2PhiyH + Hy ./ cone.y ./ cone.y

        prod[1, j] = prodt
        @views Hypatia.Cones.smat_to_svec!(prod[cone.X_idxs, j], prodX, cone.rt2)
        @views prod[cone.y_idxs] = prody

    end

    return prod
end

function update_invhessprod_aux(cone::QuantRateDistortion)
    @assert !cone.invhessprod_aux_updated
    @assert cone.grad_updated
    @assert cone.hessprod_aux_updated

    rt2 = cone.rt2
    zi = 1 / cone.z
    (Λx, Ux) = cone.X_fact

    Δ2x_inv = inv.(Λx * Λx')
    @. cone.Δ2x_comb_inv = 1 / (zi * cone.Δ2x_log + Δ2x_inv)
    @. cone.Δ2y_comb_inv = 1 / (zi / cone.y + 1 / cone.y / cone.y)

    for k in 1:cone.n
        temp = Ux * (cone.Δ2x_comb_inv .* (Ux[k, :] * Ux[k, :]')) * Ux'
        @views cone.schur[:, k] = -diag(temp)
    end
    cone.schur -= diagm(reshape(sum(reshape(cone.Δ2y_comb_inv', (cone.n, cone.n-1)), dims=2), (cone.n)))
    cone.schur += diagm(cone.w / zi)
    
    sym_hess = Symmetric(cone.schur, :U)
    cone.schur_fact = Hypatia.Cones.posdef_fact_copy!(zero(sym_hess), sym_hess)    

    cone.invhessprod_aux_updated = true
    return
end

function Hypatia.Cones.inv_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::QuantRateDistortion{T},
) where {T <: Real}
    @assert cone.grad_updated
    cone.hessprod_aux_updated || update_hessprod_aux(cone)
    cone.invhessprod_aux_updated || update_invhessprod_aux(cone)

    rt2 = cone.rt2
    (Λx, Ux) = cone.X_fact
    Hx = cone.Hx

    @inbounds for j in axes(arr, 2)

        # Get input direction
        Ht = arr[1, j]
        @views Hx_vec = arr[cone.X_idxs, j]
        @views Hypatia.Cones.svec_to_smat!(Hx, Hx_vec, rt2)
        Hy = arr[cone.y_idxs, j]
        LinearAlgebra.copytri!(Hx, 'U')

        Wx = Hx + Ht * cone.DPhiX
        Wy = Hy + Ht * cone.DPhiy

        # Solve linear system
        tempx = Ux * (cone.Δ2x_comb_inv .* (Ux' * Wx * Ux)) * Ux'
        tempy = Wy .* cone.Δ2y_comb_inv
        tempw = diag(tempx) .+ reshape(sum(reshape(tempy', (cone.n, cone.n-1)), dims=2), (cone.n))
        tempw = cone.schur_fact \ tempw

        tempx = Wx + diagm(tempw)
        tempy = Wy + reshape(repeat(tempw, cone.n-1), cone.m)
        
        prodX = Ux * (cone.Δ2x_comb_inv .* (Ux' * tempx * Ux)) * Ux'
        prody = tempy .* cone.Δ2y_comb_inv
        prodt = cone.z * cone.z * Ht + dot(prodX, cone.DPhiX) + dot(prody, cone.DPhiy)

        prod[1, j] = prodt
        @views prod[cone.y_idxs, j] = prody
        @views Hypatia.Cones.smat_to_svec!(prod[cone.X_idxs, j], prodX, rt2)

    end
        
    return prod
end

function update_dder3_aux(cone::QuantRateDistortion)
    @assert !cone.dder3_aux_updated
    @assert cone.hessprod_aux_updated

    Δ3_log!(cone.Δ3x_log, cone.Δ2x_log, cone.X_fact.values)

    cone.dder3_aux_updated = true
    return
end

function Hypatia.Cones.dder3(cone::QuantRateDistortion{T}, dir::AbstractVector{T}) where {T <: Real}
    @assert cone.grad_updated
    cone.hessprod_aux_updated || update_hessprod_aux(cone)
    cone.dder3_aux_updated || update_dder3_aux(cone)

    rt2 = cone.rt2
    (Λx, Ux) = cone.X_fact
    Hx = cone.Hx
    zi = 1 / cone.z

    # Get input direction
    Ht = dir[1]
    @views Hx_vec = dir[cone.X_idxs]
    @views Hypatia.Cones.svec_to_smat!(Hx, Hx_vec, rt2)
    Hy = dir[cone.y_idxs]
    Hw = diag(Hx) .+ reshape(sum(reshape(Hy', (cone.n, cone.n-1)), dims=2), (cone.n))
    LinearAlgebra.copytri!(Hx, 'U')

    # Derivatives of function
    D2PhiwH  = Hw ./ cone.w
    D3PhiwHH = -(Hw ./ cone.w).^2

    UHU   = Ux' * Hx * Ux
    D2PhiXH  = Ux * (cone.Δ2x_log .* UHU) * Ux' - diagm(D2PhiwH)
    D3PhiXHH = Δ2_frechet(UHU .* cone.Δ3x_log, Ux, UHU) - diagm(D3PhiwHH)

    D2PhiyH  = Hy ./ cone.y - reshape(repeat(D2PhiwH, cone.n-1), cone.m)
    D3PhiyHH = - (Hy ./ cone.y).^2 - reshape(repeat(D3PhiwHH, cone.n-1), cone.m)

    DPhiXH = dot(cone.DPhiX, Hx)
    DPhiyH = dot(cone.DPhiy, Hy)
    D2PhiXHH = dot(D2PhiXH, Hx)        
    D2PhiyHH = dot(D2PhiyH, Hy)        
    chi = Ht - DPhiXH - DPhiyH
    chi2 = chi * chi

    # Third derivatives of barrier
    dder3_t = -2 * (zi^3) * chi2 - (zi^2) * (D2PhiXHH + D2PhiyHH)

    dder3_X  = -dder3_t * cone.DPhiX
    dder3_X -=  2 * (zi^2) * chi * D2PhiXH
    dder3_X +=  zi * D3PhiXHH
    dder3_X -=  2 * cone.Xi * Hx * cone.Xi * Hx * cone.Xi

    dder3_y  = -dder3_t * cone.DPhiy
    dder3_y -=  2 * (zi^2) * chi * D2PhiyH
    dder3_y +=  zi * D3PhiyHH
    dder3_y -=  2 * (Hy.^2) ./ (cone.y.^3)

    dder3             = cone.dder3
    dder3[1]          = dder3_t
    @views Hypatia.Cones.smat_to_svec!(dder3[cone.X_idxs], dder3_X, rt2)
    @views dder3[cone.y_idxs] = dder3_y

    @. dder3 *= -0.5

    return dder3
end