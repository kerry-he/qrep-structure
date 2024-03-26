using LinearAlgebra
import Hypatia.Cones
# import Hypatia.spectral_outer!


mutable struct EpiCondEntropyTri{T <: Real} <: Hypatia.Cones.Cone{T}
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
    DPhiX::Matrix{T}
    H_inv_g_x::Matrix{T}
    DPhi_H_DPhi::T
    UxK::Matrix{T}
    UxK_temp::Matrix{T}
    HYY::Matrix{T}
    KHxK::Matrix{T}
    HYY_KHxK::Matrix{T}
    HYY_KHxK_chol::Cholesky{T, Matrix{T}}

    HX::Matrix{T}
    HY::Matrix{T}

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

    function EpiCondEntropyTri{T}(
        dim::Int,
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
        cone.dim = dim
        @assert dim == 1 + cone.X_dim

        return cone
    end
end

function Hypatia.Cones.reset_data(cone::EpiCondEntropyTri)
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
    cone::EpiCondEntropyTri{T},
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
    cone.DPhiX = zeros(T, N, N)
    cone.H_inv_g_x = zeros(T, N, N)
    cone.DPhi_H_DPhi = 0
    cone.UxK = zeros(T, X_dim, Y_dim)
    cone.UxK_temp = zeros(T, X_dim, Y_dim)
    cone.HYY = zeros(T, Y_dim, Y_dim)
    cone.KHxK = zeros(T, Y_dim, Y_dim)
    cone.HYY_KHxK = zeros(T, Y_dim, Y_dim)

    cone.HX = zeros(T, N, N)
    cone.HY = zeros(T, n, n)

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

Hypatia.Cones.get_nu(cone::EpiCondEntropyTri) = 1 + cone.N

function Hypatia.Cones.set_initial_point!(
    arr::AbstractVector{T},
    cone::EpiCondEntropyTri{T},
) where {T <: Real}
    arr .= 0
    rt2 = cone.rt2

    arr[1] = 0
    X = Matrix{T}(I, cone.N, cone.N) / cone.N

    @views arr_X = arr[cone.X_idxs]
    Hypatia.Cones.smat_to_svec!(arr_X, X, rt2)

    return arr
end

function Hypatia.Cones.update_feas(cone::EpiCondEntropyTri{T}) where {T <: Real}
    @assert !cone.feas_updated
    point = cone.point

    cone.is_feas = false
    @views Hypatia.Cones.svec_to_smat!(cone.X, point[cone.X_idxs], cone.rt2)
    LinearAlgebra.copytri!(cone.X, 'U')
    pTr!(cone.Y, cone.X, cone.sys, (cone.n1, cone.n2))

    XH = Hermitian(cone.X, :U)
    YH = Hermitian(cone.Y, :U)
    if isposdef(XH) && isposdef(YH)
        # TODO use LAPACK syev! instead of syevr! for efficiency
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
            idKron!(cone.matN, cone.Y_log, cone.sys, (cone.n1, cone.n2))
            @. cone.XY_log = cone.X_log - cone.matN
            cone.z = point[1] - dot(XH, Hermitian(cone.XY_log, :U))
            cone.is_feas = (cone.z > 0)
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

function Hypatia.Cones.update_grad(cone::EpiCondEntropyTri)
    @assert cone.is_feas
    rt2 = cone.rt2
    (Λx, Ux) = cone.X_fact
    Xi = cone.Xi
    zi = inv(cone.z)

    DPhiX = cone.DPhiX

    matN = cone.matN

    # println("epi=", cone.point[1], ";")
    # println("X=", cone.X, ";")
    # println("Y=", cone.Y, ";")

    spectral_outer!(Xi, Ux, inv.(Λx), matN)

    g = cone.grad
    g[1] = -zi;

    @views g_x = g[cone.X_idxs]

    @. DPhiX = cone.XY_log
    @. matN = DPhiX * zi - Xi
    Hypatia.Cones.smat_to_svec!(g_x, matN, rt2)

    cone.grad_updated = true
    return cone.grad
end

function update_hessprod_aux(cone::EpiCondEntropyTri)
    @assert !cone.hessprod_aux_updated
    @assert cone.grad_updated

    (Λx, Ux) = cone.X_fact
    (Λy, Uy) = cone.Y_fact

    Δ2_log!(cone.Δ2x_log, Λx, cone.Λx_log)
    Δ2_log!(cone.Δ2y_log, Λy, cone.Λy_log)

    cone.hessprod_aux_updated = true
    return
end


function Hypatia.Cones.hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiCondEntropyTri{T},
) where {T <: Real}
    @assert cone.grad_updated
    cone.hessprod_aux_updated || update_hessprod_aux(cone)

    rt2 = cone.rt2
    Δ2x_log = cone.Δ2x_log
    Δ2y_log = cone.Δ2y_log
    (Λx, Ux) = cone.X_fact
    (Λy, Uy) = cone.Y_fact
    Xi = cone.Xi
    zi = inv(cone.z)

    DPhiX = cone.DPhiX

    HX = cone.HX
    HY = cone.HY

    matn = cone.matn
    matn2 = cone.matn2
    matn3 = cone.matn3
    matn4 = cone.matn4
    matN = cone.matN
    matN2 = cone.matN2
    matN3 = cone.matN3

    # println("epi=", cone.point[1], ";")
    # println("X=", cone.X, ";")
    # println("Y=", cone.Y, ";")

    @inbounds for j in 1:size(arr, 2)

        @views H = arr[:, j]

        # Get slices of vector and product
        Ht = H[1]
        @views Hypatia.Cones.svec_to_smat!(HX, H[cone.X_idxs], rt2)
        LinearAlgebra.copytri!(HX, 'U')
        pTr!(HY, HX, cone.sys, (cone.n1, cone.n2))

        # println("HX=", HX)
        # println("Ht=", Ht)

        @views prod_x = prod[cone.X_idxs, j]

        # t
        DPhiY_HZ = dot(DPhiX, HX)
        prod[1, j] = Ht - DPhiY_HZ
        prod[1, j] *= zi*zi
        
        # X
        fac = (DPhiY_HZ - Ht) * zi^2
        # D2PhiXXH
        spectral_outer!(matN, Ux', HX, matN2)
        # spectral_outer!(matN, Ux', Symmetric(HX, :U), matN2)
        @. matN *= Δ2x_log
        spectral_outer!(matN3, Ux, matN, matN2)
        # spectral_outer!(matN3, Ux, Symmetric(matN, :U), matN2)
        # D2PhiXYH
        spectral_outer!(matn4, Uy', HY, matn)
        # spectral_outer!(matn4, Uy', Symmetric(HY, :U), matn)
        @. matn = Δ2y_log * matn4
        spectral_outer!(matn3, Uy, matn, matn2)
        # spectral_outer!(matn3, Uy, Symmetric(matn, :U), matn2)
        idKron!(matN2, matn3, cone.sys, (cone.n1, cone.n2))
        # D2XH = D2XtH + D2XXH + D2XYH;
        @. matN = zi * (matN3 - matN2)
        @. matN += fac * DPhiX
        spectral_outer!(matN2, Xi, HX, matN3)
        # spectral_outer!(matN2, Xi, Symmetric(HX, :U), matN3)
        @. matN += matN2
        Hypatia.Cones.smat_to_svec!(prod_x, matN, rt2)     

    end



    return prod
end

function update_invhessprod_aux(cone::EpiCondEntropyTri)
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

    DPhiX = cone.DPhiX
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

    HX = cone.HX
    HY = cone.HY
    rt2i = 1 / rt2
    
    @. matN = 1 / (Λx' * Λx)
    @. Δ2x_comb_inv = 1 / (Δ2x_log*zi + matN)
    # D2x_inv = 1 ./ (obj.Dx' .* obj.Dx);
    # obj.D2x_comb_inv = 1 ./ (obj.D2x_log/obj.xi + D2x_inv);

    # Construct matrices
    # TODO
    k = 0
    @inbounds for j in 1:n, i in 1:j
        k += 1

        HX .= 0

        if cone.sys == 1
            @inbounds for l = 0:cone.n1-1
                copyto!(vecN, Ux[cone.n2*l + i, :])
                copyto!(vecN2, Ux[cone.n2*l + j, :])
                mul!(HX, vecN, vecN2', true, true)
            end
        else
            @inbounds for l = 1:cone.n2
                copyto!(vecN, Ux[cone.n2*(i - 1) + l, :])
                copyto!(vecN2, Ux[cone.n2*(j - 1) + l, :])
                mul!(HX, vecN, vecN2', true, true)
            end         
        end
            
        if i != j
            @. matN = HX + HX'
            @. HX = rt2i * matN          
        end
        @views UxK_k = UxK[:, k]
        Hypatia.Cones.smat_to_svec!(UxK_k, HX, rt2)

        copyto!(vecn, Uy[i, :])
        copyto!(vecn2, Uy[j, :])
        mul!(HY, vecn, vecn2')
        # HY .= Uy[i, :] * Uy[j, :]'
        if i != j
            @. matn = HY + HY'
            @. HY = rt2i * matn
        end

        
        # HYY
        @. matn = cone.z * HY / Δ2y_log
        spectral_outer!(matn3, Uy, matn, matn2)
        # spectral_outer!(matn3, Uy, Symmetric(matn, :U), matn2)
        @views HYY_k = HYY[:, k]
        Hypatia.Cones.smat_to_svec!(HYY_k, matn3, rt2)

    end

    Hypatia.Cones.smat_to_svec!(vecM, Δ2x_comb_inv, 1)
    @. UxK_temp = vecM * UxK
    mul!(KHxK, UxK', UxK_temp)
    @. HYY_KHxK = HYY - KHxK;

    cone.HYY_KHxK_chol = cholesky(Hermitian(HYY_KHxK))


    # # % Block elimination for gradient
    # spectral_outer!(matN, Ux', DPhiX, matN2)
    # # spectral_outer!(matN, Ux', Symmetric(DPhiX, :U), matN2)
    # @. matN *= Δ2x_comb_inv
    # spectral_outer!(matN3, Ux, matN, matN2)
    # # spectral_outer!(matN3, Ux, Symmetric(matN, :U), matN2)

    # pTr!(matn, matN3, cone.sys, (cone.n1, cone.n2))
    # smat_to_svec!(vecm, matn, rt2)
    # vecm .*= -1
    
    # vecm2 .= cone.HYY_KHxK_chol \ vecm

    # svec_to_smat!(matn, vecm2, rt2)
    # copytri!(matn, 'U')
    # idKron!(matN, matn, cone.sys, (cone.n1, cone.n2))
    # # spectral_outer!(matN4, Ux', Symmetric(matN, :U), matN2)
    # spectral_outer!(matN4, Ux', matN, matN2)
    # matN4 .*= Δ2x_comb_inv
    # spectral_outer!(matN, Ux, matN4, matN2)
    # # spectral_outer!(matN, Ux, Symmetric(matN4, :U), matN2)
    # @. H_inv_g_x = matN3 - matN

    # cone.DPhi_H_DPhi = dot(H_inv_g_x, DPhiX)


    cone.invhessprod_aux_updated = true
    return
end

function Hypatia.Cones.inv_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiCondEntropyTri{T},
) where {T <: Real}
    @assert cone.grad_updated
    cone.hessprod_aux_updated || update_hessprod_aux(cone)
    cone.invhessprod_aux_updated || update_invhessprod_aux(cone)

    rt2 = cone.rt2
    Δ2x_comb_inv = cone.Δ2x_comb_inv
    (Λx, Ux) = cone.X_fact

    DPhiX = cone.DPhiX
    HYY_KHxK = cone.HYY_KHxK
    H_inv_g_x = cone.H_inv_g_x

    matn = cone.matn
    matN = cone.matN
    matN2 = cone.matN2
    matN3 = cone.matN3
    matN4 = cone.matN4
    vecm = cone.vecm
    vecm2 = cone.vecm2

    HX = cone.HX
    HY = cone.HY

    # println()
    # println("epi=", cone.point[1], ";")
    # println("X=", cone.X, ";")
    # println("Y=", cone.Y, ";")

    @inbounds for j in 1:size(arr, 2)

        @views H = arr[:, j]

        # Get slices of vector and product
        Ht = H[1]
        @views Hypatia.Cones.svec_to_smat!(HX, H[cone.X_idxs], rt2)
        LinearAlgebra.copytri!(HX, 'U')
        pTr!(HY, HX, cone.sys, (cone.n1, cone.n2))

        # println("HX=", HX, ";")
        # println("Ht=", Ht, ";")


        # Compute combined direction
        @. matN3 = Ht * DPhiX
        @. matN3 += HX
        # W_H_DPhi = dot(H_inv_g_x, matN3)

        # Block elimination for variable
        # symmWX = Symmetric(WX, :U)
        # Ux_adjoint = adjoint(Ux)
        # mul!(matN, Ux', WX)
        # mul!(matN2, matN, Ux)
        # matN .= Ux' * WX * Ux
        spectral_outer!(matN, Ux', matN3, matN2)
        # spectral_outer!(matN, Ux_adjoint, symmWX, matN2)
        @. matN *= Δ2x_comb_inv
        spectral_outer!(matN3, Ux, matN, matN2)
        # spectral_outer!(matN3, Ux, Symmetric(matN, :U), matN2)

        pTr!(matn, matN3, cone.sys, (cone.n1, cone.n2))
        Hypatia.Cones.smat_to_svec!(vecm, matn, rt2)
        vecm .*= -1
        
        vecm2 .= cone.HYY_KHxK_chol \ vecm

        Hypatia.Cones.svec_to_smat!(matn, vecm2, rt2)
        LinearAlgebra.copytri!(matn, 'U')
        idKron!(matN, matn, cone.sys, (cone.n1, cone.n2))
        spectral_outer!(matN4, Ux', matN, matN2)
        # spectral_outer!(matN4, Ux', Symmetric(matN, :U), matN2)
        matN4 .*= Δ2x_comb_inv
        spectral_outer!(matN, Ux, matN4, matN2)
        # spectral_outer!(matN, Ux, Symmetric(matN4, :U), matN2)
        @. matN3 -= matN

        prod[1, j] = Ht * cone.z^2 + dot(matN3, DPhiX)

        @views H_inv_w_x = prod[cone.X_idxs, j]
        Hypatia.Cones.smat_to_svec!(H_inv_w_x, matN3, rt2)

    end

    # println("epi=", cone.point[1], ";")
    # println("X=", cone.X, ";")
    # println("Y=", cone.Y, ";")

    # other_prod = zeros(T, size(arr, 1), size(arr, 2))
    # hess_prod!(other_prod, prod, cone)
    # println("Err: ", sum(abs.(arr - other_prod), dims=1) )

        
    return prod
end

function update_dder3_aux(cone::EpiCondEntropyTri)
    @assert !cone.dder3_aux_updated
    @assert cone.hessprod_aux_updated

    Δ3!(cone.Δ3x_log, cone.Δ2x_log, cone.X_fact.values)
    Δ3!(cone.Δ3y_log, cone.Δ2y_log, cone.Y_fact.values)

    cone.dder3_aux_updated = true
    return
end

function Hypatia.Cones.dder3(cone::EpiCondEntropyTri{T}, dir::AbstractVector{T}) where {T <: Real}
    @assert cone.grad_updated
    cone.hessprod_aux_updated || update_hessprod_aux(cone)
    cone.dder3_aux_updated || update_dder3_aux(cone)

    n = cone.n
    N = cone.N
    rt2 = cone.rt2
    Δ2x_log = cone.Δ2x_log
    Δ3x_log = cone.Δ3x_log
    Δ2y_log = cone.Δ2y_log
    Δ3y_log = cone.Δ3y_log
    (Λx, Ux) = cone.X_fact
    (Λy, Uy) = cone.Y_fact
    Xi = cone.Xi
    zi = inv(cone.z)

    DPhiX = cone.DPhiX

    matn = cone.matn
    matn2 = cone.matn2
    matn3 = cone.matn3
    matN = cone.matN
    matN2 = cone.matN2
    matN3 = cone.matN3

    HX = cone.HX
    HY = cone.HY

    dder3 = cone.dder3

    # Get slices of vector and product
    Ht = dir[1]
    @views HZ = dir[2:end]
    @views Hypatia.Cones.svec_to_smat!(HX, dir[cone.X_idxs], rt2)
    LinearAlgebra.copytri!(HX, 'U')
    pTr!(HY, HX, cone.sys, (cone.n1, cone.n2))

    # println("epi=", cone.point[1], ";")
    # println("X=", cone.X, ";")

    # println("Ht=", Ht, ";")
    # println("HX=", HX, ";")

    # Precompute rotated directions
    UxHxUx = zeros(T, N, N)
    UyHyUy = zeros(T, n, n)
    
    spectral_outer!(UxHxUx, Ux', HX, matN)
    spectral_outer!(UyHyUy, Uy', HY, matn)
    # spectral_outer!(UxHxUx, Ux', Symmetric(HX, :U), matN)
    # spectral_outer!(UyHyUy, Uy', Symmetric(HY, :U), matn)

    # Hessian product of conditional entropy
    D2PhiXXH = zeros(T, N, N)

    @. matN = UxHxUx * Δ2x_log
    spectral_outer!(matN3, Ux, matN, matN2)
    # spectral_outer!(matN3, Ux, Symmetric(matN, :U), matN2)
    @. matn = UyHyUy * Δ2y_log
    spectral_outer!(matn3, Uy, matn, matn2)
    # spectral_outer!(matn3, Uy, Symmetric(matn, :U), matn2)
    idKron!(matN2, matn3, cone.sys, (cone.n1, cone.n2))

    @. D2PhiXXH = matN3 - matN2


    # Third directional derivatives of conditional entropy
    D3PhiXXX = zeros(T, N, N)
    Δ2_frechet!(D3PhiXXX, UxHxUx .* Δ3x_log, Ux, UxHxUx, matN2, matN3)
    Δ2_frechet!(matn, UyHyUy .* Δ3y_log, Uy, UyHyUy, matn2, matn3)
    idKron!(matN, matn, cone.sys, (cone.n1, cone.n2))
    @. D3PhiXXX -= matN


    # Third directional derivatives of the barrier function
    DPhiZ_H = dot(DPhiX, HX)
    D2PhiZ_HH = dot(D2PhiXXH, HX)
    χ = Ht - DPhiZ_H
    dder3_t = -2 * zi^3 * χ^2 - zi^2 * D2PhiZ_HH;

    matN = -dder3_t * DPhiX -
           2 * zi^2 * χ * (D2PhiXXH) +
           zi * (D3PhiXXX) -
           2*Xi * HX * Xi * HX * Xi;

    dder3[1] = dder3_t

    @views dder3_X = dder3[cone.X_idxs]
    @views Hypatia.Cones.smat_to_svec!(dder3_X, matN, rt2)

    @. dder3 *= -0.5 * 0

    return dder3
end

#-----------------------------------------------------------------------------------------------------

function pTr!(ptrX::Matrix{T}, X::Matrix{T}, sys::Int = 2, dim::Union{Tuple{Int, Int}, Nothing} = nothing) where {T <: Real}

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

function idKron!(kronI::Matrix{T}, X::Matrix{T}, sys::Int = 2, dim::Union{Tuple{Int, Int}, Nothing} = nothing) where {T <: Real}

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
    LinearAlgebra.copytri!(Δ2, 'U')
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