using LinearAlgebra
import Hypatia.Solvers
import Hypatia.Cones

# Solves the following square Newton system
#            - A'*y     - c*tau + z         = rx
#        A*x            - b*tau             = ry
#      -c'*x - b'*y                 - kappa = rtau
#     mu*H*x                    + z         = rz 
#                    mu/t^2*tau     + kappa = rkappa
# for (x, y, z, tau, kappa) given right-hand residuals (rx, ry, rz, rtau, rkappa)
# by using elimination.

mutable struct ElimSystemSolver{T <: Real} <: Hypatia.Solvers.SystemSolver{T}
    HA::Matrix{T}
    AHA::Matrix{T}
    AHA_chol::Union{Factorization{T}, T}

    xb::Vector{T}
    yb::Vector{T}
    zb::Vector{T}

    function ElimSystemSolver{T}() where {T <: Real}
        syssolver = new{T}()
        return syssolver
    end
end

function Hypatia.Solvers.load(syssolver::ElimSystemSolver{T}, solver::Hypatia.Solvers.Solver{T}) where {T <: Real}
    model = solver.model
    (n, p) = (model.n, model.p)

    syssolver.HA  = zeros(T, n, p)
    syssolver.AHA = zeros(T, p, p)

    syssolver.xb = zeros(T, model.n)
    syssolver.yb = zeros(T, model.p)
    syssolver.zb = zeros(T, model.q)

    return syssolver
end


function Hypatia.Solvers.update_lhs(
    syssolver::ElimSystemSolver{T},
    solver::Hypatia.Solvers.Solver{T},
) where {T <: Real}
    model = solver.model
    n = model.n    

    A = model.A
    HA = syssolver.HA
    AHA = syssolver.AHA

    # Compute Schur complement matrix
    blk_inv_hess_prod!(HA, A', model.cones, model.cone_idxs)
    mul!(AHA, A, HA)
    syssolver.AHA_chol = factorize(AHA)

    # Compute constant 3x3 subsystem
    (xb, yb, zb) = solve_subsystem(syssolver, solver, model.c, model.b, model.h, zeros(T, n))
    copyto!(syssolver.xb, xb)
    copyto!(syssolver.yb, yb)
    copyto!(syssolver.zb, zb)

    return syssolver
end


function Hypatia.Solvers.solve_system(
    syssolver::ElimSystemSolver{T},
    solver::Hypatia.Solvers.Solver{T},
    sol::Hypatia.Solvers.Point{T},
    rhs::Hypatia.Solvers.Point{T},
) where {T <: Real}
    model = solver.model

    mu = solver.mu
    tau = solver.point.tau[]
    (c, b, h)    = (model.c, model.b, model.h)
    (xb, yb, zb) = (syssolver.xb, syssolver.yb, syssolver.zb)

    # Compute 3x3 subsystem
    (xr, yr, zr) = solve_subsystem(syssolver, solver, rhs.x, rhs.y, rhs.z, rhs.s)

    # Backsubstitute solve system
    sol.tau[] = (rhs.tau[] + rhs.kap[] + c'*xr + b'*yr + h'*zr) / (mu/tau/tau + c'*xb + b'*yb + h'*zb)
    copyto!(sol.x, xr - sol.tau[] * xb)
    copyto!(sol.y, yr - sol.tau[] * yb)
    copyto!(sol.z, zr - sol.tau[] * zb)
    copyto!(sol.s, -model.G*sol.x - rhs.z + sol.tau[]*h)
    sol.kap[] = rhs.kap[] - mu/tau/tau * sol.tau[]

    return sol
end


function solve_subsystem(
    syssolver::ElimSystemSolver{T},
    solver::Hypatia.Solvers.Solver{T},
    rx::AbstractVector{T},
    ry::AbstractVector{T},
    rz::AbstractVector{T},
    rs::AbstractVector{T},
) where {T <: Real}
    model = solver.model
    cones = model.cones
    cone_idxs = model.cone_idxs

    n = model.n
    A = model.A

    # Compute y solution
    Hrxrs = zeros(T, n)
    blk_inv_hess_prod!(Hrxrs, rx + rs, cones, cone_idxs)
    rhs_y = A * (Hrxrs + rz) + ry
    y = syssolver.AHA_chol \ rhs_y

    # Compute x solution
    HrxrsAy = zeros(T, n)
    blk_inv_hess_prod!(HrxrsAy, rx + rs - A'*y, cones, cone_idxs)
    x = HrxrsAy + rz

    # Compute z solution
    z = -rx + A'*y

    return (x, y, z)
end

function blk_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cones::Vector{Cones.Cone{T}},
    cone_idxs::Vector{UnitRange{Int}}
) where {T <: Real}

    @inbounds for k in eachindex(cones)
        cone_idxs_k = cone_idxs[k]
        @views arr_k = arr[cone_idxs_k, :]
        @views prod_k = prod[cone_idxs_k, :]
        if Cones.use_dual_barrier(cones[k])
            Cones.inv_hess_prod!(prod_k, arr_k, cones[k])
        else
            Cones.hess_prod!(prod_k, arr_k, cones[k])
        end
    end

    return
end

function blk_inv_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cones::Vector{Cones.Cone{T}},
    cone_idxs::Vector{UnitRange{Int}}
) where {T <: Real}

    @inbounds for k in eachindex(cones)
        cone_idxs_k = cone_idxs[k]
        @views arr_k = arr[cone_idxs_k, :]
        @views prod_k = prod[cone_idxs_k, :]
        if Cones.use_dual_barrier(cones[k])
            Cones.hess_prod!(prod_k, arr_k, cones[k])
        else
            Cones.inv_hess_prod!(prod_k, arr_k, cones[k])
        end
    end

    return
end
