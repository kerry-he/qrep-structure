using LinearAlgebra
import Hypatia.Solvers
import Hypatia.Cones

import Hypatia.posdef_fact_copy!

mutable struct ElimSystemSolver{T <: Real} <: Hypatia.Solvers.SystemSolver{T}
    rhs_sub::Hypatia.Solvers.Point{T}
    sol_sub::Hypatia.Solvers.Point{T}
    sol_const::Hypatia.Solvers.Point{T}
    rhs_const::Hypatia.Solvers.Point{T}

    # H::Matrix{T}
    HA::Matrix{T}
    AHA::Matrix{T}
    AHA_chol::Factorization{T}

    use_sqrt_hess_cones::Vector{Bool}

    function ElimSystemSolver{T}() where {T <: Real}
        syssolver = new{T}()
        return syssolver
    end
end

function Hypatia.Solvers.load(syssolver::ElimSystemSolver{T}, solver::Hypatia.Solvers.Solver{T}) where {T <: Real}
    model = solver.model
    (n, p) = (model.n, model.p)

    # syssolver.H   = zeros(T, n, n)
    syssolver.HA  = zeros(T, n, p)
    syssolver.AHA = zeros(T, p, p)

    return syssolver
end


function Hypatia.Solvers.update_lhs(
    syssolver::ElimSystemSolver{T},
    solver::Hypatia.Solvers.Solver{T},
) where {T <: Real}
    model = solver.model
    cones = model.cones
    cone_idxs = model.cone_idxs
    n = model.n

    A = model.A
    # H = syssolver.H
    HA = syssolver.HA
    AHA = syssolver.AHA

    # blk_hess_prod!(H, Matrix{T}(I, n, n), cones, cone_idxs)
    blk_inv_hess_prod!(HA, A', cones, cone_idxs)
    mul!(AHA, A, HA)

    syssolver.AHA_chol = factorize(AHA)

    return syssolver
end


function Hypatia.Solvers.solve_system(
    syssolver::ElimSystemSolver{T},
    solver::Hypatia.Solvers.Solver{T},
    sol::Hypatia.Solvers.Point{T},
    rhs::Hypatia.Solvers.Point{T},
) where {T <: Real}
    model = solver.model
    cones = model.cones
    cone_idxs = model.cone_idxs

    n = model.n
    mu = solver.mu
    tau = solver.point.tau[]
    A = model.A
    c = model.c
    b = model.b

    # println("A=", A)
    # println("c=", c)
    # println("b=", b)
    # println("mu=", mu)
    # println("tau=", tau)
    # println("rhs=", rhs.vec)

    # temp_H = zeros(T, n, n)
    # blk_hess_prod!(temp_H, Matrix{T}(I, n, n), cones, cone_idxs)

    # println("H=", temp_H)
    # println()

    Hrs = zeros(T, n)
    blk_inv_hess_prod!(Hrs, rhs.s, cones, cone_idxs)

    (xr, yr, zr) = solve_subsystem(syssolver, solver, rhs.x, rhs.y, rhs.z + Hrs)
    (xb, yb, zb) = solve_subsystem(syssolver, solver, c, b, zeros(T, n))

    sol.tau[] = (rhs.tau[] + rhs.kap[] + c'*xr + b'*yr) / (mu/tau/tau + c'*xb + b'*yb)
    copyto!(sol.x, xr - sol.tau[] * xb)
    copyto!(sol.y, yr - sol.tau[] * yb)
    copyto!(sol.z, zr - sol.tau[] * zb)
    # sol.x = xr - sol.tau[] * xb
    # sol.y = yr - sol.tau[] * yb
    # sol.z = zr - sol.tau[] * zb
    
    Hz = zeros(T, n)
    blk_inv_hess_prod!(Hz, sol.z, cones, cone_idxs)
    copyto!(sol.s, Hrs - Hz)
    # sol.s = (Hrs - Hz) / mu
    sol.kap[] = rhs.kap[] - mu/tau/tau * sol.tau[]

    return sol
end


function solve_subsystem(
    syssolver::ElimSystemSolver{T},
    solver::Hypatia.Solvers.Solver{T},
    rx::AbstractVector{T},
    ry::AbstractVector{T},
    rz::AbstractVector{T},
) where {T <: Real}
    model = solver.model
    cones = model.cones
    cone_idxs = model.cone_idxs

    (n, p) = (model.n, model.p)
    A = model.A

    Hrx = zeros(T, n)
    blk_inv_hess_prod!(Hrx, rx, cones, cone_idxs)
    rhs_y = A * Hrx + A*rz + ry
    y = syssolver.AHA_chol \ rhs_y

    HAy = zeros(T, n)
    blk_inv_hess_prod!(HAy, A'*y, cones, cone_idxs)
    x = (Hrx - HAy) + rz

    z = -rx + A'*y;
    # z = zeros(T, n)
    # blk_hess_prod!(z, rz - x, cones, cone_idxs)

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
