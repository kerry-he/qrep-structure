using LinearAlgebra
import Hypatia
import Hypatia.Cones
import Hypatia.Solvers

include("systemsolvers/elim.jl")

T = Float64

function main()
    (Xn, Xm) = (3, 4)
    dim = Xn * Xm
    c = vcat(one(T), zeros(T, dim))
    A = hcat(zeros(T, dim, 1), Matrix{T}(I, dim, dim))
    b = rand(T, dim)
    G = -one(T) * I
    h = vcat(zero(T), rand(T, dim))
    cones = [Cones.EpiNormSpectral{T, T}(Xn, Xm)]
    model = Hypatia.Models.Model{T}(c, A, b, G, h, cones)

    solver = Solvers.Solver{T}(verbose = true, reduce = false, preprocess = false, syssolver = ElimSystemSolver{T}())
    Solvers.load(solver, model)
    Solvers.solve(solver)
end
 
main()