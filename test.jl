using LinearAlgebra
import Hypatia
import Hypatia.Cones
import Hypatia.Solvers

include("cones/quantmutualinf.jl")
include("utils/helper.jl")

T = Float64

ϵ = 1e-8
ni = 4
no = 4
ne = 4
V = randStinespringOperator(T, ni, no, ne)

K = QuantMutualInformation{T}(ni, no, ne, V)
Cones.setup_extra_data!(K)
K.point = Cones.set_initial_point!(zeros(T, K.dim), K)
K.grad = zeros(T, K.dim)

while true
    Cones.reset_data(K)
    x0 = randn(T, K.dim)
    Hypatia.Cones.load_point(K, x0)
    
    if Hypatia.Cones.update_feas(K)
        break
    end
end

H = randn(T, K.dim)
x0 = copyto!(zeros(T, K.dim), K.point)
f0 = K.fval
g0 = Hypatia.Cones.update_grad(K)


f1 = zeros(T, K.dim)
for i in 1:K.dim
    Cones.reset_data(K)
    x1 = copyto!(zeros(T, K.dim), x0)
    x1[i] += ϵ
    Hypatia.Cones.load_point(K, x1)
    Hypatia.Cones.update_feas(K)
    f1[i] = K.fval
end

x1 = x0 + ϵ*H
Cones.reset_data(K)
Hypatia.Cones.load_point(K, x1)
Hypatia.Cones.update_feas(K)
g1 = Hypatia.Cones.update_grad(K)


println("Gradient test (FDD=0): ", norm(0.5 * (g0 + g1) - ((f1 .- f0) ./ ϵ)))
println("Gradient test (ID=nu): ", (-g0' * x0))