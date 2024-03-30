using LinearAlgebra
import Hypatia
import Hypatia.Cones
import Hypatia.Solvers

include("cones/quantratedist.jl")
include("cones/quantcoherentinf.jl")
include("cones/quantmutualinf.jl")
include("utils/helper.jl")

T = Float64

ϵ = 1e-8
ni = 4
no = 3
ne = 2
# V = randEBChannel(T, ni, no, ne)

# N(x)  = pTr!(zeros(T, no, no), V*x*V', 2, (no, ne))
# Nc(x) = pTr!(zeros(T, ne, ne), V*x*V', 1, (no, ne))

K = QuantRateDistortion{T}(ni)
# K = QuantCoherentInformation{T}(ni, no, ne, N, Nc)
# K = QuantMutualInformation{T}(ni, no, ne, V)
Cones.setup_extra_data!(K)
K.point = Cones.set_initial_point!(zeros(T, K.dim), K)
K.grad = zeros(T, K.dim)
K.dder3 = zeros(T, K.dim)

Cones.reset_data(K)
Hypatia.Cones.update_feas(K)

while true
    Cones.reset_data(K)
    point = randn(T, K.dim)
    Hypatia.Cones.load_point(K, point)
    
    if Hypatia.Cones.update_feas(K)
        break
    end
end

H = randn(T, K.dim)
x0 = copyto!(zeros(T, K.dim), K.point)
f0 = K.fval
g0 = copyto!(zeros(T, K.dim), Hypatia.Cones.update_grad(K))
H0 = Hypatia.Cones.hess_prod!(zeros(T, K.dim), H, K)
T0 = Hypatia.Cones.dder3(K, H)


f1 = zeros(T, K.dim)
for i in 1:K.dim
    Cones.reset_data(K)
    point = copyto!(zeros(T, K.dim), x0)
    point[i] += ϵ
    Hypatia.Cones.load_point(K, point)
    Hypatia.Cones.update_feas(K)
    f1[i] = K.fval
end

x1 = x0 + ϵ*H
Cones.reset_data(K)
Hypatia.Cones.load_point(K, x1)
Hypatia.Cones.update_feas(K)
g1 = copyto!(zeros(T, K.dim), Hypatia.Cones.update_grad(K))
H1 = Hypatia.Cones.hess_prod!(zeros(T, K.dim), H, K)
T1 = Hypatia.Cones.dder3(K, H)



println("Gradient test (FDD=0): ", norm(0.5 * (g0 + g1) - ((f1 .- f0) ./ ϵ)))
println("Gradient test (ID=nu): ", (-g0' * x0))

println("Hessian test (ID=0): ",  norm(Hypatia.Cones.hess_prod!(zeros(T, K.dim), x0, K) + g0))
println("Hessian test (ID=nu): ", dot(Hypatia.Cones.hess_prod!(zeros(T, K.dim), x0, K), x0))
println("Hessian test (FDD=0): ", norm(0.5 * (H0 + H1) - ((g1 .- g0) ./ ϵ)))

println("Inv Hessian test (ID=0): ", norm(H .- Hypatia.Cones.inv_hess_prod!(zeros(T, K.dim), H1, K)))
println("Inv Hessian test (ID=nu): ", dot(Hypatia.Cones.inv_hess_prod!(zeros(T, K.dim), g0, K), g0))

print("TOA test: ", norm(0.5 * (T0 .+ T1) - ((H1 .- H0) ./ ϵ)))