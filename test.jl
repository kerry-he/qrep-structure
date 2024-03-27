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
# V = randStinespringOperator(T, ni, no, ne)

V = [[-0.42435477 -0.02503615 -0.56239342 -0.10982374]
     [ 0.04504629 -0.24455436  0.05238686  0.06787125]
     [ 0.17952156 -0.2788913  -0.28060775 -0.06578033]
     [-0.03139112  0.17185153  0.15482007 -0.07751487]
     [-0.12762372  0.02089742  0.57991927  0.34284975]
     [ 0.05808248 -0.07116984  0.14353133  0.05145701]
     [ 0.08983062 -0.01722523 -0.30570814  0.25967733]
     [-0.22105663 -0.15485544  0.07067319 -0.41961934]
     [ 0.31330474  0.12643154  0.0835     -0.21668587]
     [-0.13832104  0.08023248 -0.03845787  0.24786989]
     [ 0.13508919 -0.30452615  0.22677924 -0.27482441]
     [-0.3478387   0.0356285   0.18496465 -0.06173028]
     [-0.49758569  0.45006581  0.08125296 -0.25443166]
     [ 0.29236514  0.41963828 -0.01450109 -0.30769973]
     [ 0.31865413  0.54059786 -0.12629999  0.0973069 ]
     [ 0.13015662 -0.12306487  0.07774004 -0.50266847]]

K = QuantMutualInformation{T}(ni, no, ne, V)
Cones.setup_extra_data!(K)
K.point = Cones.set_initial_point!(zeros(T, K.dim), K)
K.grad = zeros(T, K.dim)

Cones.reset_data(K)
Hypatia.Cones.update_feas(K)


# while true
#     Cones.reset_data(K)
#     point = randn(T, K.dim)
#     Hypatia.Cones.load_point(K, point)
    
#     if Hypatia.Cones.update_feas(K)
#         break
#     end
# end

H = randn(T, K.dim)
x0 = copyto!(zeros(T, K.dim), K.point)
f0 = K.fval
g0 = Hypatia.Cones.update_grad(K)


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
g1 = Hypatia.Cones.update_grad(K)


println("Gradient test (FDD=0): ", norm(0.5 * (g0 + g1) - ((f1 .- f0) ./ ϵ)))
println("Gradient test (ID=nu): ", (-g0' * x0))