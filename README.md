
# Exploiting Structure in Quantum Relative Entropy Programs

## About

This repository contains code to solve quantum relative entropy programs of the form

$$\min_{X\in\mathbb{H}^{n}} \quad S({\mathcal{G}(X)}\\|{\mathcal{H}(X)}), \quad \text{subj. to} \quad A(X)=b, \ X\succeq0,$$

by implementing efficient cone oracles for the cone

$$  \mathcal{K}_{\text{qre}}^{\mathcal{G}, \mathcal{H}} \coloneqq  \text{cl} \\{ (t, X)\in\mathbb{R}\times\mathbb{H}^n\_{\++} : t \geq S({\mathcal{G}(X)}\\|{\mathcal{H}(X)}) \\}, $$

to use with the generic conic programming software [Hypatia](https://github.com/jump-dev/Hypatia.jl). We currently have implementations for the following variants of quantum relative entropy.

| Function | Code |
| --- | --- |
| Quantum conditional entropy | `QuantCondEntropy{T}(n, m, k)` 
| Quantum mutual information | `QuantMutualInformation{T}(n, m, p, V)` |
| Quantum coherent information (for degradable channels) | `QuantCoherentInformation{T}(n, m, p, N, Nc)` |
| Quantum key rate | `QuantKeyRate{T, R}(Klist, Zlist)` |

See the [cones folder](https://github.com/kerry-he/qrep-structure/tree/main/cones) for more detail about these cone oracles.

## Getting started

The main dependencies of our code are Julia and Hypatia. This can be installed by following the installation instructions [here](https://github.com/jump-dev/Hypatia.jl/tree/master). 

## Usage

Here, we show an example of how we can compute the entanglement assisted channel capacity of a qubit amplitude damping channel from [here](https://github.com/hfawzi/cvxquad/blob/master/examples/entanglement_assisted_capacity.m). 

	import Hypatia
	include("cones/quantmutualinf.jl")
	include("systemsolvers/elim.jl")

	T = Float64

	# Define problem data
	(ni, no, ne) = (2, 2, 2)
	gamma = 0.2
	V = [1 0; 0 sqrt(gamma); 0 sqrt(1 - gamma); 0 0] # Stinespring isometry of amplitude damping channel

	# Express problem as standard form conic program
	c     = [1.; 0.; 0.; 0.] / log(2.) # Convert units to bits
	A     = [0.  1.  0.  1.] # Unit trace constraint
	b     = [1.]
	G     = -1.0*I
	h     = [0.; 0.; 0.; 0.]
	cones = [QuantMutualInformation{T}(ni, no, ne, V)]

	# Parse into Hypatia and solve
	model  = Hypatia.Models.Model{T}(c, A, b, G, h, cones)
	solver = Solvers.Solver{T}(reduce=false, syssolver=ElimSystemSolver{T}())
	Solvers.load(solver, model)
	Solvers.solve(solver)

Additional examples, including 

 - quantum key rates,
 - quantum rate-distortion function,
 - entanglement assisted channel capacity,
 - quantum-quantum channel capacity for degradable channels, and
 - ground state energy of Hamiltonians,

can be found in the [examples folder](https://github.com/kerry-he/qrep-structure/tree/main/examples).

Note that we have also implemented a simple block elimination method to solve the Newton equations, which can improve computational performance over Hypatia's default Newton system solver when `G` is a square diagonal matrix.
