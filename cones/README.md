# Cones

This folder includes custom cone oracles to be used with the Hypatia solver. We provide some additional detail about these cones below. For all cones, `T` is used to define the variable types (e.g., `T=Float64`).

### Quantum conditional entropy cone
We implement a cone for the epigraph of quantum conditional entropy

$$  \mathcal{K}_{\text{qce}} \coloneqq \text{cl} \\{ (t, X)\in\mathbb{R}\times\mathbb{H}^{nm}\_{\++} : t \geq -S(X) + S(\text{tr}_k[X]) \\}. $$

This cone is called using `QuantCondEntropy{T}(n, m, k)` where
- `n`: Dimension of the first system
- `m`: Dimension of the second system
- `k`: Which system is being traced out (`k=1` or `k=2`)

### Quantum mutual information
We implement a cone for the epigraph of the (homogenized) quantum mutual information

$$  \mathcal{K}_{\text{qce}} \coloneqq  \text{cl} \\{ (t, X)\in\mathbb{R}\times\mathbb{H}^{n}\_{\++} : t \geq -S(X) - S(\text{tr}_2[VXV^\dagger]) + S(\text{tr}_1[VXV^\dagger]) + S(\text{tr}[X]) \\}. $$

This cone is called using `QuantMutualInformation{T}(n, m, p, V)` where
- `n`: Dimension of input system
- `m`: Dimension of output system
- `p`: Dimension of environment
- `V`: Stinespring isometry of a quantum channel

### Quantum coherent information
We implement a cone for the epigraph of the quantum coherent information for degradable channels, i.e., channels $\mathcal{N}$ such that the complementary channel $\mathcal{N}\_{\text{c}}$ satisfies $\mathcal{N}\_{\text{c}} = \Xi \circ \mathcal{N}$ for some channel $\Xi$.

$$  \mathcal{K}_{\text{qce}} \coloneqq  \text{cl} \\{ (t, X)\in\mathbb{R}\times\mathbb{H}^{n}\_{\++} : t \geq -S(\mathcal{N}(X)) + S(\text{tr}_1[W\mathcal{N}(X)W^\dagger]) \\}. $$

This cone is called using `QuantMutualInformation{T}(n, m, p, N, Nc)` where
- `n`: Dimension of input system of $\mathcal{N}$
- `m`: Dimension of output system of $\mathcal{N}$
- `p`: Dimension of environment of $\mathcal{N}$
- `N`: Anonymous function representing the quantum channel $\mathcal{N}$
- `Nc`: Anonymous function representing the quantum channel $\mathcal{N}_{\text{c}}$

### Quantum key rate
We implement a cone to compute for quantum key rates. This is done by implementing the cone

$$  \mathcal{K}_{\text{qce}} \coloneqq  \text{cl} \\{ (t, X)\in\mathbb{R}\times\mathbb{H}^{n}\_{\++} : t \geq -S(\mathcal{G}(X)) + S(\mathcal{Z}(\mathcal{G}(X))) \\}. $$

Where $\mathcal{Z}$ is a pinching channel, i.e., maps off-diagonal blocks of a matrix to zero. This cone is called using `QuantKeyRate{T, R}(Klist, Zlist)` where
- `Klist`: List of Kraus operators $K_i$ of $\mathcal{G}$, where $\mathcal{G}(X)=\sum_i K_i X K_i^\dagger$
- `Zlist`: List of Kraus operators $Z_i$ of $\mathcal{Z}$, where $\mathcal{Z}(X)=\sum_i Z_i X Z_i^\dagger$
