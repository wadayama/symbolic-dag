# Tutorial 2 — Proving conditional independence (the rewrite engine)

Tutorial 1 built a symbolic CMI and evaluated it. This tutorial does something a
numerical library cannot: it **proves**, symbolically and for every dimension at
once, when a conditional mutual information is exactly zero.

By the end you will:

- Understand that `I(A; B | C) = 0` iff the **cross conditional covariance**
  `Σ_{AB|C}` is the zero matrix.
- See the rewrite engine prove this for the chain and fork, and refuse to for the
  (opened) collider.
- Understand why the Hermitian assumption and a rule *strategy* are both needed.

---

## 1. The idea

For Gaussian variables, `I(A; B | C) = 0` exactly when `A` and `B` are
conditionally independent given `C`, which holds iff the off-diagonal block of the
joint conditional covariance vanishes:

```
Σ_{AB|C} = Σ_{A,B} − Σ_{A,C} Σ_{C,C}^{-1} Σ_{C,B} = 0.
```

`symbolic-dag` stores this cross block on every `SymbolicCMI` (as `.cross`) and
exposes `.is_conditionally_independent()`, which asks the rewrite engine to reduce
it to the zero matrix.

---

## 2. The chain: `I(X; Z | Y) = 0`

```python
import sympy as sp
from symbolic_dag import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
    hermitian,
)

d = sp.Symbol("d", positive=True, integer=True)
A, B = sp.MatrixSymbol("A", d, d), sp.MatrixSymbol("B", d, d)
SX, SY, SZ = (hermitian(s, d) for s in ("Sigma_X", "Sigma_Y", "Sigma_Z"))

K = compute_k_blocks_multiroot(
    num_nodes=3, roots=[0], parents={1: [0], 2: [1]},
    edge_mats={(1, 0): A, (2, 1): B}, root_covs={0: SX}, noise_covs={1: SY, 2: SZ},
)
I = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])   # I(X; Z | Y)

print("cross block before:", I.cross)
print("conditionally independent:", I.is_conditionally_independent())   # True
```

The cross block starts as a non-trivial expression in `A, B, Σ_X, Σ_Y`; the
structural rewrite phase (distribute `Adjoint`, apply `Adjoint(Σ) → Σ`, cancel
`M^{-1} M → I`, combine `X + (−X) → 0`) collapses it to the zero matrix. That is a
machine proof of `I(X; Z | Y) = 0` valid for **every** dimension `d`.

---

## 3. The fork, and why the Hermitian assumption matters

The fork `X → Y`, `X → Z` is conditionally independent given the common cause:

```python
K = compute_k_blocks_multiroot(
    num_nodes=3, roots=[0], parents={1: [0], 2: [0]},
    edge_mats={(1, 0): A, (2, 0): B}, root_covs={0: SX}, noise_covs={1: SY, 2: SZ},
)
I = conditional_mutual_information_from_k(K, A=[1], B=[2], C=[0])   # I(Y; Z | X)
print(I.is_conditionally_independent())   # True
```

Here the cross block reduces to `A (Σ_X − Σ_X^H) B^H`, which is the zero matrix
**iff `Σ_X` is Hermitian**. That is exactly the fact `sympy` does not know on its
own — and exactly what the `hermitian(...)` tag supplies. Build `Σ_X` as a plain
`MatrixSymbol` instead and the engine can no longer prove independence: the
`Σ_X^H` term survives.

> **Strategy matters.** The rewrite rules are not confluent as a flat set: an
> aggressive expansion rule (Woodbury) can rewrite a matrix inverse *before* the
> inverse-cancellation that the proof needs, leaving a correct-but-unrecognised
> expression. The engine therefore runs in **phases** — structural normalization
> first, low-rank expansion only afterwards. `is_conditionally_independent()`
> uses the structural phase only.

---

## 4. The collider: conditioning *opens* a dependence

For the collider `X → Z ← Y` with independent sources, `X` and `Y` are marginally
independent, but conditioning on the common effect `Z` opens a dependence:

```python
SX2, SY2, SZ2 = (hermitian(s, d) for s in ("Sigma_X", "Sigma_Y", "Sigma_Z"))
K = compute_k_blocks_multiroot(
    num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
    edge_mats={(2, 0): A, (2, 1): B}, root_covs={0: SX2, 1: SY2}, noise_covs={2: SZ2},
)
I_marg = conditional_mutual_information_from_k(K, A=[0], B=[1], C=[])
I_cond = conditional_mutual_information_from_k(K, A=[0], B=[1], C=[2])
print(I_marg.is_conditionally_independent())   # True  (independent sources)
print(I_cond.is_conditionally_independent())   # False (collider opened by Z)
```

The marginal cross block is zero by construction (the K-recursion sets distinct
roots' cross-covariance to the zero matrix — the independence assumption); the
conditioned cross block is a genuinely non-zero matrix.

---

## 5. What is next?

Tutorial 3 turns from *proving things are zero* to *deriving non-zero closed
forms*: the Wirtinger gradient of a CMI with respect to a precoder, and its
stationarity (KKT) condition.
