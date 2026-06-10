# Tutorial 1 — Installation and your first symbolic CMI

This tutorial installs `symbolic-dag` and computes a symbolic conditional mutual
information on the smallest interesting gadget — a three-node Markov chain
`X → Y → Z` — keeping the gains and covariances as opaque symbols.

By the end you will:

- Have a working `symbolic-dag` environment.
- Understand the chain as a 3-node linear Gaussian DAG.
- Have built a symbolic CMI with `conditional_mutual_information_from_k` and read
  off its lazy log-determinant form.
- Have evaluated it numerically and checked it against an independent NumPy
  oracle.

---

## 1. Install the library

`symbolic-dag` is a small pure-Python package built on SymPy and NumPy. Use
[`uv`](https://docs.astral.sh/uv/) to manage the environment.

```bash
git clone https://github.com/wadayama/symbolic-dag.git
cd symbolic-dag
uv sync          # installs sympy, numpy, pytest into a fresh .venv
```

Confirm the install:

```bash
uv run pytest
```

You should see the suite pass. (A handful of cross-validation tests against the
sibling `cmi-dag` library are skipped unless that repository is available
locally; that is expected and is covered in Tutorial 4.)

---

## 2. The model

A Gaussian Markov chain `X → Y → Z`:

```
   X  ──►  [ A ]  ──►  Y = A X + N_Y  ──►  [ B ]  ──►  Z = B Y + N_Z
```

As a DAG this is three nodes with one root:

- Node `V_0 = X` is the source root, with input covariance `Σ_X` (Hermitian PD).
- Node `V_1 = Y` has parent `{X}` and edge gain `A`, noise `Σ_Y`.
- Node `V_2 = Z` has parent `{Y}` and edge gain `B`, noise `Σ_Z`.

Everything is **complex** (Wirtinger convention); `^H` denotes the conjugate
transpose. The matrices are kept symbolic, so the dimension `d` can itself be a
symbol — the result holds for every dimension at once.

---

## 3. Build the symbolic CMI

```python
import sympy as sp
from symbolic_dag import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
    hermitian,
)

d = sp.Symbol("d", positive=True, integer=True)        # symbolic dimension
A, B = sp.MatrixSymbol("A", d, d), sp.MatrixSymbol("B", d, d)
SX, SY, SZ = (hermitian(s, d) for s in ("Sigma_X", "Sigma_Y", "Sigma_Z"))

K = compute_k_blocks_multiroot(
    num_nodes=3, roots=[0], parents={1: [0], 2: [1]},
    edge_mats={(1, 0): A, (2, 1): B},
    root_covs={0: SX}, noise_covs={1: SY, 2: SZ},
)

I = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[])   # I(X; Z)
for sign, M in I.logdet_terms:
    print(f"  {'+' if sign > 0 else '-'} log det  {M}")
```

What just happened:

- `compute_k_blocks_multiroot` propagated `Σ_X` through the chain, producing the
  canonical K-blocks `K[(0,0)] = Σ_X`, `K[(1,0)] = A Σ_X`,
  `K[(1,1)] = A Σ_X A^H + Σ_Y`, `K[(2,1)]`, `K[(2,2)]`, … as **matrix
  expressions**.
- `conditional_mutual_information_from_k(K, A, B, C)` assembled the Schur-complement
  conditional covariances and returned a **lazy** `SymbolicCMI`: a sum of signed
  log-determinant terms (here `+log det Σ_{X}`, `+log det Σ_{Z}`,
  `−log det Σ_{XZ}`), held symbolically — not expanded into one giant formula.

`hermitian(...)` creates covariance symbols the engine knows are Hermitian PD; it
is what later lets us *prove* identities (Tutorial 2).

---

## 4. Check it numerically (PyTorch, one call)

`symbolic-dag` is PyTorch-oriented for its numerics (aligned with its numerical
sibling `cmi-dag`). The simplest way to verify a symbolic CMI is one call: it
draws a random complex point at a concrete dimension, evaluates the CMI in
PyTorch, and compares it against an independent computation.

```python
print(I.check(dim=2))
#   {'passed': True, 'max_abs_err': 1.8e-14, 'samples': 4}
```

Under the hood, `to_torch` lowers the symbolic CMI to a **differentiable** torch
scalar, so `I.torch_value(subs, dim)` also drops straight into your own PyTorch
experiments and autograd (used for the gradient checks in Tutorial 3).

> The library also ships a torch-free NumPy oracle (`numpy_cmi`) used internally
> as an extra independent cross-check; you rarely need it directly.

---

## 5. What is next?

- **Tutorial 2** uses the rewrite engine to *prove* conditional independence: for
  this chain, `I(X; Z | Y) = 0` as a matrix identity, for every dimension.
- **Tutorial 3** derives closed-form Wirtinger gradients (and the stationarity /
  KKT condition) for a MIMO precoder.
- **Tutorial 4** introduces the named-node builder and cross-validates everything
  against the actual `cmi-dag` library.
