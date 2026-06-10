# Tutorial 3 — Closed-form Wirtinger gradients and KKT

This tutorial derives, symbolically, the gradient of a conditional mutual
information with respect to a precoder matrix — the kind of derivation usually
done by hand in MIMO information theory — and reads off the stationarity (KKT)
condition that characterises the optimal precoder.

By the end you will:

- Understand the MIMO precoder gadget and the quantity `I(X0; Y | X1)`.
- Have derived the closed-form Wirtinger gradient `∂I/∂F*` with
  `.wirtinger_grad(F)`.
- Have obtained the optimal-precoder condition with `.stationarity(F)`.
- Understand the `×2` autograd convention used to cross-check it.

---

## 1. The model

A MIMO precoder feeding one receiver, with a second (interfering / conditioning)
input:

```
   X0 ──► [ H F ] ──┐
                    ├──► Y = H F X0 + X1 + N,   N ~ CN(0, R)
   X1 ──► [  I  ] ──┘
```

`F` is the precoder we differentiate with respect to; `H` is the channel, `Σ0` the
input covariance of `X0`, and `R` the noise covariance. We study

```
I(X0; Y | X1) = log det(R + H F Σ0 F^H H^H) − log det R,
```

the information the receiver gains about `X0` given the interferer `X1`.

---

## 2. Build it and differentiate

```python
import sympy as sp
from symbolic_dag import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
    hermitian,
)

d = sp.Symbol("d", positive=True, integer=True)
H, F = sp.MatrixSymbol("H", d, d), sp.MatrixSymbol("F", d, d)
S0, R = hermitian("Sigma_0", d), hermitian("R", d)

K = compute_k_blocks_multiroot(
    num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
    edge_mats={(2, 0): H * F, (2, 1): sp.Identity(d)},
    root_covs={0: S0, 1: sp.Identity(d)}, noise_covs={2: R},
)
I = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])

G = I.wirtinger_grad(F)
print(G)
#   Adjoint(H)*(R + H*F*Sigma_0*Adjoint(F)*Adjoint(H))**(-1)*H*F*Sigma_0
```

The gradient is **derived**, not hand-coded. `sympy`'s native matrix
differentiation returns the zero matrix here; `symbolic-dag`'s engine instead uses
the matrix-calculus identity `d log det M = tr(M^{-1} dM)` and extracts the
coefficient of `dF^H` by the cyclic property of the trace. The result,

```
∂I/∂F* = H^H (R + H F Σ0 F^H H^H)^{-1} H F Σ0,
```

is a single dimension-independent matrix expression.

---

## 3. The stationarity (KKT) condition

```python
print(I.stationarity(F))
#   Eq(Adjoint(H)*(R + H*F*Sigma_0*Adjoint(F)*Adjoint(H))**(-1)*H*F*Sigma_0, 0)
```

Setting `∂I/∂F* = 0` is the first-order condition for the optimal precoder. Where
a numerical optimiser (cmi-dag's `pga_ascent`) only *reaches* a stationary point,
the symbolic condition *characterises* it — and exposes its dependence on every
parameter (`H`, `Σ0`, `R`) in closed form.

---

## 4. Check it numerically (PyTorch, one call)

Verify the gradient against PyTorch autograd in a single call (PyTorch is a core
dependency, so this works after a plain `uv sync`). The library handles the **convention
factor of 2**: for a real loss and a complex leaf `F`, PyTorch's `.grad` equals
`2 · ∂I/∂F*`, so internally it checks `autograd == 2 · G`.

```python
print(I.check_gradient(F, dim=3))
#   {'passed': True, 'max_abs_err': 2.5e-14}
print(I.check(dim=3))                 # the CMI value, too
#   {'passed': True, 'max_abs_err': 1.8e-14, 'samples': 4}
```

Under the hood, `to_torch` lowers the symbolic CMI to a **differentiable** torch
scalar (`I.torch_value(subs, dim)`), so you can also drop the symbolic CMI
straight into your own PyTorch experiments and autograd. See
`examples/precoder_gradient.py` for the runnable version.

This same factor-of-2 is the `½`-vs-no-`½` bookkeeping that distinguishes real and
complex conventions; `symbolic-dag` and cmi-dag both use the complex (no-`½`)
convention for the CMI itself, so their **values** match directly and only the
gradient carries the `×2`.

---

## 5. Arbitrary `A`, `B`, `C`, and the hand-off

The gradient is not limited to the precoder: it handles **arbitrary conditioning
`C` and multi-node information sets, as long as `A` or `B` is a single node**
(internally it conditions on `C` one node at a time, so the conditional
covariances stay single matrices). For example `I(X0, X1; Y | Z)` differentiates
just as readily. Both-multi-node sets and trace-form (MMSE) objectives are
deferred.

Once you have the closed forms, hand them to your own analysis with `to_latex` /
`report`:

```python
print(I.report(F))   # LaTeX: the CMI, the gradient dI/dF*, and the KKT condition
```

The library's job ends at producing these closed forms; the problem-specific
regime / optimal-structure analysis (e.g. SVD water-filling) is yours. Tutorial 4
shows the named-node builder and the full cross-validation against `cmi-dag`.
