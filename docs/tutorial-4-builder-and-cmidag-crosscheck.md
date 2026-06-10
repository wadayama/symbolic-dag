# Tutorial 4 — The builder and cross-validation against cmi-dag

The final tutorial covers two practical things: the ergonomic named-node builder
(for readable models and output), and how `symbolic-dag` is cross-validated
against the actual numerical `cmi-dag` library.

By the end you will:

- Build a DAG with named nodes via `GaussianDAG` and see it produce the same
  result as the functional core.
- Numerically verify your own symbolic results in PyTorch — value (`.check`) and
  gradient (`.check_gradient`) — with no extra setup.
- Understand how the test suite drives the real `cmi-dag` as an oracle.
- Know how to run the cmi-dag cross-validation battery yourself.

---

## 1. The named-node builder

The functional API (`compute_k_blocks_multiroot` + indices) mirrors cmi-dag and is
the canonical surface. For readability — especially for the printed symbolic forms
— `GaussianDAG` lets you name nodes `X`, `Y`, `Z` instead of integers:

```python
import sympy as sp
from symbolic_dag import GaussianDAG, hermitian

d = sp.Symbol("d", positive=True, integer=True)
A, B = sp.MatrixSymbol("A", d, d), sp.MatrixSymbol("B", d, d)
SX, SY, SZ = (hermitian(s, d) for s in ("Sigma_X", "Sigma_Y", "Sigma_Z"))

G = GaussianDAG()
G.add_source("X", cov=SX)                       # sources first → root prefix
G.add_node("Y", parents={"X": A}, noise=SY)
G.add_node("Z", parents={"Y": B}, noise=SZ)

I = G.cmi(A=["X"], B=["Z"], C=["Y"])            # I(X; Z | Y)
print(I.is_conditionally_independent())          # True
```

The builder adds **no new capability** — it lowers the names to the prefix-root
integer indices (sources get `0..K-1`, non-sources `K..` in topological order) and
calls the same functional core. cmi-dag users can keep their index-based code; new
users get readable named output.

---

## 2. Verify your own result (PyTorch, no extra setup)

You can numerically verify any symbolic result yourself — this is the
PyTorch-main path and needs nothing beyond `uv sync` (torch is a core
dependency). Take a precoder model `Y = (H F) X0 + X1 + N` and study
`I(X0; Y | X1)`:

```python
import sympy as sp
from sympy import Identity, MatrixSymbol
from symbolic_dag import (
    compute_k_blocks_multiroot, conditional_mutual_information_from_k, hermitian,
)

d = sp.Symbol("d", positive=True, integer=True)
H, F = MatrixSymbol("H", d, d), MatrixSymbol("F", d, d)
S0, R = hermitian("Sigma_0", d), hermitian("R", d)

K = compute_k_blocks_multiroot(
    num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
    edge_mats={(2, 0): H * F, (2, 1): Identity(d)},
    root_covs={0: S0, 1: Identity(d)}, noise_covs={2: R},
)
I = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])

print(I.check(dim=3))              # the CMI value
#   {'passed': True, 'max_abs_err': ~1e-14, 'samples': 4}
print(I.check_gradient(F, dim=3))  # the Wirtinger gradient
#   {'passed': True, 'max_abs_err': ~1e-13}
```

`.check` recomputes the CMI the same way `cmi-dag` does — multi-root K-recursion +
Schur complement + `slogdet` — on random complex points and confirms your closed
form agrees. `.check_gradient` differentiates the *very same* torch CMI with
PyTorch autograd and checks `autograd == 2 · ∂I/∂F*` (the Wirtinger convention).
No cmi-dag checkout is needed for either. (A torch-free `numpy_cmi` oracle is also
available for a fully independent cross-check, but PyTorch is the main path.)

## 3. Cross-validation against the real cmi-dag

Every symbolic result is verified two ways:

1. **PyTorch self-check** (`.check`, `.check_gradient`) — the symbolic CMI is
   lowered to a differentiable torch scalar (`to_torch`) and checked against an
   independent Schur-complement evaluation and against torch **autograd** for the
   gradient. PyTorch is a core dependency, so this runs by default. (An
   independent torch-free NumPy oracle, `numpy_cmi`, gives an extra cross-check.)
2. **The actual `cmi-dag` library** — for the strongest check, the test suite
   imports `cmi_dag` and drives its numerical CMI and its PyTorch **autograd**,
   comparing on random complex points across dimensions.

The cmi-dag bridge lives in `tests/cmidag_oracle.py`. Because `cmi-dag` is a
sibling repository rather than a PyPI package, the bridge adds its repository root
to `sys.path` (override with the `SYMBOLIC_DAG_CMIDAG_PATH` environment variable).
When that repository is absent the cmi-dag cross-check tests skip cleanly; the
rest of the suite (including the PyTorch self-checks) still runs.

---

## 4. Run the battery

```bash
uv run pytest        # full suite: symbolic + PyTorch self-checks + cmi-dag cross-validation
```

The battery checks, for chain / fork / collider / MAC gadgets and the precoder, at
dimensions `d = 1, 2, 3`:

- **CMI value** — `SymbolicCMI.evaluate` vs cmi-dag's
  `conditional_mutual_information_from_k`, agreeing to ~1e-10. (Both use the
  complex, no-`½` convention, so the values match directly.)
- **Wirtinger gradient** — the symbolic `∂I/∂F*` vs cmi-dag's autograd `F.grad`,
  agreeing to ~1e-9 after the convention factor of 2.
- **d-separation** — the symbolic proof (`is_conditionally_independent()`) agrees
  with cmi-dag's numerical `I ≈ 0`.

This is the division of labour the two libraries are designed for: cmi-dag
*discovers* numerically; symbolic-dag *explains* in closed form; and each checks
the other.

---

## 5. Where to go next

- The `examples/` directory has the polished, runnable versions:
  `gadgets.py`, `mac_cmi.py`, and `precoder_gradient.py`.
- Later phases will add the **answer layer** — regime maps, closed-form threshold
  extraction (`C_i(θ) = C_j(θ)`), and LaTeX export — that turn these symbolic CMIs
  and gradients into design rules.
