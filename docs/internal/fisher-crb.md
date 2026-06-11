# (internal) Fisher information / Cramér–Rao bound — ISAC layer

> **Status: experimental, NOT part of the public API.** Implemented in
> `symbolic_dag/_fisher.py`, intentionally **not exported** from `symbolic_dag`
> and **not listed** in the README / public API table, so the public concept
> stays *"symbolic conditional mutual information for linear Gaussian DAGs."*
> It is committed (students / collaborators / people who find it may use it) but
> unadvertised. Promote it to the public API when it is mature.

## Why this exists

The Slepian–Bangs **Fisher information** / **Cramér–Rao bound** is the *sensing*
dual of the library's *communication* quantities: both the CMI/MMSE and the
FIM/CRB are **log-det / trace / inverse of a linear-Gaussian covariance**. So a
small layer lets the same engine serve **ISAC** (integrated sensing and
communication) waveform design — reproducing, *from the DAG model*, the sensing
closed forms (FIM, CRB/PCRB, and the design gradient `∂CRB/∂F*`) that ISAC papers
derive by hand. The hard parts are already in the engine: covariances come from
the K-recursion, the design gradient reuses `trace_grad`, numeric checks reuse
`to_torch`.

## API (`from symbolic_dag._fisher import ...`)

| Function | Purpose |
|---|---|
| `fisher_information_matrix(R, dR=(), dmu=(), *, prior=None)` | Slepian–Bangs FIM `J` (`P×P` sympy Matrix): `J[i,j] = tr(R⁻¹ dR_i R⁻¹ dR_j) + (dμ_i^H R⁻¹ dμ_j + dμ_j^H R⁻¹ dμ_i)` `[+ prior]`. `R` is the observation covariance (e.g. from the K-recursion); `dR`, `dμ` are the parameter derivatives (covariance- and mean-dependent terms); `prior` is an optional additive (Bayesian/posterior) FIM. |
| `cramer_rao_bound(J, indices=None)` | `J⁻¹` (full), or `(J⁻¹)[i,i]` for `indices=i` — the CRB on parameter `i` with the rest as nuisances (the Schur-complement / posterior-CRB form). |
| `crb_trace(J, indices=None)` | `tr(J⁻¹)` over all parameters (or a subset) — the common ISAC scalar metric. |
| `crb_grad(metric, var)` | Wirtinger **design gradient** `∂(metric)/∂var*` of a scalar CRB metric, reusing `trace_grad` + scalar chain rule. Autograd returns `2 ×` this. |
| `crb_value(metric, subs, dim)` | numeric value of a (real) CRB metric (torch). |
| `crb_grad_check(metric, var, dim, …)` | check `crb_grad` vs PyTorch autograd (`autograd == 2 × crb_grad`). |

**Sensing-model input (the human's part).** `dR`, `dμ`, and the steering matrices
`A_k` come from the array geometry and the prior — a sensing-specific input, just
as the channel matrices are inputs on the comms side. The parametric derivative
`∂R/∂θ` (w.r.t. real angle/delay/Doppler) is supplied; the *design* gradient
`∂CRB/∂F*` (w.r.t. the complex precoder) is what the library computes.

## Worked example — reproducing an ISAC paper's PCRB (cf. 2026002049)

A multi-antenna BS estimates a point target's angle θ; the posterior CRB is a
function of the **transmit covariance** `R_X`:

```python
import sympy as sp
from sympy import MatrixSymbol, Trace, Adjoint
from symbolic_dag import hermitian, compute_k_blocks_multiroot, get_K
from symbolic_dag.assumptions import apply_hermitian
from symbolic_dag._fisher import cramer_rao_bound, crb_trace, crb_grad, crb_grad_check

d = sp.Symbol("d", positive=True, integer=True)
F, A2 = MatrixSymbol("F", d, d), MatrixSymbol("A_2", d, d)
Sig, A1, A3 = hermitian("Sigma", d), hermitian("A_1", d), hermitian("A_3", d)

# transmit covariance R_X = F Σ Fᴴ, built by the library's K-recursion
K = compute_k_blocks_multiroot(2, [0], {1: [0]}, {(1, 0): F}, {0: Sig}, {1: hermitian("Z", d)})
RX = F * Sig * F.adjoint()

# the 2×2 FIM for (θ, α) in transmit-covariance / steering form
J = sp.Matrix([[Trace(A1 * RX), Trace(A2 * RX)],
               [Trace(Adjoint(A2) * RX), Trace(A3 * RX)]])

pcrb = cramer_rao_bound(J, 0)     # = tr(A3 RX) / (tr(A1 RX) tr(A3 RX) − |tr(A2 RX)|²)
                                  #   == the paper's PCRB_θ = 1/(tr A1 RX − |tr A2 RX|²/tr A3 RX)
g = crb_grad(pcrb, F)             # ∂PCRB/∂F*  — the ISAC sensing-design gradient
crb_grad_check(pcrb, F, dim=3)    # {'passed': True, 'max_abs_err': ~1e-18}  (vs autograd)
```

Because the communication rate `I(x;y)` is already the public library's job, both
ISAC objectives — sensing PCRB and communication rate, and **both** design
gradients — come from one engine on one linear-Gaussian DAG (Shannon info +
Fisher info unified).

## Verification & tests

`tests/test_fisher.py` (7 tests): the PCRB reproduces the paper's closed form;
`crb_grad` matches autograd to ~1e-18 for the PCRB and `tr(J⁻¹)` metrics (d=2,3);
`fisher_information_matrix` values match a direct FIM (covariance- and mean-term).

**ICC reproduction test** (`test_single_target_angle_crb_reproduction`): the
single-target angle CRB with a complex-amplitude nuisance — the data-FIM CRB at the
heart of MIMO-radar ISAC papers (e.g. 2026002049). Echo mean `μ = s·b(θ)`; build the
3-parameter FIM (`θ, Re s, Im s`) with `fisher_information_matrix(N, dmu=[s·ḃ, b, j·b])`,
take `cramer_rao_bound(J, 0)`, and check it equals both a direct numerical FIM inverse
and the projection closed form `σ²/(2‖P_b^⊥(s·ḃ)‖²)` — agreement ~1e-18.

## Known gaps / future work

- `crb_grad` is verified for `cramer_rao_bound(J, i)` and `crb_trace(J)`; other
  scalar metrics built from `Trace` atoms should work but are not all tested.
- No convenience to build `A_1, A_2, A_3` from a steering vector `a(θ)` and its
  derivative `ȧ(θ)` (a `point_target_fim` helper) — deferred; supply them directly.
- `∂R/∂θ` parametric derivatives are user-supplied (not auto-derived from geometry).
- When promoting to the public API: export from `symbolic_dag`, add to the README
  API table + `CLAUDE.md`, and decide the public names (`fisher_*` / `crb_*`).
