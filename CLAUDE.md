# Guide for Claude Code — symbolic-dag × Wolfram × verification

This file orients an AI agent (Claude Code) working in this repository. `symbolic-dag`
derives **closed-form** conditional mutual information, Wirtinger gradients, MMSE/KKT
and related quantities for **linear Gaussian DAGs**, keeping gains and covariances as
*opaque symbols* (so results are dimension-independent, not numbers). It is the
symbolic sibling of the numerical [`cmi-dag`](https://github.com/wadayama/cmi-dag).

The library is most powerful **not in isolation** but as the exact-symbolic core of a
small, fully-verified pipeline:

```
   Claude Code  (orchestrate · translate · VERIFY · interpret)
        │
        ├─ symbolic-dag → exact closed form from the model
        │     I(A;B|C), ∂I/∂F*, MMSE Σ_{X|Y}, Wiener filter, linear KKT
        │     (the dimension-independent block algebra a general CAS is bad at)
        │
        └─ Wolfram Engine (optional) → the heavy symbolic work OUT of scope here
              ∫ / Expectation over fading, special-function simplification
              (Meijer-G → e^{1/ρ}E₁, ExpIntegralEi, …), spectral limits
```

Each tool does what it is best at; **every stage should be machine-verified** (PyTorch
autograd / cmi-dag / Monte-Carlo), so results are *checked*, never just plausible.

## Scope — what to mechanize, what stays human

Mechanize (the library does it, exactly):
- closed-form CMI for any disjoint `A, B, C` (single- or multi-node), plus the
  sibling quantities: conditional entropy, total correlation, directed information,
  Gaussian KL, and weighted-CMI-sum objectives (`composite_cmi`),
- closed-form Wirtinger gradient `∂I/∂F*` and the KKT condition `=0`; the gradient
  w.r.t. a **Hermitian input covariance** `Q` too (the capacity gradient `∂/∂Q`),
- MMSE error covariance `Σ_{X|Y}`, the Wiener filter, **linear** stationarity solving,
- d-separation / conditional independence **proofs** (symbolic, not numeric).

Leave to the analyst (reach for Wolfram / a human, *not* a new library layer):
- **capacity KKT** (`F` inside an inverse → nonlinear; needs an eigen-ansatz → water-filling),
- **ergodic averaging** `E_H[·]` and the fading-density / eigenvalue (Wishart→Laguerre) reduction,
- **asymptotic limits** (Szegő/Toeplitz `n→∞`, large-system RMT / deterministic equivalents),
- **regime / threshold / optimal-structure** extraction.

Recurring pattern in real papers: *symbolic-dag supplies the exact finite/instantaneous
closed form (the integrand or pre-limit form); the paper's headline is a limit or
average of it.* That boundary is deliberate — don't try to grow the library past it.

## A minimal end-to-end example

```python
import sympy as sp
from sympy import MatrixSymbol, Identity
from symbolic_dag import (compute_k_blocks_multiroot,
                          conditional_mutual_information_from_k, hermitian)

d = sp.Symbol("d", positive=True, integer=True)      # symbolic dimension (or a concrete int)
H, F = MatrixSymbol("H", d, d), MatrixSymbol("F", d, d)   # plain gains
S0, R = hermitian("Sigma_0", d), hermitian("R", d)       # covariances: Hermitian PD

# Model: Y = (H F) X0 + X1 + N, study I(X0; Y | X1).
K = compute_k_blocks_multiroot(
    num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
    edge_mats={(2, 0): H * F, (2, 1): Identity(d)},      # key = (child, parent) -> gain
    root_covs={0: S0, 1: Identity(d)}, noise_covs={2: R})
I = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])

G = I.wirtinger_grad(F)        # closed-form ∂I/∂F*  (autograd returns 2·G)
I.check(dim=3)                 # value vs an independent torch computation -> {'passed': True, ...}
I.check_gradient(F, dim=3)     # gradient vs PyTorch autograd
print(I.to_markdown(F))        # or .to_latex() / .to_pdf("out.pdf", var=F) / .to_mathematica(F)
```

Run anything with `uv run python …` / `uv run pytest`; `uv sync` reproduces the env.

## Modeling the DAG — conventions and gotchas

- **Node indices.** Sources (roots) come **first** and must be a prefix `{0,…,K-1}` in
  topological order; the remaining nodes follow. `A`, `B`, `C` are lists of node indices.
- **The six arguments.** `num_nodes`; `roots=[…]`; `parents={child:[parents]}`;
  `edge_mats={(child,parent): gain}`; `root_covs={root: cov}`; `noise_covs={non-root: cov}`.
- **Symbols.** Covariances must be `hermitian(name, d)` (the engines know they are
  Hermitian PD and apply `Adjoint(Σ)→Σ`); gains/precoders are plain `MatrixSymbol`.
  Use `Identity(d)` / `ZeroMatrix` where needed. `d` may be symbolic or an `int`.
- **Interference: condition, or not?** This silently changes the formula — choose by how
  the receiver actually decodes:
  - *jointly decoded / SIC* (e.g. a multiple-access channel, or a stream decoded after
    cancelling another) → **condition** on those nodes: `I(x_k; y | x_others)`;
  - *interference treated as noise* (broadcast/interference per-user rate, a wiretap's
    eavesdropper) → **do not** condition; the interferers stay part of the (colored)
    channel: `I(x_k; y)`.
- **MMSE.** `mmse_error_covariance(K, target, observations)` is `Σ_{target|observations}`;
  it equals a paper's MSE matrix `E_k` when the input covariance is the identity.

## The hand-off API (taking the closed form to the next step)

For a `SymbolicCMI` `I`, a variable `F`, any `MatrixExpr` `M`:
- `I.to_latex()` / `I.report(F)` — LaTeX strings (structural or expanded).
- `I.to_pdf("out.pdf", var=F, png=True)` — standalone PDF/PNG via `pdflatex`, for a quick
  visual check (no-op clear error if `pdflatex`/`pdftocairo` are absent).
- `I.to_mathematica(F)` — Wolfram Language for the gradient (`Dot` / `ConjugateTranspose`
  / `Inverse` / `Det`; names → `Subscript[…]`). `I.to_mathematica()` gives the CMI;
  `I.to_mathematica(scalar=True)` flattens 1×1 matrices into a scalar form ready for
  Wolfram `Integrate` / `Expectation`.
- `from_mathematica(s)` — parse a Wolfram result back into `sympy` (special functions
  mapped), to evaluate / cross-check it.
- `I.to_markdown(F)` — LLM- and human-readable Markdown with LaTeX math.

## Verifying results (do this for everything)

- **A CMI / its gradient:** `I.check(dim)` and `I.check_gradient(F, dim)` (the latter
  checks PyTorch `autograd == 2 ×` the Wirtinger gradient). A torch-free oracle exists:
  `numpy_cmi(...)` / `I.numeric_check(subs, ref)`.
- **Any other expression** (a gradient `G`, an MMSE covariance, a `solve_stationary`
  result): lower both sides to torch with `to_torch(expr, subs, dim)` and compare to an
  independent computation; or autograd a torch objective and compare to `2·to_torch(G,…)`.
  Build `subs` with `random_torch_point(I, dim)` or your own complex128 tensors
  (**Hermitian PD** for covariances). PyTorch is the main numerical surface; `numpy_cmi`
  is an internal torch-free oracle only.

## Driving Wolfram (optional, when integrals / special functions are needed)

A licensed Wolfram/Mathematica install ships `wolframscript`, often **not on `PATH`**.
Detect it, then run code (each call starts a kernel — allow a generous timeout):

```bash
WS=$(command -v wolframscript \
     || ls /Applications/Wolfram.app/Contents/MacOS/wolframscript \
           /Applications/Mathematica.app/Contents/MacOS/wolframscript 2>/dev/null | head -1)
"$WS" -code 'Print[FullSimplify[
   Expectation[Log[1 + rho g], g \[Distributed] ExponentialDistribution[1]], rho > 0]]'
#  -> E^(1/rho) Gamma[0, 1/rho]
```

Tips that make this work in practice:
- wrap the result in `Print[…]`; single-quote the `-code` string; add assumptions
  (`FullSimplify[expr, rho>0]`, `Assumptions -> rho>0` on integrals);
- for an **ergodic** step, get a scalar integrand with `I.to_mathematica(scalar=True)`
  (matrix forms don't integrate); reduce MIMO to its eigenvalue density (Wishart→Laguerre)
  by hand, then `Expectation[…]` (Rayleigh) or `Integrate[Log[1+rho λ] p[λ], {λ,0,∞}]`;
- single-letter Wolfram built-ins (`N D E I K O`) may clash with symbol names — rename;
- **close the loop:** bring the result back with `from_mathematica(s)` and cross-check it
  numerically against a Monte-Carlo / `numpy_cmi` value before trusting it. Full loop:
  `to_mathematica(scalar=True)` → `wolframscript` `Expectation` → `from_mathematica` →
  `.evalf()` == Monte-Carlo (SISO Rayleigh → `e^{1/ρ}E₁(1/ρ)`).

## Worked recipes

1. **Reproduce a paper's closed form from the model.** Map the system to a DAG, compute
   the CMI / gradient / MMSE, verify numerically. E.g. a 2-user MAC → individual rates
   `I(x_k;y|x_other)` and sum rate `I(x_1,x_2;y)`; a MIMO wiretap → secrecy rate
   `I(x;y_b) − I(x;y_e)`; a precoder/combiner → MMSE/Wiener filter via `lmmse_estimator`
   or `trace_grad` + `solve_stationary`.
2. **Ergodic closed form.** instantaneous rate (`scalar=True`) → reduce to a
   scalar/eigenvalue integrand → Wolfram `Expectation`/`Integrate` → clean special-function
   form → `from_mathematica` → verify vs Monte-Carlo.
3. **Optimal linear design.** build the MSE objective, `trace_grad` for the first-order
   condition, `solve_stationary` for the closed-form optimum (Wiener / RZF style).

## House rules

- **No fabricated data.** Every formula or number you report must come from actual code or
  `wolframscript` execution and be cross-checked — never hand-wave or guess a result.
- **PyTorch is the main numerical surface** (`.check` / `.check_gradient` / `torch_value`);
  `numpy_cmi` is an internal torch-free oracle only.
- Prefer the existing engines over re-deriving by hand; if you must hand-simplify a form,
  verify it equals the library's output numerically before presenting it.
