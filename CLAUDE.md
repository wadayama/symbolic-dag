# Working in this repo with Claude Code — symbolic-dag × Wolfram × verification

`symbolic-dag` derives **closed-form** conditional mutual information, Wirtinger
gradients, MMSE/KKT and related quantities for **linear Gaussian DAGs**, keeping
gains and covariances as *opaque symbols* (so the result is dimension-independent,
not a number). It is the symbolic sibling of the numerical
[`cmi-dag`](https://github.com/wadayama/cmi-dag).

The single most effective way to use this library is **not in isolation** but as
the exact-symbolic core of a three-way collaboration:

```
   Claude Code  (orchestrate · translate · VERIFY · interpret)
        │
        ├─ symbolic-dag → exact closed form from the model
        │     I(A;B|C), ∂I/∂F*, MMSE Σ_{X|Y}, Wiener filter, linear KKT
        │     (the dimension-independent block algebra a general CAS is bad at)
        │
        └─ Wolfram Engine → the heavy symbolic work OUT of scope here
              ∫ / Expectation over fading, special-function simplification
              (Meijer-G → e^{1/ρ}E₁, ExpIntegralEi, …), spectral limits
```

Each tool does what it is best at; **every stage is machine-verified** (PyTorch
autograd / cmi-dag / Monte-Carlo), so results are *checked*, never just plausible.

## Scope — what to mechanize, what stays human

Mechanize (the library does it, exactly):
- closed-form CMI for any disjoint `A, B, C` (single- or multi-node),
- closed-form Wirtinger gradient `∂I/∂F*` and the KKT condition `=0`,
- MMSE error covariance `Σ_{X|Y}`, the Wiener filter, **linear** stationarity solving,
- d-separation / conditional independence **proofs** (symbolic, not numeric).

Leave to the analyst (and reach for Wolfram / the human, not a new library layer):
- **capacity KKT** (`F` inside an inverse → nonlinear; needs an eigen-ansatz → water-filling),
- **ergodic averaging** `E_H[·]` and the fading-density / eigenvalue (Wishart→Laguerre) reduction,
- **asymptotic limits** (Szegő/Toeplitz `n→∞`, large-system RMT / deterministic equivalents),
- **regime / threshold / optimal-structure** extraction.

A recurring pattern across real papers: *symbolic-dag supplies the exact
finite/instantaneous closed form (the integrand or pre-limit form); the paper's
headline is a limit or average of it.* That boundary is deliberate and consistent.

## The hand-off API (taking the closed form to the next step)

For a `SymbolicCMI` `I` and a variable `F`:
- `I.to_latex()` / `I.report(F)` — LaTeX strings (structural or expanded).
- `I.to_pdf("out.pdf", var=F, png=True)` — standalone PDF/PNG via `pdflatex`, for an
  immediate visual check.
- `I.to_mathematica(F)` — Wolfram Language string of the gradient (`Dot` /
  `ConjugateTranspose` / `Inverse` / `Det` / `ArrayFlatten`; names → `Subscript[…]`).
  `I.to_mathematica()` gives the CMI scalar.
- `I.to_markdown(F)` — LLM- and human-readable Markdown with LaTeX math.

Verify numerically (PyTorch is the main path; torch is a core dependency):
- `I.check(dim)`, `I.check_gradient(F, dim)` (autograd == 2 × the Wirtinger gradient),
- `numpy_cmi(...)` / `I.numeric_check(subs, ref)` — torch-free independent oracle.

## Driving Wolfram from Claude Code

A licensed Wolfram/Mathematica install ships `wolframscript`. It is often **not on
`PATH`**; on macOS look inside the app bundle, e.g.
`/Applications/Wolfram.app/Contents/MacOS/wolframscript` (the older path is
`/Applications/Mathematica.app/Contents/MacOS/wolframscript`). Detect with
`command -v wolframscript` first, then fall back to the bundle.

Run code with `wolframscript -code '...'` (each invocation starts a kernel; allow a
generous timeout). Tips that made simplification work in practice:
- add assumptions: `FullSimplify[expr, ρ > 0]`, `Assumptions -> ρ > 0` on integrals;
- for an **ergodic** step, reduce to a scalar/eigenvalue integrand first (human), then
  `Expectation[Log[1 + ρ g], g \[Distributed] ExponentialDistribution[1]]` (Rayleigh)
  or `Integrate[Log[1 + ρ λ] p[λ], {λ, 0, ∞}]` with the Wishart/Laguerre density `p`;
- single-letter Wolfram built-ins (`N D E I K O`) may clash with symbol names — rename.

**Always close the loop:** pull the Wolfram result back and cross-check numerically
against a Monte-Carlo / `numpy_cmi` value before trusting it.

## Worked recipes (validated in practice)

1. **Reproduce a paper's closed form from the model.** Map the system to a DAG
   (`compute_k_blocks_multiroot`), compute the CMI / gradient / MMSE, and check it
   numerically. E.g. a 2-user MAC → individual rates `I(x_k;y|x_other)` and sum rate
   `I(x_1,x_2;y)`; a MIMO wiretap → secrecy rate `I(x;y_b) − I(x;y_e)`; a precoder/
   combiner → MMSE Wiener filter via `lmmse_estimator` / `trace_grad` + `solve_stationary`.
2. **Ergodic closed form.** `symbolic-dag` instantaneous rate → reduce to scalar/
   eigenvalue integrand → `wolframscript` `Expectation`/`Integrate` → clean special-
   function form → verify vs Monte-Carlo. (SISO Rayleigh → `e^{1/ρ}E₁(1/ρ)`.)
3. **Optimal linear design.** Build the MSE, `trace_grad` for the first-order
   condition, `solve_stationary` for the closed-form optimum (Wiener / RZF style).

## House rules

- **No fabricated data.** Every formula/number reported must come from actual code or
  `wolframscript` execution and be cross-checked; never hand-wave a result. (This
  mirrors the parent project's data-integrity rule.)
- **PyTorch is the main numerical surface** (`.check` / `.check_gradient` /
  `torch_value`); `numpy_cmi` is an internal torch-free oracle only.
- Run things with `uv run …`; `uv sync` reproduces the environment.
```
