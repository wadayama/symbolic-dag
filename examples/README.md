# `symbolic-dag/examples/`

Three self-contained scripts that double as runnable pedagogy for the library's
public API. Each is independent — no shared utilities, no configuration files.

## Quick start

```bash
uv run python examples/gadgets.py
uv run python examples/mac_cmi.py
uv run python examples/precoder_gradient.py
```

All run after a plain `uv sync` (PyTorch is a core dependency). The numeric checks
are PyTorch, aligned with the numerical sibling `cmi-dag`.

## Scripts

| Script | Demonstrates |
|---|---|
| `gadgets.py` | chain / fork / collider: conditional independence **proved** symbolically (the cross conditional covariance reduces to zero), and the lazy CMI log-det terms printed. |
| `mac_cmi.py` | the 2-user MAC sum-rate facet `I(X0, X1; Y)` (a multi-node information set); the chain rule `I(X0,X1;Y) = I(X0;Y|X1) + I(X1;Y)`; one-call PyTorch value check. |
| `precoder_gradient.py` | the **closed-form Wirtinger gradient** `∂I/∂F*` and the stationarity / KKT condition for a MIMO precoder, derived symbolically and (with the `cmidag` extra) cross-checked against cmi-dag's PyTorch autograd to ~1e-14. |

## Verification convention

Every example prints a numerical check: the symbolic result is substituted at a
concrete random complex point and compared to an **independent** computation — the
PyTorch check (`.check`, `to_torch`) and/or the actual `cmi-dag` library. Agreement is to
~1e-9 or better. This is the project's guard against fabricated results: no number
printed here is taken on faith from the symbolic path alone.

## Reproducibility

Each script seeds its random draw (`numpy.random.default_rng(0)`), so re-running
produces identical numbers. Channel and gain entries are standard complex
Gaussians; covariances are random Hermitian positive-definite.

## Companion library

These examples are the symbolic counterpart to
[`cmi-dag/examples/`](https://github.com/wadayama/cmi-dag/tree/main/examples).
There the precoder optimisation is run *numerically* with projected gradient
ascent; here the same gradient is *derived in closed form* and checked against it.
