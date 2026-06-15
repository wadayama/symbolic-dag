# dag-demo — GUI demo for symbolic-dag

Draw a linear Gaussian DAG like in a paint program, assign the roles A / B / C to
nodes, press a button, and the symbolic closed form of the conditional mutual
information `I(V_A; V_B | V_C)` is displayed. The backend is the
[`symbolic-dag`](../README.md) library in the parent directory (an editable path
dependency, `path = ".."`).

## Run

```bash
cd demo
uv sync                                   # first time only
uv run python desktop.py
```

A native window (pywebview) opens. **No HTTP server is started**: the drawing /
rendering layer (Cytoscape.js + KaTeX) runs in an embedded WebView, and the
symbolic engine runs in-process, reached directly through a **JS↔Python bridge**
(`window.pywebview.api.compute`) — no localhost port either. All front-end assets
(Cytoscape / KaTeX / fonts) are vendored under `static/vendor/`, so the app runs
**fully offline, no network required**. For distribution it can be bundled into a
`.app` / `.exe` with `pyinstaller` etc. (note the binary gets large because
PyTorch is bundled).

## Usage

| Action | How |
| --- | --- |
| Add node | Double-click the canvas (or "+ Add node") |
| Add edge | Turn on "Connect mode" → click parent → click child |
| Assign role | Select a node → A / B / C buttons (or keys `a` `b` `c`; clear with `n`) |
| Compute | "Compute I(A;B\|C)" |
| Wirtinger gradient (channel) | **With an edge selected**, compute → additionally shows ∂I/∂H* w.r.t. that gain H |
| Precoder | Select a node → "Precoder F" (or key `p`) toggles it. **All outgoing edges** of that node become `H·F` (shared as an attribute of the transmit node) |
| Wirtinger gradient (precoder) | **With a precoded node selected**, compute → shows ∂I/∂F* (no KKT display in this demo) |
| LMMSE estimation | Check "Show LMMSE (W / E)" → shows the Wiener filter `W` estimating A (a single node) from B and C, and the error covariance `E = Σ_{A\|B,C}`. E is exactly the second term of the two-term CMI form (the bridge between information and estimation error) |
| Export code | "Export code" emits runnable `symbolic-dag` source for the drawn DAG, with a **High-level (builder)** / **Low-level (functional)** toggle and a Copy button. Use the GUI as a DAG builder, then paste the code into a script/notebook |

- An in-degree-0 node is automatically a source (covariance `Σ_name`, drawn as a
  rounded rectangle). Every other node is `V = Σ_p H_{p→child} V_p + N_child`
  (noise `N_name`).
- Everything is a complex, arbitrary-dimension symbolic matrix (Wirtinger
  convention, no ½, nats).
- "PyTorch numerical check" matches the symbolic form against an independent
  computation at random complex points (`SymbolicCMI.check`).
- The expanded form is shown as a **two-term entropy difference**
  `I = log det Σ_{B|C} − log det Σ_{B|AC}` (folding in the Schur identity
  `|Σ_{AB|C}| = |Σ_{A|C}|·|Σ_{B|AC}|` to avoid the joint block matrix). Each term
  is cleaned up on the demo side (`block_collapse` + distribute/expand + collect
  like terms + normalize). When the library's `"capacity"` strategy (determinant
  lemma + Sylvester) applies, the capacity form `log det(I + ·)` is also shown.
  For independent cases the two terms coincide and `= 0` is made explicit.
- At every display/verification step, the CMI value before and after each
  transformation is matched at a random complex point per request; on mismatch it
  falls back to the library's standard expanded form (a wrong formula is never
  shown).
- Determinants are rendered as `det(X)`, not `|X|`.
- An A→B→C chain is preloaded (the d-separation proof of I(X;Z|Y)=0 is one click
  away).
- **Symbolic-blow-up guard**: the computation runs in a separate process, and if
  it exceeds the timeout (default 30 s, configurable via the `DAG_DEMO_TIMEOUT`
  environment variable) the worker is killed and an error is shown. The UI never
  freezes, even on an over-large graph.
- **Code export** (GUI-as-builder): "Export code" turns the drawn DAG into
  runnable `symbolic-dag` source — the same named→index lowering `core._build`
  does, emitted as text (no symbolic computation, so it is instant). Two levels:
  *high-level* uses the named-node builder (`GaussianDAG` / `add_source` /
  `add_node` / `cmi`); *low-level* uses the functional core
  (`compute_k_blocks_multiroot` + `conditional_mutual_information_from_k`, with
  sources placed at the index prefix). Roles A/B/C become the `cmi(...)` query;
  precoders become `H·F` on the outgoing edges; multiple sources become
  `roots=[…]`. Both forms are verified to run.

## Layout

```
demo/
├── pyproject.toml      uv project (pywebview + pydantic + symbolic-dag path dep)
├── core.py             the symbolic engine: graph JSON → GaussianDAG → CMI/gradient LaTeX
│                       (GUI-independent; the process-isolated timeout lives here too)
├── desktop.py          standalone GUI (pywebview; wires core to the JS↔Python bridge)
└── static/
    ├── index.html      one-page UI (reads the vendor/ assets via relative paths)
    ├── app.js          Cytoscape.js editing logic + KaTeX rendering
    │                   (calls core through the pywebview bridge)
    ├── style.css
    └── vendor/         vendored front-end assets (for offline use)
        ├── cytoscape.min.js
        └── katex/      katex.min.js / katex.min.css / fonts/
```

`core.py` is the single source of truth for the symbolic computation, and
`desktop.py` is its thin I/O adapter.

## Third-party assets

The front-end assets under `static/vendor/` are vendored for fully-offline use.
Both are MIT-licensed, as is this project:

- [KaTeX](https://katex.org/) — © Khan Academy and contributors (MIT). License
  text in `static/vendor/katex/README.md`.
- [Cytoscape.js](https://js.cytoscape.org/) — © The Cytoscape Consortium (MIT).
  License header in `static/vendor/cytoscape.min.js`.

## Bridge API

`window.pywebview.api.compute(body)` (calls `desktop.py`'s `Api.compute`
directly). Shape of `body`:

```json
{
  "nodes": [{"id": "X", "role": "A", "precoder": true}, {"id": "Y", "role": "C"}, {"id": "Z", "role": "B"}],
  "edges": [{"source": "X", "target": "Y"}, {"source": "Y", "target": "Z"}],
  "expand": false,
  "check": true,
  "grad_edge": {"source": "X", "target": "Y"},
  "grad_node": null
}
```

Response: `model` (LaTeX array of the model equations), `latex` (structural CMI
form), `independent` (symbolic d-separation proof from the rewrite engine),
`latex_expanded` / `latex_capacity` (optional), `gradient.latex` (optional;
`grad_edge` gives ∂I/∂H*, `grad_node` gives the precoder ∂I/∂F*), `lmmse.W` /
`lmmse.E` (when `lmmse: true`; `lmmse.note` if A has more than one node),
`check.passed` / `check.max_abs_err`. Include `lmmse: true` in the request to
enable it.
