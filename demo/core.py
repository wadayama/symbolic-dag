"""The CMI engine behind the desktop app (``desktop.py``).

The UI sends the drawn DAG (nodes with optional A/B/C roles, edges, optionally a
selected edge/node for the gradient). This module lowers it to a
``symbolic_dag.GaussianDAG``, computes I(V_A; V_B | V_C), and returns:

- the structural and (optionally) expanded LaTeX of the CMI,
- the model equations (one line per non-source node),
- the symbolic conditional-independence verdict (rewrite-engine proof),
- a PyTorch numerical check of the value at random complex points,
- for a selected edge/precoder, the closed-form Wirtinger gradient.

It has no GUI dependency: errors are raised as :class:`ComputeError`
(``status``/``detail``), which the desktop bridge turns into an error payload.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
from collections import Counter
from typing import Any

import sympy as sp
from pydantic import BaseModel
from sympy import MatrixSymbol

from symbolic_dag import (
    GaussianDAG,
    hermitian,
    lmmse_estimator,
    mmse_error_covariance,
    random_torch_point,
    simplify_expr,
    simplify_logdet_terms,
    to_mathematica,
    to_torch,
)
from symbolic_dag.expr import SymbolicCMI


class ComputeError(Exception):
    """A user-facing computation error with a status code and a message."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class NodeIn(BaseModel):
    id: str
    role: str | None = None  # "A" | "B" | "C" | None
    precoder: bool = False  # transmit precoder F on all outgoing edges


class EdgeIn(BaseModel):
    source: str  # parent
    target: str  # child


class GraphIn(BaseModel):
    nodes: list[NodeIn]
    edges: list[EdgeIn]
    expand: bool = False
    check: bool = True
    check_dim: int = 2
    grad_edge: EdgeIn | None = None  # edge whose gain H to differentiate by
    grad_node: str | None = None  # precoded node whose F to differentiate by
    lmmse: bool = False  # Wiener filter / error covariance for estimating A


def _toposort(nodes: list[str], edges: list[EdgeIn]) -> list[str]:
    """Kahn topological sort; raises on cycles."""
    indeg = {n: 0 for n in nodes}
    children: dict[str, list[str]] = {n: [] for n in nodes}
    for e in edges:
        indeg[e.target] += 1
        children[e.source].append(e.target)
    queue = [n for n in nodes if indeg[n] == 0]
    order: list[str] = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for c in children[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)
    if len(order) != len(nodes):
        raise ComputeError(400, "The graph has a cycle (it must be a DAG).")
    return order


def _build(graph: GraphIn):
    """Lower the request graph to a GaussianDAG + the per-edge gain symbols."""
    names = [n.id for n in graph.nodes]
    if len(set(names)) != len(names):
        raise ComputeError(400, "Duplicate node names.")
    for e in graph.edges:
        if e.source not in set(names) or e.target not in set(names):
            raise ComputeError(400, "An edge references an undefined node.")
        if e.source == e.target:
            raise ComputeError(400, "Self-loops are not allowed.")

    d = sp.Symbol("d", positive=True, integer=True)
    order = _toposort(names, graph.edges)
    parents_of: dict[str, list[str]] = {n: [] for n in names}
    for e in graph.edges:
        parents_of[e.target].append(e.source)
    has_precoder = {n.id: n.precoder for n in graph.nodes}

    gains: dict[tuple[str, str], MatrixSymbol] = {}  # (parent, child) -> H
    precoders: dict[str, MatrixSymbol] = {}  # transmit node -> F
    dag = GaussianDAG()
    model_lines: list[str] = []
    for n in order:
        ps = parents_of[n]
        if not ps:
            dag.add_source(n, hermitian(f"Sigma_{n}", d))
        else:
            pmap = {}
            terms = []
            for p in ps:
                H = MatrixSymbol(f"H_{p}{n}", d, d)
                gains[(p, n)] = H
                if has_precoder.get(p):
                    F = precoders.setdefault(p, MatrixSymbol(f"F_{p}", d, d))
                    pmap[p] = H * F
                    terms.append(rf"H_{{{p}{n}}} F_{{{p}}} V_{{{p}}}")
                else:
                    pmap[p] = H
                    terms.append(rf"H_{{{p}{n}}} V_{{{p}}}")
            dag.add_node(n, pmap, hermitian(f"N_{n}", d))
            model_lines.append(rf"V_{{{n}}} = {' + '.join(terms)} + N_{{{n}}}")
    source_lines = [
        rf"V_{{{n}}} \sim \mathcal{{CN}}(0,\, \Sigma_{{{n}}})"
        for n in order
        if not parents_of[n]
    ]
    return dag, gains, precoders, source_lines + model_lines


def _with_terms(cmi, terms) -> SymbolicCMI:
    return SymbolicCMI(
        definitions=cmi.definitions, output=cmi.output, metadata=cmi.metadata,
        A=cmi.A, B=cmi.B, C=cmi.C, logdet_terms=terms, cross=cmi.cross,
    )


def _values_match(cmi, candidate) -> bool:
    subs = random_torch_point(cmi, 2)
    diff = abs(cmi.torch_value(subs, 2) - candidate.torch_value(subs, 2))
    return float(diff) <= 1e-9


def _matrices_match(cmi, M1, M2) -> bool:
    subs = random_torch_point(cmi, 2)
    diff = (to_torch(M1, subs, 2) - to_torch(M2, subs, 2)).abs().max()
    return float(diff) <= 1e-9


def _render_terms(cmi, terms) -> str:
    lhs = cmi.to_latex().split(" = ")[0]
    rhs = "".join(
        (r"+\log" if s > 0 else r"-\log") + r"\det\left(" + sp.latex(M) + r"\right)"
        for s, M in terms
    ).lstrip("+")
    signed = Counter()
    for s, M in terms:
        signed[M] += s
    if all(v == 0 for v in signed.values()):
        rhs += " = 0"
    return f"{lhs} = {rhs}"


def _expanded_latex(cmi) -> str:
    """Fallback: the 3-term expanded rendering with display cleanup, guarded."""
    try:
        clean = cmi.simplify("display")
        if _values_match(cmi, clean):
            return clean.to_latex(expand=True, simplify=None, det_style="det")
    except Exception:
        pass
    return cmi.to_latex(expand=True, det_style="det")


def _gradient_payload(G, var) -> dict[str, str]:
    """LaTeX + Mathematica of an already-computed Wirtinger gradient ``G``.

    ``var`` is the differentiation symbol (its ``sp.latex`` labels ``∂I/∂var*``).
    The Mathematica form is a deterministic print of the same ``G`` whose value is
    checked by ``check_gradient``; guard it so a print failure never drops the
    LaTeX.
    """
    v = sp.latex(var)
    grad = {
        "var": v,
        "latex": rf"\frac{{\partial I}}{{\partial {v}^{{*}}}} = {sp.latex(G)}",
    }
    try:
        grad["mathematica"] = to_mathematica(G)
    except Exception:
        pass
    return grad


def _compute(graph: GraphIn) -> dict[str, Any]:
    A = [n.id for n in graph.nodes if n.role == "A"]
    B = [n.id for n in graph.nodes if n.role == "B"]
    C = [n.id for n in graph.nodes if n.role == "C"]
    if not A or not B:
        raise ComputeError(400, "Select at least one A node and one B node.")

    dag, gains, precoders, model_lines = _build(graph)
    cmi = dag.cmi(A, B, C)

    out: dict[str, Any] = {
        "model": model_lines,
        "latex": cmi.to_latex(det_style="det"),
        "independent": bool(cmi.is_conditionally_independent()),
    }
    # Mathematica/Wolfram form of the CMI: a deterministic print of the SAME
    # verified symbolic expression (so it inherits the PyTorch check below).
    # Formatting must never block the verified result, so guard it.
    try:
        out["mathematica"] = to_mathematica(cmi)
    except Exception:
        pass
    if graph.expand:
        try:
            two = cmi.two_term().simplify("display")
            if not _values_match(cmi, two):
                raise ValueError("two-term form mismatch")
            out["latex_expanded"] = _render_terms(cmi, two.logdet_terms)
            cap = simplify_logdet_terms(two.logdet_terms, "capacity")
            if (
                cap is not None
                and 0 < len(cap) < len(two.logdet_terms)
                and _values_match(cmi, _with_terms(cmi, cap))
            ):
                out["latex_capacity"] = _render_terms(cmi, cap)
            # Mathematica forms of the same (verified) two-term / capacity logdets.
            # Nested guard so a printing failure never loses the verified LaTeX.
            try:
                out["mathematica_expanded"] = to_mathematica(
                    _with_terms(cmi, two.logdet_terms)
                )
                if "latex_capacity" in out:
                    out["mathematica_capacity"] = to_mathematica(_with_terms(cmi, cap))
            except Exception:
                pass
        except Exception:
            # fall back to the cleaned 3-term rendering (its own guard inside)
            out["latex_expanded"] = _expanded_latex(cmi)

    if graph.grad_edge is not None:
        key = (graph.grad_edge.source, graph.grad_edge.target)
        H = gains.get(key)
        if H is None:
            raise ComputeError(400, "The edge to differentiate is not in the graph.")
        G = cmi.wirtinger_grad(H)
        out["gradient"] = _gradient_payload(G, H)

    if graph.grad_node is not None:
        F = precoders.get(graph.grad_node)
        if F is None:
            raise ComputeError(
                400,
                "The selected node has no precoder (or has no outgoing edge, so F "
                "does not appear in the expression); ∂I/∂F* is undefined.",
            )
        G = cmi.wirtinger_grad(F)
        out["gradient"] = _gradient_payload(G, F)

    if graph.lmmse:
        if len(A) != 1:
            out["lmmse"] = {"note": "LMMSE is shown only when A is a single node (A = the estimation target)."}
        else:
            try:
                K = cmi.metadata["K"]
                target = cmi.A[0]
                obs = sorted(cmi.B + cmi.C)  # the receiver observes B and knows C
                W_raw = lmmse_estimator(K, target, obs)
                E_raw = mmse_error_covariance(K, target, obs)
                W = simplify_expr(W_raw, "display")
                E = simplify_expr(E_raw, "display")
                if not (_matrices_match(cmi, W, W_raw) and _matrices_match(cmi, E, E_raw)):
                    W, E = W_raw, E_raw  # never display an unverified cleanup
                names = cmi.metadata.get("node_names", {})
                a = names.get(target, str(target))
                ob = ",".join(names.get(i, str(i)) for i in obs)
                out["lmmse"] = {
                    "W": rf"W = {sp.latex(W)}",
                    "E": rf"E = \Sigma_{{{a}\mid {ob}}} = {sp.latex(E)}",
                }
                # Mathematica forms of the same (verified) W / E matrices.
                try:
                    out["lmmse"]["W_mathematica"] = "W = " + to_mathematica(W)
                    out["lmmse"]["E_mathematica"] = "E = " + to_mathematica(E)
                except Exception:
                    pass
            except Exception as exc:
                out["lmmse"] = {"note": f"LMMSE derivation failed: {exc}"}

    if graph.check:
        try:
            res = cmi.check(dim=graph.check_dim)
            out["check"] = {
                "passed": bool(res["passed"]),
                "max_abs_err": float(res["max_abs_err"]),
            }
        except Exception as exc:  # torch missing, degenerate point, ...
            out["check"] = {"passed": None, "error": str(exc)}

    return out


# ---- timeout harness: symbolic blow-up must not hang the UI -----------------
# sympy rewriting is CPU-bound and uninterruptible in-thread, so each request
# runs in a worker PROCESS that can be killed on timeout (the UI then shows a
# clear "graph too complex" error instead of freezing mid-demo).

TIMEOUT_S = float(os.environ.get("DAG_DEMO_TIMEOUT", "30"))
_POOL: cf.ProcessPoolExecutor | None = None


def _get_pool() -> cf.ProcessPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = cf.ProcessPoolExecutor(max_workers=1)
    return _POOL


def _kill_pool() -> None:
    global _POOL
    if _POOL is not None:
        for p in _POOL._processes.values():
            p.kill()
        _POOL.shutdown(wait=False, cancel_futures=True)
        _POOL = None


def _compute_worker(graph_dict: dict) -> dict[str, Any]:
    # ComputeError does not survive the pickle boundary reliably; ferry it.
    try:
        return _compute(GraphIn(**graph_dict))
    except ComputeError as e:
        return {"_error": e.detail, "_status": e.status}


def run_compute(graph: GraphIn) -> dict[str, Any]:
    """Run :func:`_compute` in a killable worker process with a timeout.

    Raises :class:`ComputeError` on timeout or on a user-facing error inside the
    worker. Returns the result payload otherwise.
    """
    future = _get_pool().submit(_compute_worker, graph.model_dump())
    try:
        result = future.result(timeout=TIMEOUT_S)
    except cf.TimeoutError:
        _kill_pool()
        raise ComputeError(
            408,
            f"Symbolic computation timed out after {TIMEOUT_S:.0f}s (the expression "
            "blows up at this size). Shrink the graph, or uncheck the expanded form "
            "/ LMMSE / numerical check.",
        )
    if "_error" in result:
        raise ComputeError(result.get("_status", 400), result["_error"])
    return result


# ---- code export: graph -> runnable symbolic_dag source (no symbolic compute) -
# Pure string building from the drawn graph. This is the same named->index
# lowering core._build does, emitted as text instead of objects, so it is instant
# (no symbolic computation, no timeout needed). Two levels:
#   "high"  -> the named-node builder (GaussianDAG / add_source / add_node / cmi)
#   "low"   -> the functional core (compute_k_blocks_multiroot + cmi_from_k),
#              which requires the sources to occupy the index prefix.


def _pylist(xs: list[str]) -> str:
    return "[" + ", ".join(f'"{x}"' for x in xs) + "]"


def _gain_expr(p: str, n: str, has_precoder: dict[str, bool]) -> str:
    g = f'MatrixSymbol("H_{p}{n}", d, d)'
    if has_precoder.get(p):
        g += f' * MatrixSymbol("F_{p}", d, d)'  # precoded node: outgoing edge is H*F
    return g


def _export_structure(graph: GraphIn):
    """Validate and derive (order, parents_of, sources, has_precoder, A, B, C)."""
    names = [n.id for n in graph.nodes]
    if len(set(names)) != len(names):
        raise ComputeError(400, "Duplicate node names.")
    nameset = set(names)
    for e in graph.edges:
        if e.source not in nameset or e.target not in nameset:
            raise ComputeError(400, "An edge references an undefined node.")
        if e.source == e.target:
            raise ComputeError(400, "Self-loops are not allowed.")
    order = _toposort(names, graph.edges)
    parents_of: dict[str, list[str]] = {n: [] for n in names}
    for e in graph.edges:
        parents_of[e.target].append(e.source)
    has_precoder = {n.id: n.precoder for n in graph.nodes}
    sources = {n for n in names if not parents_of[n]}
    A = [n.id for n in graph.nodes if n.role == "A"]
    B = [n.id for n in graph.nodes if n.role == "B"]
    C = [n.id for n in graph.nodes if n.role == "C"]
    return order, parents_of, sources, has_precoder, A, B, C


def _source_high(order, parents_of, sources, has_precoder, A, B, C) -> str:
    L = [
        "import sympy as sp",
        "from sympy import MatrixSymbol",
        "from symbolic_dag import GaussianDAG, hermitian",
        "",
        'd = sp.Symbol("d", positive=True, integer=True)',
        "",
        "G = GaussianDAG()",
    ]
    for n in order:
        if n in sources:
            L.append(f'G.add_source("{n}", cov=hermitian("Sigma_{n}", d))')
        else:
            items = ", ".join(
                f'"{p}": {_gain_expr(p, n, has_precoder)}' for p in parents_of[n]
            )
            L.append(f'G.add_node("{n}", parents={{{items}}}, noise=hermitian("N_{n}", d))')
    L.append("")
    if A and B:
        L.append(f"cmi = G.cmi({_pylist(A)}, {_pylist(B)}, {_pylist(C)})")
        L.append("print(cmi.to_latex())")
    else:
        L.append("# Assign roles A and B (>=1 node each) to form a query, e.g.:")
        L.append('# cmi = G.cmi(["A_node"], ["B_node"], ["C_node"])')
    return "\n".join(L) + "\n"


def _source_low(order, parents_of, sources, has_precoder, A, B, C) -> str:
    src = [n for n in order if n in sources]  # sources first: required by the
    rest = [n for n in order if n not in sources]  # functional core's index prefix
    seq = src + rest
    idx = {n: i for i, n in enumerate(seq)}
    L = [
        "import sympy as sp",
        "from sympy import MatrixSymbol",
        "from symbolic_dag import (",
        "    compute_k_blocks_multiroot,",
        "    conditional_mutual_information_from_k,",
        "    hermitian,",
        ")",
        "",
        'd = sp.Symbol("d", positive=True, integer=True)',
        "",
        "# node index (sources first): " + ", ".join(f"{idx[n]}={n}" for n in seq),
        "edge_mats = {",
    ]
    for n in rest:
        for p in parents_of[n]:
            L.append(f"    ({idx[n]}, {idx[p]}): {_gain_expr(p, n, has_precoder)},")
    L.append("}")
    parents_lit = ", ".join(f"{idx[n]}: {[idx[p] for p in parents_of[n]]}" for n in rest)
    rootcov = ", ".join(f'{idx[s]}: hermitian("Sigma_{s}", d)' for s in src)
    noisecov = ", ".join(f'{idx[n]}: hermitian("N_{n}", d)' for n in rest)
    L += [
        "",
        "K = compute_k_blocks_multiroot(",
        f"    num_nodes={len(seq)}, roots={[idx[s] for s in src]},",
        f"    parents={{{parents_lit}}},",
        "    edge_mats=edge_mats,",
        f"    root_covs={{{rootcov}}},",
        f"    noise_covs={{{noisecov}}},",
        ")",
        "",
    ]
    if A and B:
        ai, bi, ci = [idx[n] for n in A], [idx[n] for n in B], [idx[n] for n in C]
        L.append(f"cmi = conditional_mutual_information_from_k(K, A={ai}, B={bi}, C={ci})")
        L.append("print(cmi.to_latex())")
    else:
        L.append("# Assign roles A and B to form a query, e.g.:")
        L.append("# cmi = conditional_mutual_information_from_k(K, A=[0], B=[1], C=[])")
    return "\n".join(L) + "\n"


def to_source(graph: GraphIn, level: str = "high") -> str:
    """Emit runnable symbolic_dag source for the drawn graph (``high``/``low``)."""
    order, parents_of, sources, has_precoder, A, B, C = _export_structure(graph)
    if level == "low":
        return _source_low(order, parents_of, sources, has_precoder, A, B, C)
    return _source_high(order, parents_of, sources, has_precoder, A, B, C)
