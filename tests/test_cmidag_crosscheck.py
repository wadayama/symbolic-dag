"""Cross-validation against the actual numerical cmi-dag library.

These are the headline verification tests: the symbolic CMI value and the
symbolic Wirtinger gradient are checked against cmi-dag's own numerical
computation and autograd, on random complex points across dimensions. They run
only when the optional ``cmidag`` extra (torch) and the cmi-dag repository are
available; otherwise they skip.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
from sympy import Identity, MatrixSymbol

from cmidag_oracle import (
    cmidag_available,
    cmidag_cmi,
    cmidag_grad,
    cmidag_precoder_grad,
)

from symbolic_dag.assumptions import hermitian
from symbolic_dag.information import conditional_mutual_information_from_k
from symbolic_dag.krecursion import compute_k_blocks_multiroot

pytestmark = pytest.mark.skipif(
    not cmidag_available(),
    reason="cmi-dag repository unavailable (set SYMBOLIC_DAG_CMIDAG_PATH to a checkout)",
)

DIM = sp.Symbol("n", positive=True, integer=True)


def _rng(s):
    return np.random.default_rng(s)


def _rc(d, rng):
    return rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))


def _hpd(d, rng):
    A = _rc(d, rng)
    return A @ A.conj().T + d * np.eye(d)


# (name, num_nodes, roots, parents, edges[(j,i)->symname], rootcov{idx->symname},
#  noisecov{idx->symname}, A, B, C)
_GADGETS = [
    ("chain", 3, [0], {1: [0], 2: [1]}, {(1, 0): "A", (2, 1): "B"},
     {0: "SX"}, {1: "SY", 2: "SZ"}, [0], [2], []),
    ("chain_cond", 3, [0], {1: [0], 2: [1]}, {(1, 0): "A", (2, 1): "B"},
     {0: "SX"}, {1: "SY", 2: "SZ"}, [0], [2], [1]),
    ("fork", 3, [0], {1: [0], 2: [0]}, {(1, 0): "A", (2, 0): "B"},
     {0: "SX"}, {1: "SY", 2: "SZ"}, [1], [2], [0]),
    ("collider", 3, [0, 1], {2: [0, 1]}, {(2, 0): "A", (2, 1): "B"},
     {0: "SX", 1: "SY"}, {2: "SZ"}, [0], [1], [2]),
    ("mac", 3, [0, 1], {2: [0, 1]}, {(2, 0): "A", (2, 1): "B"},
     {0: "SX", 1: "SY"}, {2: "SZ"}, [0, 1], [2], []),
]


def _symbols():
    A, B = MatrixSymbol("A", DIM, DIM), MatrixSymbol("B", DIM, DIM)
    SX, SY, SZ = (hermitian(s, DIM) for s in ("SX", "SY", "SZ"))
    return {"A": A, "B": B, "SX": SX, "SY": SY, "SZ": SZ}


@pytest.mark.parametrize("gadget", _GADGETS, ids=[g[0] for g in _GADGETS])
@pytest.mark.parametrize("d", [1, 2, 3])
def test_cmi_value_matches_cmidag(gadget, d):
    name, num_nodes, roots, parents, edges, rootcov, noisecov, A, B, C = gadget
    sym = _symbols()
    K = compute_k_blocks_multiroot(
        num_nodes, roots, parents,
        edge_mats={k: sym[v] for k, v in edges.items()},
        root_covs={k: sym[v] for k, v in rootcov.items()},
        noise_covs={k: sym[v] for k, v in noisecov.items()},
    )
    I = conditional_mutual_information_from_k(K, A, B, C)

    rng = _rng(hash((name, d)) % (2**31))
    vals = {"A": _rc(d, rng), "B": _rc(d, rng),
            "SX": _hpd(d, rng), "SY": _hpd(d, rng), "SZ": _hpd(d, rng)}
    subs = {sym[k]: sp.Matrix(vals[k]) for k in sym}
    subs[DIM] = d
    ref = cmidag_cmi(
        num_nodes, roots, parents,
        {k: vals[v] for k, v in edges.items()},
        {k: vals[v] for k, v in rootcov.items()},
        {k: vals[v] for k, v in noisecov.items()},
        A, B, C,
    )
    ok, err = I.numeric_check(subs, ref, atol=1e-10)
    assert ok, f"{name} d={d}: symbolic vs cmi-dag err={err:.2e}"


@pytest.mark.parametrize("d", [1, 2, 3])
def test_wirtinger_gradient_matches_cmidag_autograd(d):
    # precoder: Y=(HF)X0 + X1 + N, I(X0;Y|X1); cmi-dag autograd = 2 x symbolic grad
    H, F = MatrixSymbol("H", DIM, DIM), MatrixSymbol("F", DIM, DIM)
    S0, R = hermitian("Sigma_0", DIM), hermitian("R", DIM)
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): H * F, (2, 1): Identity(DIM)},
        root_covs={0: S0, 1: Identity(DIM)}, noise_covs={2: R},
    )
    I = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])
    G = I.wirtinger_grad(F)

    rng = _rng(500 + d)
    Hn, Fn, S0n, Rn = _rc(d, rng), _rc(d, rng), _hpd(d, rng), _hpd(d, rng)
    subs = {H: sp.Matrix(Hn), F: sp.Matrix(Fn),
            S0: sp.Matrix(S0n), R: sp.Matrix(Rn), DIM: d}
    Gn = np.array(
        (G.subs(subs).doit() if isinstance(G.subs(subs).doit(), sp.MatrixBase)
         else G.subs(subs).doit().as_explicit()).tolist(), complex
    )
    autograd = cmidag_precoder_grad(Hn, Fn, S0n, Rn)
    assert np.max(np.abs(autograd - 2.0 * Gn)) < 1e-9, "gradient mismatch vs autograd"


# Generalized gradient (arbitrary A,B,C with A or B single) vs autograd.
# F sits in the edge (sink, 0) = H @ F; other root edges are identity.
_GRAD_GADGETS = [
    ("|C|=2", 4, [0, 1, 2], {3: [0, 1, 2]}, 3, [0], [3], [1, 2]),
    ("multinodeA", 3, [0, 1], {2: [0, 1]}, 2, [0, 1], [2], []),
    ("multinodeA_|C|=1", 4, [0, 1, 2], {3: [0, 1, 2]}, 3, [0, 1], [3], [2]),
]


@pytest.mark.parametrize("gadget", _GRAD_GADGETS, ids=[g[0] for g in _GRAD_GADGETS])
@pytest.mark.parametrize("d", [2, 3])
def test_generalized_gradient_matches_autograd(gadget, d):
    name, nn, roots, parents, sink, A, B, C = gadget
    H, F = MatrixSymbol("H", DIM, DIM), MatrixSymbol("F", DIM, DIM)
    S0, R = hermitian("Sigma_0", DIM), hermitian("R", DIM)
    edge = {(sink, 0): H * F}
    root_covs = {0: S0}
    for i in roots:
        if i != 0:
            edge[(sink, i)] = Identity(DIM)
            root_covs[i] = Identity(DIM)
    K = compute_k_blocks_multiroot(nn, roots, parents, edge, root_covs, {sink: R})
    I = conditional_mutual_information_from_k(K, A, B, C)
    G = I.wirtinger_grad(F)  # arbitrary A,B,C (A or B single), arbitrary |C|

    rng = _rng(900 + d + hash(name) % 100)
    Hn, Fn, S0n, Rn = _rc(d, rng), _rc(d, rng), _hpd(d, rng), _hpd(d, rng)
    subs = {H: sp.Matrix(Hn), F: sp.Matrix(Fn), S0: sp.Matrix(S0n), R: sp.Matrix(Rn), DIM: d}
    r = G.subs(subs).doit()
    Gn = np.array((r if isinstance(r, sp.MatrixBase) else r.as_explicit()).tolist(), complex)

    static = {}
    rc_np = {0: S0n}
    for i in roots:
        if i != 0:
            static[(sink, i)] = np.eye(d) + 0j
            rc_np[i] = np.eye(d) + 0j
    autograd = cmidag_grad(nn, roots, parents, static, rc_np, {sink: Rn},
                           A, B, C, (sink, 0), Hn, Fn)
    assert np.max(np.abs(autograd - 2.0 * Gn)) < 1e-9, f"{name} d={d}"


@pytest.mark.parametrize("d", [1, 2, 3])
def test_dseparation_proved_and_numerically_zero(d):
    sym = _symbols()
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0], parents={1: [0], 2: [1]},
        edge_mats={(1, 0): sym["A"], (2, 1): sym["B"]},
        root_covs={0: sym["SX"]}, noise_covs={1: sym["SY"], 2: sym["SZ"]},
    )
    I = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])  # I(X;Z|Y)
    assert I.is_conditionally_independent()  # symbolic proof

    rng = _rng(700 + d)
    vals = {"A": _rc(d, rng), "B": _rc(d, rng),
            "SX": _hpd(d, rng), "SY": _hpd(d, rng), "SZ": _hpd(d, rng)}
    ref = cmidag_cmi(
        3, [0], {1: [0], 2: [1]},
        {(1, 0): vals["A"], (2, 1): vals["B"]},
        {0: vals["SX"]}, {1: vals["SY"], 2: vals["SZ"]},
        [0], [2], [1],
    )
    assert abs(ref) < 1e-9  # cmi-dag numeric also ~0
