"""Unit tests for symbolic_dag.information / expr — symbolic CMI vs NumPy oracle."""

from __future__ import annotations

import numpy as np
import sympy as sp
from sympy import MatrixSymbol

from symbolic_dag.assumptions import hermitian
from symbolic_dag.information import conditional_mutual_information_from_k
from symbolic_dag.krecursion import compute_k_blocks_multiroot
from symbolic_dag.numeric import numpy_cmi, numpy_k_blocks

DIM = sp.Symbol("n", positive=True, integer=True)


def _rng(seed):
    return np.random.default_rng(seed)


def _rc(d, rng):
    return rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))


def _hpd(d, rng):
    A = _rc(d, rng)
    return A @ A.conj().T + d * np.eye(d)


def _chain():
    A, B = MatrixSymbol("A", DIM, DIM), MatrixSymbol("B", DIM, DIM)
    SX, SY, SZ = (hermitian(s, DIM) for s in ("Sigma_X", "Sigma_Y", "Sigma_Z"))
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0], parents={1: [0], 2: [1]},
        edge_mats={(1, 0): A, (2, 1): B}, root_covs={0: SX}, noise_covs={1: SY, 2: SZ},
    )
    syms = {"A": A, "B": B, "Sigma_X": SX, "Sigma_Y": SY, "Sigma_Z": SZ}
    return K, syms


def _mac():
    A, B = MatrixSymbol("A", DIM, DIM), MatrixSymbol("B", DIM, DIM)
    SX, SY, SZ = (hermitian(s, DIM) for s in ("Sigma_X", "Sigma_Y", "Sigma_Z"))
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): A, (2, 1): B}, root_covs={0: SX, 1: SY}, noise_covs={2: SZ},
    )
    syms = {"A": A, "B": B, "Sigma_X": SX, "Sigma_Y": SY, "Sigma_Z": SZ}
    return K, syms


def _subs_and_np(syms, d, seed):
    rng = _rng(seed)
    vals = {
        "A": _rc(d, rng), "B": _rc(d, rng),
        "Sigma_X": _hpd(d, rng), "Sigma_Y": _hpd(d, rng), "Sigma_Z": _hpd(d, rng),
    }
    subs = {syms[k]: sp.Matrix(vals[k]) for k in syms}
    subs[DIM] = d
    return subs, vals


def test_chain_value_matches_numpy_oracle():
    K, syms = _chain()
    for d in (1, 2, 3):
        subs, vals = _subs_and_np(syms, d, 10 + d)
        I = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[])  # I(X;Z)
        Knp = numpy_k_blocks(
            3, [0], {1: [0], 2: [1]},
            {(1, 0): vals["A"], (2, 1): vals["B"]},
            {0: vals["Sigma_X"]}, {1: vals["Sigma_Y"], 2: vals["Sigma_Z"]},
        )
        ref = numpy_cmi(Knp, [0], [2], [])
        ok, err = I.numeric_check(subs, ref, atol=1e-9)
        assert ok, f"d={d} err={err:.2e}"


def test_chain_dsep_value_zero():
    K, syms = _chain()
    I = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])  # I(X;Z|Y)
    for d in (1, 2, 3):
        subs, _ = _subs_and_np(syms, d, 20 + d)
        assert abs(I.evaluate(subs)) < 1e-9


def test_mac_multinode_matches_numpy_oracle():
    K, syms = _mac()
    for d in (1, 2):
        subs, vals = _subs_and_np(syms, d, 30 + d)
        I = conditional_mutual_information_from_k(K, A=[0, 1], B=[2], C=[])  # I(X0,X1; Y)
        Knp = numpy_k_blocks(
            3, [0, 1], {2: [0, 1]},
            {(2, 0): vals["A"], (2, 1): vals["B"]},
            {0: vals["Sigma_X"], 1: vals["Sigma_Y"]}, {2: vals["Sigma_Z"]},
        )
        ref = numpy_cmi(Knp, [0, 1], [2], [])
        ok, err = I.numeric_check(subs, ref, atol=1e-9)
        assert ok, f"d={d} err={err:.2e}"


def test_mac_conditional_chain_rule():
    # I(X0,X1;Y) == I(X0;Y|X1) + I(X1;Y)   (numeric chain rule)
    K, syms = _mac()
    for d in (1, 2):
        subs, _ = _subs_and_np(syms, d, 40 + d)
        I_joint = conditional_mutual_information_from_k(K, [0, 1], [2], []).evaluate(subs)
        I_0g1 = conditional_mutual_information_from_k(K, [0], [2], [1]).evaluate(subs)
        I_1 = conditional_mutual_information_from_k(K, [1], [2], []).evaluate(subs)
        assert abs(I_joint - (I_0g1 + I_1)) < 1e-9
