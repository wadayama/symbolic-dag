"""Unit tests for symbolic_dag.rewrite — d-separation proof + rule identities."""

from __future__ import annotations

import numpy as np
import sympy as sp
from sympy import Adjoint, Identity, Inverse, MatrixSymbol

from symbolic_dag.assumptions import hermitian
from symbolic_dag.information import conditional_mutual_information_from_k
from symbolic_dag.krecursion import compute_k_blocks_multiroot
from symbolic_dag.rewrite import (
    EXPANSION,
    STRUCTURAL,
    proves_zero,
    run_phases,
    simplify_expr,
)

DIM = sp.Symbol("n", positive=True, integer=True)


def _chain():
    A, B = MatrixSymbol("A", DIM, DIM), MatrixSymbol("B", DIM, DIM)
    SX, SY, SZ = (hermitian(s, DIM) for s in ("Sigma_X", "Sigma_Y", "Sigma_Z"))
    return compute_k_blocks_multiroot(
        num_nodes=3, roots=[0], parents={1: [0], 2: [1]},
        edge_mats={(1, 0): A, (2, 1): B}, root_covs={0: SX}, noise_covs={1: SY, 2: SZ},
    )


def _fork():
    A, B = MatrixSymbol("A", DIM, DIM), MatrixSymbol("B", DIM, DIM)
    SX, SY, SZ = (hermitian(s, DIM) for s in ("Sigma_X", "Sigma_Y", "Sigma_Z"))
    return compute_k_blocks_multiroot(
        num_nodes=3, roots=[0], parents={1: [0], 2: [0]},
        edge_mats={(1, 0): A, (2, 0): B}, root_covs={0: SX}, noise_covs={1: SY, 2: SZ},
    )


def test_chain_dsep_proved():
    K = _chain()
    I = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])
    assert proves_zero(I.cross)               # engine proves S = 0
    assert I.is_conditionally_independent()


def test_fork_dsep_proved():
    K = _fork()
    I = conditional_mutual_information_from_k(K, A=[1], B=[2], C=[0])  # I(Y;Z|X)
    assert I.is_conditionally_independent()


def test_collider_not_independent_when_conditioned():
    # collider X->Z<-Y: I(X;Y) marginal is 0, but I(X;Y|Z) is NOT (cross != 0)
    A, B = MatrixSymbol("A", DIM, DIM), MatrixSymbol("B", DIM, DIM)
    SX, SY, SZ = (hermitian(s, DIM) for s in ("Sigma_X", "Sigma_Y", "Sigma_Z"))
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): A, (2, 1): B}, root_covs={0: SX, 1: SY}, noise_covs={2: SZ},
    )
    I_marg = conditional_mutual_information_from_k(K, [0], [1], [])
    I_cond = conditional_mutual_information_from_k(K, [0], [1], [2])
    assert I_marg.is_conditionally_independent()           # marginal indep
    assert not I_cond.is_conditionally_independent()       # opened by collider


def _rng(s):
    return np.random.default_rng(s)


def test_det_lemma_identity_numeric():
    # log det(Σ + U V) == log det Σ + log det(I + V Σ^-1 U), engine-rewritten
    n = DIM
    S = hermitian("Sigma", n)
    U, V = MatrixSymbol("U", n, n), MatrixSymbol("V", n, n)
    expr = sp.log(sp.Determinant(S + U * V))
    rew = run_phases(expr, [EXPANSION])["expr"]
    d, rng = 3, _rng(1)
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    Sn = A @ A.conj().T + d * np.eye(d)
    Un = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    Vn = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    sub = {S: sp.Matrix(Sn), U: sp.Matrix(Un), V: sp.Matrix(Vn), n: d}

    def ldet(M):
        r = M.subs(sub).doit()
        r = r if isinstance(r, sp.MatrixBase) else r.as_explicit()
        return np.linalg.slogdet(np.array(r.tolist(), complex))[1]

    raw = np.linalg.slogdet(Sn + Un @ Vn)[1]
    rew_val = sum(ldet(t.args[0].arg) for t in rew.args)  # sum of log det terms
    assert abs(raw - rew_val) < 1e-8


def test_woodbury_identity_numeric():
    n = DIM
    S = hermitian("Sigma", n)
    U, V, Cm = (MatrixSymbol(s, n, n) for s in ("U", "V", "Cm"))
    expr = Inverse(S + U * Cm * V)
    rew = run_phases(expr, [EXPANSION])["expr"]
    d, rng = 3, _rng(2)

    def npm(M, sub):
        r = M.subs(sub).doit()
        r = r if isinstance(r, sp.MatrixBase) else r.as_explicit()
        return np.array(r.tolist(), complex)

    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    Sn = A @ A.conj().T + d * np.eye(d)
    Un = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    Vn = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    B = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    Cn = B @ B.conj().T + d * np.eye(d)
    sub = {S: sp.Matrix(Sn), U: sp.Matrix(Un), V: sp.Matrix(Vn), Cm: sp.Matrix(Cn), n: d}
    raw = np.linalg.inv(Sn + Un @ Cn @ Vn)
    assert np.max(np.abs(npm(expr, sub) - npm(rew, sub))) < 1e-7
    assert np.max(np.abs(raw - npm(rew, sub))) < 1e-7
