"""Tests for the internal Fisher/CRB layer (symbolic_dag._fisher).

NOT a public-API test: ``_fisher`` is an experimental, unexported (ISAC) layer.
The headline check is the design gradient ``dCRB/dF*`` vs PyTorch autograd, and a
reproduction of the posterior-CRB closed form of an ISAC beamforming paper
(2026002049). Requires torch (a core dependency).
"""

from __future__ import annotations

import importlib.util

import pytest
import sympy as sp
from sympy import Adjoint, MatrixSymbol, Trace

from symbolic_dag import hermitian
from symbolic_dag._fisher import (
    cramer_rao_bound,
    crb_grad_check,
    crb_trace,
    crb_value,
    fisher_information_matrix,
)
from symbolic_dag._fisher import _random_point  # test helper

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch unavailable (it is a core dependency; run `uv sync`)",
)

DIM = sp.Symbol("d", positive=True, integer=True)


def _isac_fim():
    """The 2x2 transmit-covariance FIM of an ISAC point-target model (cf. 2026002049)."""
    F, A2 = MatrixSymbol("F", DIM, DIM), MatrixSymbol("A_2", DIM, DIM)
    Sig = hermitian("Sigma", DIM)
    A1, A3 = hermitian("A_1", DIM), hermitian("A_3", DIM)
    RX = F * Sig * F.adjoint()  # transmit covariance R_X = F Sigma F^H
    J = sp.Matrix([[Trace(A1 * RX), Trace(A2 * RX)],
                   [Trace(Adjoint(A2) * RX), Trace(A3 * RX)]])
    return J, F


def test_pcrb_reproduces_paper_form():
    # cramer_rao_bound(J, 0) == tr(A3 RX) / (tr(A1 RX) tr(A3 RX) - |tr(A2 RX)|^2)
    J, F = _isac_fim()
    pcrb = cramer_rao_bound(J, 0)
    t1, t2, t2c, t3 = (J[0, 0], J[0, 1], J[1, 0], J[1, 1])
    expected = t3 / (t1 * t3 - t2 * t2c)
    subs = _random_point(
        {a for a in sp.preorder_traversal(pcrb) if isinstance(a, MatrixSymbol)}, 3, seed=5
    )
    assert abs(crb_value(pcrb, subs, 3) - crb_value(expected, subs, 3)) < 1e-9


@pytest.mark.parametrize("d", [2, 3])
def test_crb_grad_matches_autograd(d):
    J, F = _isac_fim()
    assert crb_grad_check(cramer_rao_bound(J, 0), F, dim=d, seed=10 + d)["passed"]
    assert crb_grad_check(crb_trace(J), F, dim=d, seed=20 + d)["passed"]


@pytest.mark.parametrize("d", [2, 3])
def test_fisher_information_matrix_value(d):
    import torch

    # covariance-parameter model: dR/dtheta = R1 -> J = tr(R^-1 R1 R^-1 R1)
    R, R1 = hermitian("R", DIM), hermitian("R1", DIM)
    J = fisher_information_matrix(R, dR=[R1])
    subs = _random_point({R, R1}, d, seed=30 + d)
    Rn, R1n = subs[R], subs[R1]
    Rinv = torch.linalg.inv(Rn)
    ref = float(torch.trace(Rinv @ R1n @ Rinv @ R1n).real)
    assert abs(crb_value(J[0, 0], subs, d) - ref) < 1e-7


def test_fisher_mean_term_value():
    import torch

    # mean-parameter model: dmu/dtheta = m1 (column), R = N -> J = 2 Re(m1^H N^-1 m1)
    d = 3
    m1 = MatrixSymbol("m_1", DIM, 1)
    N = hermitian("N", DIM)
    J = fisher_information_matrix(N, dmu=[m1])
    g = torch.Generator().manual_seed(7)
    C = torch.complex128
    m1n = torch.complex(torch.randn(d, 1, dtype=torch.float64, generator=g),
                        torch.randn(d, 1, dtype=torch.float64, generator=g))
    A = torch.complex(torch.randn(d, d, dtype=torch.float64, generator=g),
                      torch.randn(d, d, dtype=torch.float64, generator=g))
    Nn = A @ A.mH + d * torch.eye(d, dtype=C)
    subs = {m1: m1n, N: Nn}
    ref = float((2 * (m1n.mH @ torch.linalg.inv(Nn) @ m1n)).real)
    assert abs(crb_value(J[0, 0], subs, d) - ref) < 1e-7
