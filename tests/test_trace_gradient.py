"""Trace / MMSE-objective Wirtinger gradient vs PyTorch autograd.

The estimation-error covariance ``Sigma_{X|Y}`` is built block-free, its trace is
the scalar MMSE, and ``trace_grad`` gives the closed-form Wirtinger gradient w.r.t.
a precoder. The check differentiates the *same* objective with PyTorch autograd
(``autograd == 2 * symbolic``), so it is an independent cross-check of the symbolic
matrix-calculus engine. Requires torch (a core dependency).
"""

from __future__ import annotations

import importlib.util

import pytest
import sympy as sp
from sympy import Identity, MatrixSymbol

from symbolic_dag import (
    compute_k_blocks_multiroot,
    conditional_covariance_seq,
    hermitian,
    mmse_error_covariance,
    to_torch,
    trace_grad,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch unavailable (it is a core dependency; run `uv sync`)",
)

DIM = sp.Symbol("d", positive=True, integer=True)


def _precoder_K():
    # Y = (H F) X0 + X1 + N; estimate X0 from Y.
    H, F = MatrixSymbol("H", DIM, DIM), MatrixSymbol("F", DIM, DIM)
    S0, R = hermitian("Sigma_0", DIM), hermitian("R", DIM)
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): H * F, (2, 1): Identity(DIM)},
        root_covs={0: S0, 1: Identity(DIM)}, noise_covs={2: R},
    )
    return K, H, F, S0, R


def _autograd_trace_grad(M, F, subs, d):
    """PyTorch autograd of ``tr(M)`` w.r.t. ``F`` (== 2 x the Wirtinger gradient)."""
    import torch

    J = torch.trace(to_torch(M, subs, d)).real
    J.backward()
    return subs[F].grad


@pytest.mark.parametrize("d", [2, 3])
def test_trace_grad_matches_autograd(d):
    import torch

    K, H, F, S0, R = _precoder_K()
    M = mmse_error_covariance(K, target=0, observations=[2])  # Sigma_{X0|Y}
    G = trace_grad(M, F)  # closed-form d tr(M) / dF*

    g = torch.Generator().manual_seed(10 + d)
    C = torch.complex128

    def rc():
        return torch.complex(
            torch.randn(d, d, dtype=torch.float64, generator=g),
            torch.randn(d, d, dtype=torch.float64, generator=g),
        )

    def hpd():
        A = rc()
        return A @ A.mH + d * torch.eye(d, dtype=C)

    subs = {H: rc(), F: rc().clone().requires_grad_(True), S0: hpd(), R: hpd()}
    autograd = _autograd_trace_grad(M, F, subs, d)
    subs_ng = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in subs.items()}
    err = float((autograd - 2.0 * to_torch(G, subs_ng, d)).abs().max())
    assert err < 1e-7, f"d={d}: trace-grad vs autograd err={err:.2e}"


def test_mmse_error_covariance_is_conditional_cov():
    K, *_ = _precoder_K()
    assert mmse_error_covariance(K, 0, [2]) == conditional_covariance_seq(K, 0, [2])
