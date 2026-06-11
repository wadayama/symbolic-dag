"""MMSE / linear KKT solving: ``solve_stationary`` and ``lmmse_estimator``.

The MMSE estimator MSE is quadratic in the filter ``W``, so its Wirtinger gradient
is affine and the stationarity ``dMSE/dW* = 0`` has the closed-form Wiener
solution. These tests build the MSE objective, differentiate it with the trace
engine, solve the resulting linear equation, and check numerically that the
solution (a) zeroes the gradient, (b) matches the closed-form ``lmmse_estimator``,
and (c) that the synthetic one-/two-sided shapes solve correctly. Requires torch.
"""

from __future__ import annotations

import importlib.util

import pytest
import sympy as sp
from sympy import Identity, MatrixSymbol

from symbolic_dag import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
    get_K,
    hermitian,
    lmmse_estimator,
    mmse_error_covariance,
    solve_stationary,
    to_torch,
    trace_grad,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch unavailable (it is a core dependency; run `uv sync`)",
)

DIM = sp.Symbol("d", positive=True, integer=True)


def _channel_K():
    # X(0) -> Y(1): constant channel H, noise R. Estimate X from Y.
    H = MatrixSymbol("H", DIM, DIM)
    SX, R = hermitian("Sigma_X", DIM), hermitian("R", DIM)
    K = compute_k_blocks_multiroot(
        num_nodes=2, roots=[0], parents={1: [0]},
        edge_mats={(1, 0): H}, root_covs={0: SX}, noise_covs={1: R},
    )
    return K, H, SX, R


def _rng_subs(symbols_hpd, symbols_any, d, seed):
    import torch

    g = torch.Generator().manual_seed(seed)
    C = torch.complex128

    def rc():
        return torch.complex(
            torch.randn(d, d, dtype=torch.float64, generator=g),
            torch.randn(d, d, dtype=torch.float64, generator=g),
        )

    def hpd():
        A = rc()
        return A @ A.mH + d * torch.eye(d, dtype=C)

    subs = {s: hpd() for s in symbols_hpd}
    subs.update({s: rc() for s in symbols_any})
    return subs


@pytest.mark.parametrize("d", [2, 3])
def test_solve_stationary_recovers_wiener(d):
    import torch

    K, H, SX, R = _channel_K()
    W = MatrixSymbol("W", DIM, DIM)
    SXX, SYY = get_K(K, 0, 0), get_K(K, 1, 1)
    SXY, SYX = get_K(K, 0, 1), get_K(K, 1, 0)
    # MSE of the linear estimator X_hat = W Y
    E = SXX - W * SYX - SXY * W.adjoint() + W * SYY * W.adjoint()
    G = trace_grad(E, W)                       # affine in W
    W_sol = solve_stationary(G, W)             # closed-form optimum
    W_ref = lmmse_estimator(K, 0, [1])         # Wiener filter

    subs = _rng_subs([SX, R], [H], d, seed=40 + d)
    Wsol_n = to_torch(W_sol, subs, d)
    # (a) matches the closed-form Wiener filter
    assert float((Wsol_n - to_torch(W_ref, subs, d)).abs().max()) < 1e-9
    # (b) zeroes the gradient
    Gn = to_torch(G, {**subs, W: Wsol_n}, d)
    assert float(Gn.abs().max()) < 1e-9
    # (c) achieves the MMSE error covariance
    M = mmse_error_covariance(K, 0, [1])
    Esol = to_torch(E, {**subs, W: Wsol_n}, d)
    assert float((Esol - to_torch(M, subs, d)).abs().max()) < 1e-9


@pytest.mark.parametrize("d", [2, 3])
def test_solve_stationary_left_and_two_sided(d):
    import torch

    P, Q, Sm = (MatrixSymbol(s, DIM, DIM) for s in ("P", "Q", "Sm"))
    W = MatrixSymbol("W", DIM, DIM)
    subs = _rng_subs([], [P, Q, Sm], d, seed=70 + d)

    # left-linear: P W + Sm = 0  ->  W = -P^{-1} Sm
    W_left = solve_stationary(P * W + Sm, W)
    res_l = to_torch(P, subs, d) @ to_torch(W_left, subs, d) + to_torch(Sm, subs, d)
    assert float(res_l.abs().max()) < 1e-9

    # two-sided single term: P W Q + Sm = 0  ->  W = -P^{-1} Sm Q^{-1}
    W_two = solve_stationary(P * W * Q + Sm, W)
    res_t = (to_torch(P, subs, d) @ to_torch(W_two, subs, d) @ to_torch(Q, subs, d)
             + to_torch(Sm, subs, d))
    assert float(res_t.abs().max()) < 1e-9


def test_solve_stationary_rejects_nonlinear():
    # A capacity gradient has the variable nested inside an inverse -> not linear.
    H, F = MatrixSymbol("H", DIM, DIM), MatrixSymbol("F", DIM, DIM)
    S0, R = hermitian("Sigma_0", DIM), hermitian("R", DIM)
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): H * F, (2, 1): Identity(DIM)},
        root_covs={0: S0, 1: Identity(DIM)}, noise_covs={2: R},
    )
    I = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])
    with pytest.raises(NotImplementedError):
        solve_stationary(I.wirtinger_grad(F), F)


def test_solve_stationary_rejects_var_in_scalar_coefficient():
    # `var` hiding inside a SCALAR coefficient (e.g. Trace(W P) * W) makes the
    # equation nonlinear; it must raise, not return a pseudo-solution.
    from sympy import Trace

    W, P, S = (MatrixSymbol(s, DIM, DIM) for s in ("W", "P", "S"))
    with pytest.raises(NotImplementedError):
        solve_stationary(Trace(W * P) * W + S, W)
