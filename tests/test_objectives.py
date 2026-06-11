"""Tests for composite objectives and the entropy / TC / directed-info / KL quantities."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import sympy as sp
from sympy import Identity, MatrixSymbol

from symbolic_dag import (
    compute_k_blocks_multiroot,
    composite_cmi,
    conditional_entropy_from_k,
    conditional_mutual_information_from_k,
    directed_information_from_k,
    gaussian_kl,
    hermitian,
    total_correlation_from_k,
)

torch_only = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch unavailable (core dependency; run `uv sync`)",
)

DIM = sp.Symbol("d", positive=True, integer=True)


def _mac():
    H1, H2 = MatrixSymbol("H1", DIM, DIM), MatrixSymbol("H2", DIM, DIM)
    S1, S2, N = hermitian("S1", DIM), hermitian("S2", DIM), hermitian("N", DIM)
    K = compute_k_blocks_multiroot(
        3, [0, 1], {2: [0, 1]}, {(2, 0): H1, (2, 1): H2}, {0: S1, 1: S2}, {2: N}
    )
    return K, H1, H2, S1, S2, N


def _eye_subs(D):
    K, H1, H2, S1, S2, N = _mac()
    return {H1: sp.Matrix(np.eye(D)), H2: sp.Matrix(np.eye(D)),
            S1: sp.Matrix(2 * np.eye(D)), S2: sp.Matrix(2 * np.eye(D)),
            N: sp.Matrix(1.5 * np.eye(D)), DIM: D}


# ----- item 1: composite objectives -----
def test_composite_value_is_weighted_sum():
    K, *_ = _mac()
    R1 = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])
    R2 = conditional_mutual_information_from_k(K, A=[1], B=[2], C=[0])
    f = composite_cmi([(2, R1), (1, R2)])
    subs = _eye_subs(3)
    assert abs(f.evaluate(subs) - (2 * R1.evaluate(subs) + R2.evaluate(subs))) < 1e-12


@torch_only
@pytest.mark.parametrize("d", [2, 3])
def test_composite_gradient_matches_autograd(d):
    H1, F1, H2, F2 = (MatrixSymbol(s, DIM, DIM) for s in ("H1", "F1", "H2", "F2"))
    S1, S2, N = hermitian("S1", DIM), hermitian("S2", DIM), hermitian("N", DIM)
    K = compute_k_blocks_multiroot(
        3, [0, 1], {2: [0, 1]}, {(2, 0): H1 * F1, (2, 1): H2 * F2},
        {0: S1, 1: S2}, {2: N},
    )
    R1 = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])
    R2 = conditional_mutual_information_from_k(K, A=[1], B=[2], C=[0])
    f = composite_cmi([(2, R1), (1, R2)])
    assert f.check_gradient(F1, dim=d, seed=d)["passed"]


# ----- item 2: directed information -----
@torch_only
@pytest.mark.parametrize("d", [2, 3])
def test_directed_information_gradient_matches_autograd(d):
    H1, H2, H3 = (MatrixSymbol(s, DIM, DIM) for s in ("H1", "H2", "H3"))
    S1, S2, N = hermitian("S1", DIM), hermitian("S2", DIM), hermitian("N", DIM)
    # x1(0),x2(1) inputs; y1(2)<-x1; y2(3)<-x1,x2,y1 (causal feedback)
    K = compute_k_blocks_multiroot(
        4, [0, 1], {2: [0], 3: [0, 1, 2]},
        {(2, 0): H1, (3, 0): H2, (3, 1): H3, (3, 2): Identity(DIM)},
        {0: S1, 1: S2}, {2: N, 3: N},
    )
    di = directed_information_from_k(K, X_seq=[0, 1], Y_seq=[2, 3])
    assert len(di.terms) == 2
    assert di.check_gradient(H1, dim=d, seed=10 + d)["passed"]


def test_directed_information_rejects_bad_sequences():
    K, *_ = _mac()
    with pytest.raises(ValueError):
        directed_information_from_k(K, X_seq=[0], Y_seq=[2, 1])  # unequal length


# ----- item 2: conditional entropy -----
def test_conditional_entropy_value():
    K, *_ = _mac()
    he = conditional_entropy_from_k(K, A=[2], C=[0, 1])  # h(y|x1,x2) = logdet(pi e N)
    D = 3
    subs = _eye_subs(D)
    ref = float(np.linalg.slogdet(np.pi * np.e * 1.5 * np.eye(D))[1])
    assert abs(he.evaluate(subs) - ref) < 1e-10


@torch_only
@pytest.mark.parametrize("d", [2, 3])
def test_conditional_entropy_gradient(d):
    K, H1, *_ = _mac()
    he = conditional_entropy_from_k(K, A=[2], C=[0])  # depends on H1 through y
    assert he.check_gradient(H1, dim=d, seed=20 + d)["passed"]


# ----- item 2: total correlation -----
def test_total_correlation_two_nodes_equals_cmi():
    # TC(y; x0 | -) over a chain should equal I(x0; y) ... use a fork: condition makes them dependent
    H1, H2 = MatrixSymbol("H1", DIM, DIM), MatrixSymbol("H2", DIM, DIM)
    S, N1, N2 = hermitian("S", DIM), hermitian("N1", DIM), hermitian("N2", DIM)
    K = compute_k_blocks_multiroot(
        3, [0], {1: [0], 2: [0]}, {(1, 0): H1, (2, 0): H2}, {0: S}, {1: N1, 2: N2}
    )
    tc = total_correlation_from_k(K, nodes=[1, 2], C=[])
    I12 = conditional_mutual_information_from_k(K, A=[1], B=[2], C=[])
    D = 3
    subs = {H1: sp.Matrix(np.eye(D)), H2: sp.Matrix(1.3 * np.eye(D)),
            S: sp.Matrix(2 * np.eye(D)), N1: sp.Matrix(np.eye(D)),
            N2: sp.Matrix(np.eye(D)), DIM: D}
    assert abs(tc.evaluate(subs) - I12.evaluate(subs)) < 1e-10


def test_total_correlation_zero_for_independent_roots():
    K, *_ = _mac()
    tc = total_correlation_from_k(K, nodes=[0, 1])  # independent sources
    assert abs(tc.evaluate(_eye_subs(3))) < 1e-10


# ----- item 2: Gaussian KL -----
def test_gaussian_kl_value_and_self_zero():
    Q0, Q1 = hermitian("Q0", DIM), hermitian("Q1", DIM)
    kl = gaussian_kl(Q0, Q1)
    D = 3
    subs = {Q0: sp.Matrix(2 * np.eye(D)), Q1: sp.Matrix(3 * np.eye(D)), DIM: D}
    ref = float(
        np.trace(np.linalg.inv(3 * np.eye(D)) @ (2 * np.eye(D))).real - D
        + np.linalg.slogdet(3 * np.eye(D))[1] - np.linalg.slogdet(2 * np.eye(D))[1]
    )
    assert abs(kl.evaluate(subs) - ref) < 1e-10
    # KL(Q||Q) = 0
    same = {Q0: sp.Matrix(2 * np.eye(D)), Q1: sp.Matrix(2 * np.eye(D)), DIM: D}
    assert abs(kl.evaluate(same)) < 1e-10


@torch_only
@pytest.mark.parametrize("d", [2, 3])
def test_gaussian_kl_gradient_through_design_var(d):
    # Sigma1 = F Q F^H is a function of a precoder F; gradient w.r.t. F is supported.
    F = MatrixSymbol("F", DIM, DIM)
    Q0, Q = hermitian("Q0", DIM), hermitian("Q", DIM)
    kl = gaussian_kl(Q0, F * Q * F.adjoint())
    assert kl.check_gradient(F, dim=d, seed=30 + d)["passed"]


# ----- item 4: Hermitian-variable gradient (d/dQ) -----
@pytest.mark.parametrize("d", [2, 3])
def test_kl_gradient_wrt_covariance(d):
    from symbolic_dag import hermitian_grad_check

    Q0, Q1 = hermitian("Q0", DIM), hermitian("Q1", DIM)
    kl = gaussian_kl(Q0, Q1)
    # d KL/dQ1 = Q1^-1 - Q1^-1 Q0 Q1^-1 ; d KL/dQ0 = Q1^-1 - Q0^-1
    assert hermitian_grad_check(kl, Q1, dim=d, seed=40 + d)["passed"]
    assert hermitian_grad_check(kl, Q0, dim=d, seed=50 + d)["passed"]


@pytest.mark.parametrize("d", [2, 3])
def test_cmi_gradient_wrt_input_covariance(d):
    # d I(X0;Y|X1)/dQ for the precoder model, Q = Cov(X0). (Capacity gradient.)
    from symbolic_dag import hermitian_grad_check

    H, F = MatrixSymbol("H", DIM, DIM), MatrixSymbol("F", DIM, DIM)
    Q, R = hermitian("Q", DIM), hermitian("R", DIM)
    K = compute_k_blocks_multiroot(
        3, [0, 1], {2: [0, 1]}, {(2, 0): H * F, (2, 1): Identity(DIM)},
        {0: Q, 1: Identity(DIM)}, {2: R},
    )
    I = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])
    assert hermitian_grad_check(I, Q, dim=d, seed=60 + d)["passed"]


def test_capacity_gradient_closed_form():
    # d log det(N + H Q H^H)/dQ = H^H (N + H Q H^H)^-1 H  (the MIMO capacity gradient)
    from symbolic_dag.assumptions import apply_hermitian
    from symbolic_dag.matderiv import wirtinger_grad_logdet
    from symbolic_dag.rewrite import simplify_expr

    H = MatrixSymbol("H", DIM, DIM)
    Q, N = hermitian("Q", DIM), hermitian("N", DIM)
    dQ = MatrixSymbol("dQ", DIM, DIM)
    G = simplify_expr(apply_hermitian(wirtinger_grad_logdet(N + H * Q * H.adjoint(), Q, dQ)), "normalize")
    expected = H.adjoint() * (N + H * Q * H.adjoint()).I * H

    def _np(M, D):
        rng = np.random.default_rng(3)
        A = rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))
        Hn = rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))
        subs = {H: sp.Matrix(Hn), Q: sp.Matrix(A @ A.conj().T + D * np.eye(D)),
                N: sp.Matrix((lambda B: B @ B.conj().T + D * np.eye(D))(
                    rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D)))),
                DIM: D}
        r = M.subs(subs).doit()
        return np.array((r if isinstance(r, sp.MatrixBase) else r.as_explicit()).tolist(), complex)

    assert np.max(np.abs(_np(G, 3) - _np(expected, 3))) < 1e-10


def test_check_gradient_rejects_hermitian():
    # the autograd==2x convention does not apply to Hermitian variables, on ANY
    # quantity type: SymbolicCMI, LogDetQuantity, and CompositeCMI must all raise.
    K, H1, *_ = _mac()
    S1 = hermitian("S1", DIM)
    I = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])
    with pytest.raises(NotImplementedError):
        I.check_gradient(S1, dim=3)
    Q0, Q1 = hermitian("Q0", DIM), hermitian("Q1", DIM)
    with pytest.raises(NotImplementedError):
        gaussian_kl(Q0, Q1).check_gradient(Q1, dim=3)
    with pytest.raises(NotImplementedError):
        composite_cmi([(1, I)]).check_gradient(S1, dim=3)
