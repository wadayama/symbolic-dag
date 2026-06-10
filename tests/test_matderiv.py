"""Unit tests for symbolic_dag.matderiv — Wirtinger gradient via finite difference."""

from __future__ import annotations

import numpy as np
import sympy as sp
from sympy import Identity, MatrixSymbol

from symbolic_dag.assumptions import hermitian
from symbolic_dag.information import conditional_mutual_information_from_k
from symbolic_dag.krecursion import compute_k_blocks_multiroot
from symbolic_dag.matderiv import wirtinger_grad_logdet

DIM = sp.Symbol("n", positive=True, integer=True)


def _rng(s):
    return np.random.default_rng(s)


def _rc(d, rng):
    return rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))


def _hpd(d, rng):
    A = _rc(d, rng)
    return A @ A.conj().T + d * np.eye(d)


def _to_np(M, sub):
    r = M.subs(sub).doit()
    r = r if isinstance(r, sp.MatrixBase) else r.as_explicit()
    return np.array(r.tolist(), complex)


def _precoder_cmi():
    """roots X0 (Sigma0), X1 (I); Y = (H F) X0 + X1 + N, Cov(N)=R. I(X0;Y|X1)."""
    H, F = MatrixSymbol("H", DIM, DIM), MatrixSymbol("F", DIM, DIM)
    S0, R = hermitian("Sigma_0", DIM), hermitian("R", DIM)
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): H * F, (2, 1): Identity(DIM)},
        root_covs={0: S0, 1: Identity(DIM)}, noise_covs={2: R},
    )
    I = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])
    return I, {"H": H, "F": F, "Sigma_0": S0, "R": R}


def test_precoder_gradient_finite_difference():
    I, syms = _precoder_cmi()
    F = syms["F"]
    G = I.wirtinger_grad(F)  # closed-form dI/dF^*

    for d in (1, 2, 3):
        rng = _rng(7 + d)
        Hn, Fn, S0n, Rn = _rc(d, rng), _rc(d, rng), _hpd(d, rng), _hpd(d, rng)
        E = _rc(d, rng)
        base = {syms["H"]: sp.Matrix(Hn), syms["Sigma_0"]: sp.Matrix(S0n),
                syms["R"]: sp.Matrix(Rn), DIM: d}

        def f(Fmat):
            sub = {**base, F: sp.Matrix(Fmat)}
            return I.evaluate(sub)

        eps = 1e-6
        fd = (f(Fn + eps * E) - f(Fn - eps * E)) / (2 * eps)
        Gn = _to_np(G, {**base, F: sp.Matrix(Fn)})
        directional = 2.0 * np.real(np.trace(Gn @ E.conj().T))
        assert abs(fd - directional) < 1e-5, f"d={d}: fd={fd:.6f} dir={directional:.6f}"


def test_stationarity_returns_eq():
    I, syms = _precoder_cmi()
    eq = I.stationarity(syms["F"])
    assert isinstance(eq, sp.Equality)


def test_grad_logdet_three_objectives_fd():
    """The engine derives gradients of several log-det objectives (not hand-coded)."""
    from sympy import Adjoint

    n = DIM
    F = MatrixSymbol("F", n, n)
    S = hermitian("S", n)
    H = MatrixSymbol("H", n, n)
    dF = MatrixSymbol("dF", n, n)
    objs = {
        "logdet(I+FSF^H)": Identity(n) + F * S * Adjoint(F),
        "logdet(I+HFF^H H^H)": Identity(n) + H * F * Adjoint(F) * Adjoint(H),
        "logdet(I+F^H S F)": Identity(n) + Adjoint(F) * S * F,
    }
    d, rng = 3, _rng(99)
    Sn, Hn, Fn, E = _hpd(d, rng), _rc(d, rng), _rc(d, rng), _rc(d, rng)
    base = {S: sp.Matrix(Sn), H: sp.Matrix(Hn), n: d}
    for name, M in objs.items():
        G = wirtinger_grad_logdet(M, F, dF)
        Gn = _to_np(G, {**base, F: sp.Matrix(Fn)})

        def f(Fmat):
            return float(np.linalg.slogdet(_to_np(M, {**base, F: sp.Matrix(Fmat)}))[1])

        fd = (f(Fn + 1e-6 * E) - f(Fn - 1e-6 * E)) / 2e-6
        directional = 2.0 * np.real(np.trace(Gn @ E.conj().T))
        assert abs(fd - directional) < 1e-5, f"{name}: {fd} vs {directional}"
