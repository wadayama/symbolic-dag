"""Unit tests for the LaTeX hand-off (symbolic_dag.latex)."""

from __future__ import annotations

import sympy as sp
from sympy import Identity, MatrixSymbol

from symbolic_dag import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
    hermitian,
)

DIM = sp.Symbol("d", positive=True, integer=True)


def _precoder():
    H, F = MatrixSymbol("H", DIM, DIM), MatrixSymbol("F", DIM, DIM)
    S0, R = hermitian("Sigma_0", DIM), hermitian("R", DIM)
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): H * F, (2, 1): Identity(DIM)},
        root_covs={0: S0, 1: Identity(DIM)}, noise_covs={2: R},
    )
    return conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1]), F


def test_cmi_to_latex_structural():
    I, _F = _precoder()
    s = I.to_latex()
    assert r"\Sigma_{0\mid 1}" in s and r"\Sigma_{2\mid 1}" in s
    assert r"\Sigma_{0,2\mid 1}" in s  # the joint term
    assert s.startswith("I(V_{0}; V_{2} \\mid V_{1})")


def test_cmi_to_latex_expand_runs():
    I, _F = _precoder()
    s = I.to_latex(expand=True)
    assert r"\log" in s and r"\Sigma_{0}" in s  # explicit matrix expressions


def test_report_has_cmi_gradient_kkt():
    I, F = _precoder()
    rep = I.report(F)
    assert rep.startswith(r"\begin{align*}") and rep.endswith(r"\end{align*}")
    assert r"\frac{\partial I}{\partial F^{*}}" in rep  # gradient line
    assert "(KKT)" in rep                                # stationarity line
    # the gradient body should be the derived closed form
    assert "H^{\\dagger}" in rep


def test_named_nodes_in_latex():
    import sympy as sp

    from symbolic_dag import GaussianDAG, hermitian

    d = sp.Symbol("d", positive=True, integer=True)
    A, B = sp.MatrixSymbol("A", d, d), sp.MatrixSymbol("B", d, d)
    SX, SY, SZ = (hermitian(s, d) for s in ("Sigma_X", "Sigma_Y", "Sigma_Z"))
    G = GaussianDAG()
    G.add_source("X", cov=SX)
    G.add_node("Y", parents={"X": A}, noise=SY)
    G.add_node("Z", parents={"Y": B}, noise=SZ)
    s = G.cmi(A=["X"], B=["Z"], C=["Y"]).to_latex()
    assert s.startswith(r"I(V_{X}; V_{Z} \mid V_{Y})")
    assert r"\Sigma_{X\mid Y}" in s and r"\Sigma_{X,Z\mid Y}" in s
