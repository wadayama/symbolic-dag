"""Tests for the display layer: "display" strategy, two_term(), scalar-level
log-det simplification (capacity), and det_style rendering.

All transformations here are presentation-level and must be value-preserving;
every test that restructures terms verifies the value against the independent
PyTorch Schur path (``.check``).
"""

from __future__ import annotations

import sympy as sp
from sympy import Identity, MatrixSymbol

from symbolic_dag import (
    GaussianDAG,
    hermitian,
    simplify_expr,
    simplify_logdet_terms,
)

d = sp.Symbol("d", positive=True, integer=True)


def _single_link():
    g = GaussianDAG()
    g.add_source("X", hermitian("Sigma_X", d))
    g.add_node("Z", {"X": MatrixSymbol("H_XZ", d, d)}, hermitian("N_Z", d))
    return g.cmi(["X"], ["Z"])


def _relay():
    g = GaussianDAG()
    g.add_source("X", hermitian("Sigma_X", d))
    g.add_node("Y", {"X": MatrixSymbol("H_XY", d, d)}, hermitian("N_Y", d))
    g.add_node("Z", {"Y": MatrixSymbol("H_YZ", d, d)}, hermitian("N_Z", d))
    return g


def _two_rx():
    """Both A and B multi-node: 2 sources, 2 receivers (block two-term path)."""
    g = GaussianDAG()
    g.add_source("X1", hermitian("Sigma_X1", d))
    g.add_source("X2", hermitian("Sigma_X2", d))
    for rx in ("Y1", "Y2"):
        g.add_node(
            rx,
            {"X1": MatrixSymbol(f"H_X1{rx}", d, d), "X2": MatrixSymbol(f"H_X2{rx}", d, d)},
            hermitian(f"N_{rx}", d),
        )
    return g.cmi(["X1", "X2"], ["Y1", "Y2"])


# ---- the "display" strategy -------------------------------------------------


def test_display_collects_cancelling_pair():
    N, T1, T2 = (MatrixSymbol(s, d, d) for s in ("N", "T1", "T2"))
    e = sp.MatAdd(N, T1, -T2, T2)
    assert simplify_expr(e, "display") == N + T1
    # the default strategy is untouched: it still leaves the pair in place
    assert simplify_expr(e, "normalize") != N + T1


def test_display_distributes_to_expose_cancellation():
    H = MatrixSymbol("H", d, d)
    N, T = MatrixSymbol("N", d, d), MatrixSymbol("T", d, d)
    e = H * (N + T) - H * T
    assert simplify_expr(e, "display") == H * N


def test_display_reduces_relay_residual_to_noise():
    g = _relay()
    cmi = g.cmi(["X"], ["Z"], ["Y"])
    two = cmi.two_term()
    NZ = hermitian("N_Z", d)
    cleaned = [simplify_expr(M, "display") for _, M in two.logdet_terms]
    assert cleaned == [NZ, NZ]  # Sigma_{Z|Y} = Sigma_{Z|XY} = N_Z, hence I = 0


# ---- two_term ---------------------------------------------------------------


def test_two_term_value_matches_single_link():
    two = _single_link().two_term()
    assert len(two.logdet_terms) == 2
    res = two.check(dim=2)
    assert res["passed"], res


def test_two_term_value_matches_block_case():
    two = _two_rx()
    assert len(two.A) > 1 and len(two.B) > 1
    two = two.two_term()
    res = two.check(dim=2)
    assert res["passed"], res


def test_two_term_structural_latex():
    g = _relay()
    two = g.cmi(["X"], ["Z"], ["Y"]).two_term()
    s = two.to_latex()
    assert r"\Sigma_{Z\mid Y}" in s and r"\Sigma_{Z\mid X,Y}" in s
    assert r"\Sigma_{X,Z" not in s  # no joint block in the two-term form


def test_two_term_requires_k_blocks():
    import pytest
    from symbolic_dag.expr import SymbolicCMI

    bare = SymbolicCMI(A=(0,), B=(1,), C=())
    with pytest.raises(ValueError, match="K-blocks"):
        bare.two_term()


# ---- scalar-level simplification (the capacity wart fix) ---------------------


def test_capacity_fires_at_scalar_level():
    cmi = _single_link()
    clean = cmi.two_term().simplify("display")
    terms = simplify_logdet_terms(clean.logdet_terms, "capacity")
    # log det(N + H S H^A) - log det(N) -> the single capacity term
    assert terms is not None and len(terms) == 1
    sign, M = terms[0]
    assert sign == 1
    assert any(isinstance(a, Identity) for a in M.args)
    # value-preserving: same K / A / B / C, new terms, checked independently
    from symbolic_dag.expr import SymbolicCMI

    cap = SymbolicCMI(
        definitions=cmi.definitions, output=cmi.output, metadata=cmi.metadata,
        A=cmi.A, B=cmi.B, C=cmi.C, logdet_terms=terms, cross=cmi.cross,
    )
    res = cap.check(dim=3)
    assert res["passed"], res


def test_capacity_via_to_latex_expanded():
    clean = _single_link().two_term().simplify("display")
    s = clean.to_latex(expand=True, simplify="capacity", det_style="det")
    assert s.count(r"\log") == 1 and r"\mathbb{I}" in s


def test_scalar_level_preserves_normalize_output():
    # the default expanded rendering must be unchanged by the scalar-level path
    cmi = _single_link()
    from symbolic_dag.rewrite import simplify_logdet_terms as slt

    scalar = slt(cmi.logdet_terms, "normalize")
    per_term = [(s, simplify_expr(M, "normalize")) for s, M in cmi.logdet_terms]
    assert scalar == per_term


def test_cancelled_terms_render_zero():
    two = _relay().cmi(["X"], ["Z"], ["Y"]).two_term().simplify("display")
    assert two.to_latex(expand=True, simplify="display").endswith("= 0")


# ---- det_style ----------------------------------------------------------------


def test_det_style_det():
    cmi = _single_link()
    s = cmi.to_latex(det_style="det")
    assert r"\det\left(" in s and r"\left|" not in s
    e = cmi.to_latex(expand=True, det_style="det")
    assert r"\det\left(" in e and r"\left|" not in e


def test_det_style_default_unchanged():
    cmi = _single_link()
    assert r"\left|" in cmi.to_latex()
    assert r"\det" not in cmi.to_latex()


def test_det_style_validated():
    import pytest

    with pytest.raises(ValueError, match="det_style"):
        _single_link().to_latex(det_style="norm")


def test_report_det_style():
    cmi = _single_link()
    H = MatrixSymbol("H_XZ", d, d)
    r = cmi.report(H, det_style="det")
    assert r"\det\left(" in r and r"\frac{\partial I}" in r
