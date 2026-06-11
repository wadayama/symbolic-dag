"""Tests for the hand-off / type-setting exporters (symbolic_dag.handoff).

``to_mathematica`` and ``to_markdown`` are pure string exporters (no external
tools). ``render_pdf`` needs ``pdflatex`` and is skipped when it is absent.
"""

from __future__ import annotations

import os
import shutil

import pytest
import sympy as sp
from sympy import Identity, MatrixSymbol

from symbolic_dag import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
    hermitian,
    to_markdown,
    to_mathematica,
)

DIM = sp.Symbol("d", positive=True, integer=True)


def _precoder_cmi():
    H, F = MatrixSymbol("H", DIM, DIM), MatrixSymbol("F", DIM, DIM)
    S1, N = hermitian("Sigma_1", DIM), hermitian("N", DIM)
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): H * F, (2, 1): Identity(DIM)},
        root_covs={0: S1, 1: Identity(DIM)}, noise_covs={2: N},
    )
    return conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1]), F


def test_to_mathematica_gradient():
    I, F = _precoder_cmi()
    wl = I.to_mathematica(F)  # gradient in Wolfram Language
    for token in ("Dot[", "ConjugateTranspose[", "Inverse[", "Subscript[Sigma, 1]"):
        assert token in wl, f"missing {token!r} in {wl}"
    # subscripted names must not leak a raw underscore (Blank in WL)
    assert "Sigma_1" not in wl


def test_to_mathematica_cmi():
    I, _ = _precoder_cmi()
    wl = I.to_mathematica()  # the CMI scalar
    assert wl.startswith("Plus[")
    assert "Log[Det[" in wl


def test_to_markdown_sections():
    I, F = _precoder_cmi()
    md = I.to_markdown(F)
    assert md.startswith("## ")
    assert "**Structural form**" in md
    assert "**Wirtinger gradient**" in md
    assert "**Stationarity (KKT)**" in md
    assert "$$" in md
    # no var -> no gradient section
    assert "Wirtinger gradient" not in I.to_markdown()


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex unavailable")
def test_render_pdf(tmp_path):
    I, F = _precoder_cmi()
    pdf = I.to_pdf(str(tmp_path / "out.pdf"), var=F, png=shutil.which("pdftocairo") is not None)
    assert os.path.exists(pdf) and os.path.getsize(pdf) > 0
    assert os.path.exists(tmp_path / "out.tex")
    # build intermediates cleaned up
    assert not os.path.exists(tmp_path / "out.aux")
    assert not os.path.exists(tmp_path / "out.log")
