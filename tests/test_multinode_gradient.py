"""Both-multi-node CMI Wirtinger gradient (chain-rule path) vs PyTorch autograd.

When both ``A`` and ``B`` are multi-node, ``wirtinger_grad_cmi`` uses the chain
rule of mutual information to expand the gradient into single-node terms. These
tests confirm, via ``check_gradient`` (which differentiates the same CMI with
PyTorch autograd, ``autograd == 2 * symbolic``), that the result is correct.
Requires torch (a core dependency).
"""

from __future__ import annotations

import importlib.util

import pytest
import sympy as sp
from sympy import Identity, MatrixSymbol

from symbolic_dag import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
    hermitian,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch unavailable (it is a core dependency; run `uv sync`)",
)

DIM = sp.Symbol("d", positive=True, integer=True)


def _cmi_both_multinode(with_conditioning: bool):
    """Both ``A`` and ``B`` multi-node; ``F`` sits in the first sink's edge.

    Roots are a prefix ``{0,..,K-1}`` (engine requirement). Without conditioning:
    roots ``[0,1]`` (= ``A``), sinks ``[2,3]`` (= ``B``). With conditioning: an
    extra root ``2`` (= ``C``) also feeds the sinks, which become ``[3,4]``.
    """
    H, F = MatrixSymbol("H", DIM, DIM), MatrixSymbol("F", DIM, DIM)
    S0, S1 = hermitian("Sigma_0", DIM), hermitian("Sigma_1", DIM)
    Ra, Rb = hermitian("Ra", DIM), hermitian("Rb", DIM)
    if with_conditioning:
        S2 = hermitian("Sigma_2", DIM)
        roots, root_covs = [0, 1, 2], {0: S0, 1: S1, 2: S2}
        sinks, A, B, C = [3, 4], [0, 1], [3, 4], [2]
    else:
        roots, root_covs = [0, 1], {0: S0, 1: S1}
        sinks, A, B, C = [2, 3], [0, 1], [2, 3], []
    s0, s1 = sinks
    edges = {(s0, 0): H * F, (s1, 0): Identity(DIM)}
    for i in roots:
        if i != 0:
            edges[(s0, i)] = Identity(DIM)
            edges[(s1, i)] = Identity(DIM)
    K = compute_k_blocks_multiroot(
        num_nodes=len(roots) + 2, roots=roots,
        parents={s0: roots, s1: roots},
        edge_mats=edges, root_covs=root_covs, noise_covs={s0: Ra, s1: Rb},
    )
    return conditional_mutual_information_from_k(K, A=A, B=B, C=C), F


@pytest.mark.parametrize("with_conditioning", [False, True], ids=["C=[]", "C=[4]"])
@pytest.mark.parametrize("d", [2, 3])
def test_both_multinode_gradient_matches_autograd(with_conditioning, d):
    I, F = _cmi_both_multinode(with_conditioning)
    report = I.check_gradient(F, d, seed=4)
    assert report["passed"], report
