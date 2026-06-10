"""Unit tests for symbolic_dag.krecursion."""

from __future__ import annotations

import sympy as sp
import pytest
from sympy import Adjoint, MatrixSymbol

from symbolic_dag.assumptions import hermitian
from symbolic_dag.krecursion import compute_k_blocks_multiroot, get_K


def _n():
    return sp.Symbol("n", positive=True, integer=True)


def test_chain_blocks():
    n = _n()
    A = MatrixSymbol("A", n, n)
    B = MatrixSymbol("B", n, n)
    SX, SY, SZ = (hermitian(s, n) for s in ("Sigma_X", "Sigma_Y", "Sigma_Z"))
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0], parents={1: [0], 2: [1]},
        edge_mats={(1, 0): A, (2, 1): B},
        root_covs={0: SX}, noise_covs={1: SY, 2: SZ},
    )
    assert K[(0, 0)] == SX
    assert K[(1, 0)].doit() == (A * SX).doit()              # K_YX = A Sigma_X
    assert K[(1, 1)].doit() == (A * SX * Adjoint(A) + SY).doit()
    # K_ZX = B A Sigma_X
    assert K[(2, 0)].doit() == (B * A * SX).doit()


def test_get_K_hermitian_flip():
    n = _n()
    A = MatrixSymbol("A", n, n)
    SX, SY = hermitian("Sigma_X", n), hermitian("Sigma_Y", n)
    K = compute_k_blocks_multiroot(
        num_nodes=2, roots=[0], parents={1: [0]},
        edge_mats={(1, 0): A}, root_covs={0: SX}, noise_covs={1: SY},
    )
    # upper block via flip = Adjoint of stored lower block
    assert get_K(K, 0, 1) == Adjoint(K[(1, 0)])


def test_independent_roots_zero_cross():
    n = _n()
    A = MatrixSymbol("A", n, n)
    B = MatrixSymbol("B", n, n)
    SX, SY, SZ = (hermitian(s, n) for s in ("Sigma_X", "Sigma_Y", "Sigma_Z"))
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): A, (2, 1): B},
        root_covs={0: SX, 1: SY}, noise_covs={2: SZ},
    )
    assert K[(1, 0)].is_ZeroMatrix  # independent sources -> zero cross-cov


def test_rejects_non_prefix_roots():
    n = _n()
    A = MatrixSymbol("A", n, n)
    SX, SY = hermitian("Sigma_X", n), hermitian("Sigma_Y", n)
    with pytest.raises(ValueError, match="prefix"):
        compute_k_blocks_multiroot(
            num_nodes=3, roots=[1], parents={2: [1]},
            edge_mats={(2, 1): A}, root_covs={1: SX}, noise_covs={2: SY},
        )


def test_rejects_bad_topological_order():
    n = _n()
    A = MatrixSymbol("A", n, n)
    SX, SY = hermitian("Sigma_X", n), hermitian("Sigma_Y", n)
    with pytest.raises(ValueError, match="topological order"):
        compute_k_blocks_multiroot(
            num_nodes=2, roots=[0], parents={1: [5]},  # parent 5 >= 1
            edge_mats={(1, 5): A}, root_covs={0: SX}, noise_covs={1: SY},
        )
