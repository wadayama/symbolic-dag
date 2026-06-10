"""Unit tests for the PyTorch-based verification helpers (symbolic_dag.verify).

These exercise the user-facing numerical-checking API. They require the optional
``cmidag`` extra (torch); without it they skip.
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


def _chain():
    A, B = MatrixSymbol("A", DIM, DIM), MatrixSymbol("B", DIM, DIM)
    SX, SY, SZ = (hermitian(s, DIM) for s in ("Sigma_X", "Sigma_Y", "Sigma_Z"))
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0], parents={1: [0], 2: [1]},
        edge_mats={(1, 0): A, (2, 1): B}, root_covs={0: SX}, noise_covs={1: SY, 2: SZ},
    )
    return conditional_mutual_information_from_k(K, A=[0], B=[2], C=[])


def _precoder():
    H, F = MatrixSymbol("H", DIM, DIM), MatrixSymbol("F", DIM, DIM)
    S0, R = hermitian("Sigma_0", DIM), hermitian("R", DIM)
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): H * F, (2, 1): Identity(DIM)},
        root_covs={0: S0, 1: Identity(DIM)}, noise_covs={2: R},
    )
    return conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1]), F


@pytest.mark.parametrize("d", [1, 2, 3])
def test_value_check(d):
    I = _chain()
    report = I.check(d, seed=0, samples=3)
    assert report["passed"], report


@pytest.mark.parametrize("d", [1, 2, 3])
def test_gradient_check_against_autograd(d):
    I, F = _precoder()
    report = I.check_gradient(F, d, seed=1)
    assert report["passed"], report


def test_torch_value_is_differentiable():
    import torch

    I, F = _precoder()
    subs = {}
    from symbolic_dag import random_torch_point

    subs = random_torch_point(I, 2, seed=2, requires_grad=F)
    val = I.torch_value(subs, 2)
    assert val.requires_grad
    val.backward()
    assert subs[F].grad is not None and torch.isfinite(subs[F].grad).all()
