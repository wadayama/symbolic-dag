"""Unit tests for symbolic_dag.builder — builder lowers to the functional core."""

from __future__ import annotations

import numpy as np
import sympy as sp
from sympy import MatrixSymbol

from symbolic_dag.assumptions import hermitian
from symbolic_dag.builder import GaussianDAG
from symbolic_dag.information import conditional_mutual_information_from_k
from symbolic_dag.krecursion import compute_k_blocks_multiroot

DIM = sp.Symbol("n", positive=True, integer=True)


def test_builder_matches_functional_core():
    A, B = MatrixSymbol("A", DIM, DIM), MatrixSymbol("B", DIM, DIM)
    SX, SY, SZ = (hermitian(s, DIM) for s in ("Sigma_X", "Sigma_Y", "Sigma_Z"))

    # functional (index-based)
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0], parents={1: [0], 2: [1]},
        edge_mats={(1, 0): A, (2, 1): B}, root_covs={0: SX}, noise_covs={1: SY, 2: SZ},
    )
    I_func = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])

    # builder (named)
    G = GaussianDAG()
    G.add_source("X", cov=SX)
    G.add_node("Y", parents={"X": A}, noise=SY)
    G.add_node("Z", parents={"Y": B}, noise=SZ)
    I_build = G.cmi(A=["X"], B=["Z"], C=["Y"])

    # identical log-det terms
    assert [t[1] for t in I_func.logdet_terms] == [t[1] for t in I_build.logdet_terms]


def test_builder_sources_get_prefix_indices_even_if_interleaved():
    # add a non-source dependency only after its source; sources still prefix
    A = MatrixSymbol("A", DIM, DIM)
    SX, SY = hermitian("Sigma_X", DIM), hermitian("Sigma_Y", DIM)
    G = GaussianDAG()
    G.add_source("X", cov=SX)
    G.add_node("Y", parents={"X": A}, noise=SY)
    idx, num_roots = G._index()
    assert idx["X"] == 0 and idx["Y"] == 1 and num_roots == 1


def test_builder_value_matches_numpy():
    from symbolic_dag.numeric import numpy_cmi, numpy_k_blocks

    A, B = MatrixSymbol("A", DIM, DIM), MatrixSymbol("B", DIM, DIM)
    SX, SY, SZ = (hermitian(s, DIM) for s in ("Sigma_X", "Sigma_Y", "Sigma_Z"))
    G = GaussianDAG()
    G.add_source("X", cov=SX)
    G.add_node("Y", parents={"X": A}, noise=SY)
    G.add_node("Z", parents={"Y": B}, noise=SZ)
    I = G.cmi(A=["X"], B=["Z"], C=[])

    d, rng = 2, np.random.default_rng(3)
    def rc():
        return rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    def hpd():
        m = rc()
        return m @ m.conj().T + d * np.eye(d)
    An, Bn, SXn, SYn, SZn = rc(), rc(), hpd(), hpd(), hpd()
    subs = {A: sp.Matrix(An), B: sp.Matrix(Bn), SX: sp.Matrix(SXn),
            SY: sp.Matrix(SYn), SZ: sp.Matrix(SZn), DIM: d}
    Knp = numpy_k_blocks(3, [0], {1: [0], 2: [1]},
                         {(1, 0): An, (2, 1): Bn}, {0: SXn}, {1: SYn, 2: SZn})
    ok, err = I.numeric_check(subs, numpy_cmi(Knp, [0], [2], []), atol=1e-9)
    assert ok, f"err={err:.2e}"
