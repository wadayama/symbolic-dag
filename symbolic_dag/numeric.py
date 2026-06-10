"""Numerical evaluation and an independent NumPy oracle for verification.

Two roles:

1. Evaluate a symbolic object at a concrete substitution (re-exported from
   :mod:`symbolic_dag.expr`). This substitutes ``sympy`` matrices and runs dense
   linear algebra.

2. Provide a *separate* NumPy implementation of the K-recursion and the CMI
   (:func:`numpy_k_blocks`, :func:`numpy_cmi`). It shares no code with the
   symbolic path, so agreement between ``SymbolicCMI.evaluate`` and
   :func:`numpy_cmi` is a genuine cross-check (and the project's guard against
   fabricated data). For the strongest check, the test suite also compares
   against the actual ``cmi-dag`` library (see ``tests/cmidag_oracle.py``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def numpy_k_blocks(
    num_nodes: int,
    roots: Sequence[int],
    parents: Mapping[int, Sequence[int]],
    edge_mats: Mapping[tuple[int, int], np.ndarray],
    root_covs: Mapping[int, np.ndarray],
    noise_covs: Mapping[int, np.ndarray],
    *,
    cross_root_covs: Mapping[tuple[int, int], np.ndarray] | None = None,
) -> dict[tuple[int, int], np.ndarray]:
    """Independent NumPy multi-root K-recursion (complex). Returns canonical blocks."""
    cross_root_covs = cross_root_covs or {}
    num_roots = len(sorted(roots))
    K: dict[tuple[int, int], np.ndarray] = {}

    def getK(a: int, b: int) -> np.ndarray:
        return K[(a, b)] if a >= b else K[(b, a)].conj().T

    for r in range(num_roots):
        K[(r, r)] = root_covs[r]
    for r in range(num_roots):
        for r2 in range(r):
            s = cross_root_covs.get((r, r2))
            K[(r, r2)] = (
                np.zeros((root_covs[r].shape[0], root_covs[r2].shape[0]), complex)
                if s is None else s
            )
    for j in range(num_roots, num_nodes):
        pa = list(parents[j])
        for k in range(j):
            acc = None
            for i in pa:
                term = edge_mats[(j, i)] @ getK(i, k)
                acc = term if acc is None else acc + term
            K[(j, k)] = acc
        acc = noise_covs[j].astype(complex)
        for i in pa:
            for ip in pa:
                acc = acc + edge_mats[(j, i)] @ getK(i, ip) @ edge_mats[(j, ip)].conj().T
        K[(j, j)] = acc
    return K


def _assemble(K, rows, cols):
    def getK(a, b):
        return K[(a, b)] if a >= b else K[(b, a)].conj().T
    return np.block([[getK(r, c) for c in cols] for r in rows])


def _cond_cov(K, U, C):
    U, C = sorted(U), sorted(C)
    Suu = _assemble(K, U, U)
    if not C:
        return Suu
    Suc, Scc, Scu = _assemble(K, U, C), _assemble(K, C, C), _assemble(K, C, U)
    return Suu - Suc @ np.linalg.solve(Scc, Scu)


def numpy_cmi(
    K: Mapping[tuple[int, int], np.ndarray],
    A: Sequence[int],
    B: Sequence[int],
    C: Sequence[int] = (),
) -> float:
    """Independent NumPy CMI ``I(V_A; V_B | V_C)`` (nats, complex; no 1/2 factor)."""
    A, B, C = sorted(A), sorted(B), sorted(C)
    sld = lambda M: float(np.linalg.slogdet(M)[1])
    return (
        sld(_cond_cov(K, A, C))
        + sld(_cond_cov(K, B, C))
        - sld(_cond_cov(K, sorted(A + B), C))
    )
