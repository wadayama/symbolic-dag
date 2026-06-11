"""Symbolic conditional mutual information ``I(V_A; V_B | V_C)`` from K-blocks.

Mirrors ``cmi_dag.conditional_mutual_information_from_k`` in name, signature and
conventions, but returns a lazy :class:`symbolic_dag.expr.SymbolicCMI` (a set of
signed log-determinant terms plus the cross conditional covariance), not a
scalar. The numeric value agrees with ``cmi-dag`` exactly (both complex, both
without a one-half factor).

The conditional covariance is the Schur complement

    Sigma_{U|C} = Sigma_{U,U} - Sigma_{U,C} Sigma_{C,C}^{-1} Sigma_{C,U},

assembled blockwise from the node-pair K-blocks via :func:`get_K`. For
multi-node information/conditioning sets the blocks are stacked into a
``sympy.BlockMatrix``; for the common singleton case a single block is used
directly. The CMI uses the equivalent three-term log-det form

    I(V_A; V_B | V_C) = log det Sigma_{A|C} + log det Sigma_{B|C}
                        - log det Sigma_{AB|C}.
"""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp
from sympy import BlockMatrix, MatrixExpr

from symbolic_dag.expr import SymbolicCMI
from symbolic_dag.krecursion import get_K


def _assemble(
    K: dict[tuple[int, int], MatrixExpr],
    rows: Sequence[int],
    cols: Sequence[int],
) -> MatrixExpr:
    """Stack node-pair K-blocks into ``Sigma_{rows, cols}`` (a block matrix)."""
    blocks = [[get_K(K, r, c) for c in cols] for r in rows]
    if len(rows) == 1 and len(cols) == 1:
        return blocks[0][0]
    return BlockMatrix(blocks)


def conditional_covariance(
    K: dict[tuple[int, int], MatrixExpr],
    U: Sequence[int],
    C: Sequence[int],
) -> MatrixExpr:
    """Conditional covariance ``Sigma_{U|C}`` via the Schur complement.

    With empty ``C`` this is the marginal ``Sigma_{U,U}``.
    """
    U, C = sorted(U), sorted(C)
    S_UU = _assemble(K, U, U)
    if not C:
        return S_UU
    S_UC = _assemble(K, U, C)
    S_CC = _assemble(K, C, C)
    S_CU = _assemble(K, C, U)
    return S_UU - S_UC * S_CC.I * S_CU


def conditional_covariance_seq(
    K: dict[tuple[int, int], MatrixExpr],
    u: int,
    C: Sequence[int],
) -> MatrixExpr:
    """Conditional covariance ``Sigma_{u|C}`` for a SINGLE node ``u``, block-free.

    Conditions on the nodes of ``C`` one at a time (sequential / Gram--Schmidt
    style), so every inverse is a single node-block and the result is a single
    matrix expression --- never a ``BlockMatrix``. This is what the
    differentiation engine needs: it differentiates single matrices, not opaque
    block inverses. (For just *evaluating* a CMI, the block-assembled
    :func:`conditional_covariance` is fine; this form is for the gradient.)
    """
    C = sorted(C)
    active = [u] + C
    cc = {(a, b): get_K(K, a, b) for a in active for b in active}
    for c in C:
        cci = cc[(c, c)].I
        active = [x for x in active if x != c]
        cc = {
            (a, b): (cc[(a, b)] - cc[(a, c)] * cci * cc[(c, b)]).doit()
            for a in active
            for b in active
        }
    return cc[(u, u)]


def mmse_error_covariance(
    K: dict[tuple[int, int], MatrixExpr],
    target: int,
    observations: Sequence[int],
) -> MatrixExpr:
    """LMMSE estimation-error covariance ``Sigma_{target|observations}``.

    The minimum-mean-square-error covariance of (linearly) estimating the single
    node ``target`` from the ``observations`` nodes. It is exactly
    :func:`conditional_covariance_seq` (kept block-free so it can be
    differentiated): ``tr`` of it is the scalar MMSE, and its Wirtinger gradient
    w.r.t. a precoder/filter follows from
    :func:`symbolic_dag.matderiv.trace_grad`.
    """
    return conditional_covariance_seq(K, target, observations)


def _cross_conditional(
    K: dict[tuple[int, int], MatrixExpr],
    A: Sequence[int],
    B: Sequence[int],
    C: Sequence[int],
) -> MatrixExpr:
    """Cross conditional covariance ``Sigma_{AB|C}`` (off-diagonal block).

    Equal to the zero matrix iff ``A`` and ``B`` are conditionally independent
    given ``C``; this is the object the d-separation proof reduces.
    """
    A, B, C = sorted(A), sorted(B), sorted(C)
    S_AB = _assemble(K, A, B)
    if not C:
        return S_AB
    S_AC = _assemble(K, A, C)
    S_CC = _assemble(K, C, C)
    S_CB = _assemble(K, C, B)
    return S_AB - S_AC * S_CC.I * S_CB


def conditional_mutual_information_from_k(
    K: dict[tuple[int, int], MatrixExpr],
    A: Sequence[int],
    B: Sequence[int],
    C: Sequence[int] = (),
) -> SymbolicCMI:
    """Symbolic ``I(V_A; V_B | V_C)`` from K-blocks (lazy log-det form).

    Args:
        K: Canonical symbolic K-blocks from
            :func:`symbolic_dag.krecursion.compute_k_blocks_multiroot`.
        A: First information set (non-empty node indices).
        B: Second information set (non-empty node indices).
        C: Conditioning set (default empty -> unconditional MI).

    Returns:
        A :class:`symbolic_dag.expr.SymbolicCMI`. Its numeric value (nats) agrees
        with ``cmi_dag.conditional_mutual_information_from_k``.

    Raises:
        ValueError: if ``A`` or ``B`` is empty, or ``A, B, C`` are not pairwise
            disjoint.
    """
    A, B, C = sorted(A), sorted(B), sorted(C)
    if not A or not B:
        raise ValueError("A and B must both be non-empty.")
    alln = A + B + C
    if len(alln) != len(set(alln)):
        raise ValueError(
            f"A, B, C must be pairwise disjoint; got A={A}, B={B}, C={C}."
        )

    Sig_A_C = conditional_covariance(K, A, C)
    Sig_B_C = conditional_covariance(K, B, C)
    Sig_AB_C = conditional_covariance(K, sorted(A + B), C)
    cross = _cross_conditional(K, A, B, C)

    return SymbolicCMI(
        A=tuple(A), B=tuple(B), C=tuple(C),
        logdet_terms=[(1, Sig_A_C), (1, Sig_B_C), (-1, Sig_AB_C)],
        cross=cross,
        # K is retained so the differentiation engine can re-derive the gradient
        # from the single-node two-term form (avoiding block differentiation).
        metadata={"form": "schur_logdet", "K": K},
    )
