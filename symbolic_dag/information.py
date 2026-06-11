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

from symbolic_dag.expr import LogDetQuantity, SymbolicCMI
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


def lmmse_estimator(
    K: dict[tuple[int, int], MatrixExpr],
    target: int | Sequence[int],
    observations: Sequence[int],
) -> MatrixExpr:
    """Closed-form LMMSE / Wiener estimator ``W`` with ``X_hat = W · V_observations``.

    This is the solution of the (linear) MMSE stationarity
    ``d/dW* tr E(W) = 0`` --- the Wiener filter

        W = Sigma_{target, obs} · Sigma_{obs, obs}^{-1},

    whose residual error covariance is :func:`mmse_error_covariance`. ``target``
    may be a single node or a node set. (Equivalently, ``W`` is what
    :func:`symbolic_dag.solve.solve_stationary` returns for the gradient of the
    estimator MSE; this is the direct closed form.)
    """
    tgt = [target] if isinstance(target, int) else sorted(target)
    obs = sorted(observations)
    return (_assemble(K, tgt, obs) * _assemble(K, obs, obs).I).doit()


def conditional_entropy_from_k(
    K: dict[tuple[int, int], MatrixExpr],
    A: Sequence[int],
    C: Sequence[int] = (),
) -> LogDetQuantity:
    """Differential entropy ``h(V_A | V_C)`` (nats, complex circular Gaussian).

        h(V_A | V_C) = log det( (pi e) Sigma_{A|C} )
                     = sum_i log det Sigma_{a_i | a_{<i}, C}  +  n_A log(pi e),

    built in the chained (block-free) form so each term is a single matrix
    expression and the Wirtinger engine applies. The additive constant
    ``n_A log(pi e)`` (``n_A`` = total dimension of ``A``, possibly symbolic) is
    carried explicitly; it cancels in entropy differences.
    """
    A, C = sorted(A), sorted(C)
    if not A:
        raise ValueError("A must be non-empty.")
    if set(A) & set(C):
        raise ValueError(f"A and C must be disjoint; got A={A}, C={C}.")
    terms: list[tuple[sp.Expr, MatrixExpr]] = []
    n_A: sp.Expr = sp.Integer(0)
    for i, a in enumerate(A):
        terms.append((sp.Integer(1), conditional_covariance_seq(K, a, A[:i] + C)))
        n_A = n_A + K[(a, a)].shape[0]
    return LogDetQuantity(
        logdet_terms=terms,
        constant=sp.log(sp.pi * sp.E) * n_A,
        metadata={"form": "conditional_entropy", "K": K, "A": tuple(A), "C": tuple(C)},
    )


def total_correlation_from_k(
    K: dict[tuple[int, int], MatrixExpr],
    nodes: Sequence[int],
    C: Sequence[int] = (),
) -> LogDetQuantity:
    """Total correlation (multi-information) ``TC(V_1; ...; V_n | C)`` (nats).

        TC = sum_i h(V_i | C) - h(V_1..n | C)
           = sum_{i >= 2} [ log det Sigma_{v_i|C} - log det Sigma_{v_i | v_{<i}, C} ],

    (the chained form; the ``pi e`` constants and the ``i = 1`` terms cancel).
    Zero iff the nodes are conditionally independent given ``C``. For two nodes
    this is exactly ``I(V_1; V_2 | C)``.
    """
    nodes, C = sorted(nodes), sorted(C)
    if len(nodes) < 2:
        raise ValueError("total correlation needs at least two nodes.")
    if set(nodes) & set(C):
        raise ValueError(f"nodes and C must be disjoint; got {nodes}, C={C}.")
    terms: list[tuple[sp.Expr, MatrixExpr]] = []
    for i, v in enumerate(nodes):
        if i == 0:
            continue  # the i = 1 marginal and chain terms cancel
        terms.append((sp.Integer(1), conditional_covariance_seq(K, v, C)))
        terms.append((sp.Integer(-1), conditional_covariance_seq(K, v, nodes[:i] + C)))
    return LogDetQuantity(
        logdet_terms=terms,
        metadata={"form": "total_correlation", "K": K,
                  "nodes": tuple(nodes), "C": tuple(C)},
    )


def gaussian_kl(Sigma0: MatrixExpr, Sigma1: MatrixExpr) -> LogDetQuantity:
    """KL divergence ``D( CN(0, Sigma0) || CN(0, Sigma1) )`` (nats, complex circular).

        D = tr(Sigma1^{-1} Sigma0) - n + log det Sigma1 - log det Sigma0,

    with no one-half factor (complex circular convention, matching the library).
    ``Sigma0``, ``Sigma1`` are any square Hermitian-PD matrix expressions of the
    same (possibly symbolic) dimension ``n`` --- e.g. two conditional covariances
    of the same DAG under different parameters.
    """
    n = Sigma0.shape[0]
    if Sigma1.shape[0] != n or Sigma0.shape[1] != n or Sigma1.shape[1] != n:
        raise ValueError(
            f"Sigma0 and Sigma1 must be square with equal dimension; got "
            f"{Sigma0.shape} and {Sigma1.shape}."
        )
    return LogDetQuantity(
        logdet_terms=[(sp.Integer(1), Sigma1), (sp.Integer(-1), Sigma0)],
        trace_terms=[(sp.Integer(1), sp.Inverse(Sigma1) * Sigma0)],
        constant=-sp.sympify(n),
        metadata={"form": "gaussian_kl"},
    )


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
