"""Symbolic multi-root K-recursion for linear Gaussian DAGs (complex / block).

The symbolic counterpart of ``cmi_dag.compute_k_blocks_multiroot``: it mirrors
that function's signature and conventions exactly, but the edge gains and
covariances are ``sympy`` matrix expressions (opaque ``MatrixSymbol``s, possibly
of symbolic dimension) instead of ``torch`` tensors, and the returned K-blocks
are matrix *expressions*. Keeping the gains/covariances opaque (block-symbolic,
never expanded to scalar entries) is what makes the closed form independent of
the node dimension; the scalar case is just the ``1 x 1`` specialisation.

Model (0-based indexing, complex; ``^H`` is the conjugate transpose):
    Roots r in {0, ..., K-1}:   V_r ~ CN(0, Sigma_r), mutually independent by
        default; optional cross-covariances Sigma_{r,r'} = E[V_r V_{r'}^H] may
        be supplied for correlated sources.
    Non-roots j:  V_j = sum_{i in Pa(j)} A_{ji} V_i + Z_j,  Z_j ~ CN(0, Sigma_j).

Canonical storage: ``K`` holds only ``K[(j, k)]`` for ``j >= k``; access ``K_ab``
with ``a < b`` through :func:`get_K`, which applies the Hermitian flip
``K_ab = K_ba^H`` (``sympy.Adjoint``).
"""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp
from sympy import Adjoint, MatrixExpr, ZeroMatrix

from symbolic_dag.assumptions import apply_hermitian


def hermitianize(M: MatrixExpr) -> MatrixExpr:
    """Canonicalise a block expected to be Hermitian.

    The symbolic analogue of ``cmi_dag``'s ``(A + A^H)/2`` drift correction: there
    is no floating-point drift here, so this simply imposes the covariance
    symmetry assumption ``Adjoint(Sigma) -> Sigma`` (see
    :func:`symbolic_dag.assumptions.apply_hermitian`) to keep self-blocks in a
    clean Hermitian form.
    """
    return apply_hermitian(M)


def get_K(
    K: dict[tuple[int, int], MatrixExpr], a: int, b: int
) -> MatrixExpr:
    """Return ``K_ab``, applying the Hermitian flip ``K_ab = K_ba^H`` when ``a < b``.

    ``K`` is assumed to store only canonical keys ``(j, k)`` with ``j >= k``.
    """
    if a >= b:
        return K[(a, b)]
    return Adjoint(K[(b, a)])


def _validate(
    num_nodes: int,
    roots: Sequence[int],
    parents: dict[int, list[int]],
    noise_covs: dict[int, MatrixExpr],
) -> int:
    roots = sorted(roots)
    num_roots = len(roots)
    if roots != list(range(num_roots)):
        raise ValueError(
            f"roots must be the prefix {{0, ..., K-1}} in topological order, "
            f"got {roots}."
        )
    if num_roots >= num_nodes:
        raise ValueError(
            f"num_roots ({num_roots}) must be strictly less than num_nodes "
            f"({num_nodes}): the DAG must contain at least one non-root node."
        )
    for j in range(num_roots, num_nodes):
        if j not in parents or len(parents[j]) == 0:
            raise ValueError(f"Non-root node {j} has no parents.")
        for i in parents[j]:
            if not (0 <= i < j):
                raise ValueError(
                    f"Parent {i} of node {j} violates topological order "
                    f"(0 <= i < j)."
                )
        if j not in noise_covs:
            raise ValueError(
                f"noise_covs is missing the entry for non-root node {j}."
            )
    return num_roots


def compute_k_blocks_multiroot(
    num_nodes: int,
    roots: Sequence[int],
    parents: dict[int, list[int]],
    edge_mats: dict[tuple[int, int], MatrixExpr],
    root_covs: dict[int, MatrixExpr],
    noise_covs: dict[int, MatrixExpr],
    *,
    cross_root_covs: dict[tuple[int, int], MatrixExpr] | None = None,
    symmetrize_self_blocks: bool = True,
) -> dict[tuple[int, int], MatrixExpr]:
    """Compute all canonical symbolic K-blocks ``K[(j, k)]`` (j >= k).

    Symbolic, complex analogue of ``cmi_dag.compute_k_blocks_multiroot`` with the
    same argument names and conventions. The recursion is

        K_{rr}  = Sigma_r,   K_{r r'} = Sigma_{r, r'} (cross; default 0),
        K_{jk}  = sum_{i in Pa(j)} A_{ji} K_{ik}                       (k < j),
        K_{jj}  = sum_{i, i' in Pa(j)} A_{ji} K_{i i'} A_{ji'}^H + Sigma_j,

    with ``^H = sympy.Adjoint``.

    Args:
        num_nodes: Total number of nodes ``M`` (indices ``0..M-1``).
        roots: Root indices; must be exactly the prefix ``{0, ..., K-1}`` in
            topological order, with ``K < num_nodes``.
        parents: ``parents[j]`` = list of parent indices for non-root ``j``
            (each ``i < j``). Roots need not appear.
        edge_mats: ``edge_mats[(j, i)] = A_{ji}`` (shape ``d_j x d_i``), a
            ``sympy`` matrix expression.
        root_covs: ``root_covs[r] = Sigma_r`` for each root ``r`` (Hermitian PD;
            create with :func:`symbolic_dag.assumptions.hermitian`).
        noise_covs: ``noise_covs[j] = Sigma_j`` for every non-root ``j``.
        cross_root_covs: Optional ``{(r, r'): Sigma_{r, r'}}`` for correlated
            roots, keys ``r > r'``. Missing keys default to the zero matrix
            (independent roots).
        symmetrize_self_blocks: If True, canonicalise each self-block via
            :func:`hermitianize` (imposes ``Adjoint(Sigma) -> Sigma``).

    Returns:
        Dict ``K`` with keys ``(j, k)`` for ``0 <= k <= j < num_nodes``; each
        value is a ``sympy`` matrix expression. Use :func:`get_K` for the
        Hermitian flip on ``a < b``.

    Raises:
        ValueError: on the prefix-roots / topological-order / missing-noise /
            missing-root-cov contract.
    """
    num_roots = _validate(num_nodes, roots, parents, noise_covs)
    for r in range(num_roots):
        if r not in root_covs:
            raise ValueError(f"root_covs is missing the entry for root {r}.")
    cross_root_covs = cross_root_covs or {}

    K: dict[tuple[int, int], MatrixExpr] = {}

    # Base case: root self- and cross-covariances.
    for r in range(num_roots):
        cov = root_covs[r]
        K[(r, r)] = hermitianize(cov) if symmetrize_self_blocks else cov
    for r in range(num_roots):
        for r2 in range(r):  # r > r2
            sigma = cross_root_covs.get((r, r2))
            if sigma is None:
                d_r = root_covs[r].shape[0]
                d_r2 = root_covs[r2].shape[0]
                K[(r, r2)] = ZeroMatrix(d_r, d_r2)
            else:
                K[(r, r2)] = sigma

    # Non-root nodes in topological order.
    for j in range(num_roots, num_nodes):
        pa = parents[j]
        # (1) cross blocks K_{jk}, k = 0..j-1
        for k in range(j):
            acc: MatrixExpr | None = None
            for i in pa:
                term = edge_mats[(j, i)] * get_K(K, i, k)
                acc = term if acc is None else acc + term
            assert acc is not None  # pa is non-empty
            K[(j, k)] = acc.doit()
        # (2) self block K_{jj} = sum A_ji K_ii' A_ji'^H + Sigma_j
        acc = noise_covs[j]
        for i in pa:
            Aji = edge_mats[(j, i)]
            for ip in pa:
                Ajip = edge_mats[(j, ip)]
                acc = acc + Aji * get_K(K, i, ip) * Adjoint(Ajip)
        K[(j, j)] = hermitianize(acc) if symmetrize_self_blocks else acc.doit()

    return K
