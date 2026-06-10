"""Ergonomic string-name builder for symbolic Gaussian DAGs.

The functional, index-based API (:func:`compute_k_blocks_multiroot`,
:func:`conditional_mutual_information_from_k`) mirrors ``cmi-dag`` exactly and is
the canonical surface. :class:`GaussianDAG` is a thin convenience layer on top:
nodes are named (``"X"``, ``"Y"``, ``"Z"``) for readable symbolic output, and the
builder lowers the names to the prefix-root integer-index dicts and delegates to
the functional core. It adds no new capability --- it only relabels.

Index convention: source nodes (added with :meth:`add_source`) receive the low
indices ``0..K-1`` (the required root prefix), in the order they were added;
non-source nodes receive ``K..M-1`` in add order, which is a topological order
because :meth:`add_node` requires every parent to already exist.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sympy import MatrixExpr

from symbolic_dag.expr import SymbolicCMI
from symbolic_dag.information import conditional_mutual_information_from_k
from symbolic_dag.krecursion import compute_k_blocks_multiroot


class GaussianDAG:
    """A named-node linear Gaussian DAG that compiles to the functional core."""

    def __init__(self) -> None:
        self._order: list[str] = []
        self._is_source: dict[str, bool] = {}
        self._cov: dict[str, MatrixExpr] = {}
        self._parents: dict[str, dict[str, MatrixExpr]] = {}
        self._noise: dict[str, MatrixExpr] = {}

    def _check_new(self, name: str) -> None:
        if name in self._is_source:
            raise ValueError(f"Node {name!r} already exists.")

    def add_source(self, name: str, cov: MatrixExpr) -> None:
        """Add a parentless source node with input covariance ``cov`` (Hermitian PD)."""
        self._check_new(name)
        self._order.append(name)
        self._is_source[name] = True
        self._cov[name] = cov

    def add_node(
        self,
        name: str,
        parents: Mapping[str, MatrixExpr],
        noise: MatrixExpr,
    ) -> None:
        """Add ``X_name = sum_p gain_p X_p + N``, ``Cov(N) = noise``.

        Raises:
            ValueError: if a parent is undefined (topological-order violation) or
                the name already exists.
        """
        self._check_new(name)
        for p in parents:
            if p not in self._is_source:
                raise ValueError(
                    f"Parent {p!r} of {name!r} is not defined yet; add nodes in "
                    "topological order (sources first)."
                )
        self._order.append(name)
        self._is_source[name] = False
        self._parents[name] = dict(parents)
        self._noise[name] = noise

    # ---- lowering to the functional core ----------------------------
    def _index(self) -> tuple[dict[str, int], int]:
        sources = [n for n in self._order if self._is_source[n]]
        nonsrc = [n for n in self._order if not self._is_source[n]]
        names = sources + nonsrc
        return {nm: i for i, nm in enumerate(names)}, len(sources)

    def k_blocks(self) -> dict[tuple[int, int], MatrixExpr]:
        """Compile to symbolic K-blocks via the functional core."""
        idx, num_roots = self._index()
        num_nodes = len(idx)
        parents: dict[int, list[int]] = {}
        edge_mats: dict[tuple[int, int], MatrixExpr] = {}
        root_covs: dict[int, MatrixExpr] = {}
        noise_covs: dict[int, MatrixExpr] = {}
        for nm, j in idx.items():
            if self._is_source[nm]:
                root_covs[j] = self._cov[nm]
            else:
                parents[j] = sorted(idx[p] for p in self._parents[nm])
                noise_covs[j] = self._noise[nm]
                for p, gain in self._parents[nm].items():
                    edge_mats[(j, idx[p])] = gain
        return compute_k_blocks_multiroot(
            num_nodes, list(range(num_roots)), parents,
            edge_mats, root_covs, noise_covs,
        )

    def cmi(
        self,
        A: Sequence[str],
        B: Sequence[str],
        C: Sequence[str] = (),
    ) -> SymbolicCMI:
        """Symbolic ``I(X_A; X_B | X_C)`` for named node sets."""
        idx, _ = self._index()
        K = self.k_blocks()
        return conditional_mutual_information_from_k(
            K, [idx[a] for a in A], [idx[b] for b in B], [idx[c] for c in C]
        )
