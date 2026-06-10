"""Structural assumptions for symbolic Gaussian-DAG covariances.

The symbolic engines need to know facts that ``sympy`` does not infer on its
own --- above all that covariance matrices are **Hermitian** (so that
``Sigma^H = Sigma``) and **positive definite** (so that every ``log det`` and
matrix inverse that appears is well defined). Without the Hermitian assumption,
``sympy`` keeps ``Adjoint(Sigma)`` terms and cannot, for example, recognise that
a cross conditional covariance vanishes (the symbolic proof of conditional
independence); see the smoke-test note (E7).

Rather than carry a global mutable registry, the Hermitian/PD tag is attached to
the symbol itself: a covariance is created with :func:`hermitian`, which returns
a :class:`HermitianMatrix` (a ``sympy.MatrixSymbol`` subclass). The simplify and
differentiation engines detect these by type and apply the rewrite
``Adjoint(Sigma) -> Sigma``. This keeps the assumption local to the symbols that
carry it and free of process-global state.

Conventions
-----------
- ``^H`` (conjugate transpose) is ``sympy.Adjoint``; the library is complex-first
  (Wirtinger calculus), so Hermitian --- not real-symmetric --- is the relevant
  assumption.
- Positive definiteness is a *validity condition*, not a rewrite: it guarantees
  the determinants/inverses exist. Numerical checks must supply Hermitian PD
  matrices for these symbols.
"""

from __future__ import annotations

import sympy as sp
from sympy import Adjoint, MatrixSymbol


class HermitianMatrix(MatrixSymbol):
    """A square ``MatrixSymbol`` tagged Hermitian positive definite.

    Carries no extra data beyond the tag; it is an ordinary ``MatrixSymbol`` for
    every algebraic purpose, but the engines recognise it (via ``isinstance``)
    as a covariance for which ``Adjoint(self) == self`` may be applied.
    """

    is_hermitian_cov = True


def hermitian(name: str, d: sp.Expr | int) -> HermitianMatrix:
    """Create a ``d x d`` Hermitian positive-definite covariance symbol.

    Args:
        name: Symbol name (used in printed output, e.g. ``"Sigma_X"``).
        d: Dimension. May be a concrete ``int`` or a ``sympy`` symbol (for a
            dimension-independent expression).

    Returns:
        A :class:`HermitianMatrix` of shape ``(d, d)``.
    """
    return HermitianMatrix(name, d, d)


def is_hermitian(M: sp.Basic) -> bool:
    """True if ``M`` is a covariance symbol declared Hermitian via :func:`hermitian`."""
    return isinstance(M, HermitianMatrix)


def hermitian_symbols(expr: sp.Basic) -> set[HermitianMatrix]:
    """Collect every Hermitian covariance symbol occurring in ``expr``."""
    return {a for a in sp.preorder_traversal(expr) if isinstance(a, HermitianMatrix)}


def apply_hermitian(expr: sp.Basic) -> sp.Basic:
    """Impose Hermitian symmetry ``Adjoint(Sigma) -> Sigma`` for tagged covariances.

    This is the structural assumption the matrix layer relies on; applying it is
    what turns ``sympy``'s "cannot tell" into provable identities (e.g. a cross
    conditional covariance collapsing to the zero matrix). Returns ``expr`` with
    the substitution applied and re-evaluated.
    """
    subs = {Adjoint(S): S for S in hermitian_symbols(expr)}
    return expr.doit().subs(subs).doit() if subs else expr.doit()
