"""A strategic rewrite engine for matrix log-det / inverse expressions.

``sympy.simplify`` cannot simplify the matrix layer: it does not know that
covariances are Hermitian, and it does not apply the block identities (Schur
complement, matrix-determinant lemma, Woodbury) that collapse a CMI log-det.
This module supplies that as a small fixpoint engine with an explicit
*strategy*, which the smoke tests showed is necessary --- the rules are not
confluent as a flat bag (Woodbury can pre-empt the inverse-cancellation that a
d-separation proof needs), so structural normalization must run before low-rank
expansion.

A :data:`Rule` is ``expr -> expr | None`` (``None`` = does not apply). The
engine rebuilds each node from its rewritten arguments and repeats to a fixpoint
or an iteration cap. :func:`run_phases` applies ordered rule groups, each to a
fixpoint, before the next.
"""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp
from sympy import Adjoint, Identity, Inverse, MatAdd, MatMul, ZeroMatrix

from symbolic_dag.assumptions import HermitianMatrix

Rule = Callable[[sp.Basic], "sp.Basic | None"]


# ----------------------------------------------------------------------
# structural (normalization) rules
# ----------------------------------------------------------------------
def r_adjoint_distribute(e):
    """Push ``Adjoint`` inward: (XY)^H->Y^H X^H, (X+Y)^H->X^H+Y^H, (M^-1)^H->(M^H)^-1, (X^H)^H->X."""
    if isinstance(e, Adjoint):
        a = e.arg
        if isinstance(a, MatMul):
            return MatMul(*[Adjoint(x) for x in reversed(a.args)])
        if isinstance(a, MatAdd):
            return MatAdd(*[Adjoint(x) for x in a.args])
        if isinstance(a, Inverse):
            return Inverse(Adjoint(a.arg))
        if isinstance(a, Adjoint):
            return a.arg
        if isinstance(a, (Identity, ZeroMatrix)):
            return a
    return None


def r_symmetry(e):
    """``Adjoint(Sigma) -> Sigma`` for Hermitian covariance symbols (the structural assumption)."""
    if isinstance(e, Adjoint) and isinstance(e.arg, HermitianMatrix):
        return e.arg
    return None


def r_inverse_cancel(e):
    """Cancel an adjacent ``M^-1 M`` or ``M M^-1`` in a product to the identity."""
    if isinstance(e, MatMul):
        coeff, mats = e.as_coeff_mmul()
        args = list(mats.args)
        for i in range(len(args) - 1):
            a, b = args[i], args[i + 1]
            if (isinstance(b, Inverse) and b.arg == a) or (
                isinstance(a, Inverse) and a.arg == b
            ):
                rest = args[:i] + args[i + 2:]
                return coeff * (MatMul(*rest) if rest else Identity(a.shape[0]))
    return None


def r_matadd_combine(e):
    """Combine a sum of additive inverses to the zero matrix (``X + (-X) -> 0``)."""
    if isinstance(e, MatAdd):
        z = e.doit()
        if z == 0 or getattr(z, "is_ZeroMatrix", False):
            return ZeroMatrix(*e.shape)
    return None


# ----------------------------------------------------------------------
# expansion rules (low-rank / capacity form)
# ----------------------------------------------------------------------
def _split_lowrank(M):
    """Split a 2-term ``MatAdd`` ``A + (U C V)`` into ``(base A, [factors])``; base a leaf."""
    if not isinstance(M, MatAdd) or len(M.args) != 2:
        return None
    from sympy import MatrixSymbol

    for base, upd in (M.args, M.args[::-1]):
        if isinstance(base, MatrixSymbol) and isinstance(upd, MatMul):
            _, mm = upd.as_coeff_mmul()
            facs = list(mm.args)
            if len(facs) in (2, 3):
                return base, facs
    return None


def r_det_lemma(e):
    """``log det(A + U C V) -> log det A + log det(I + C V A^-1 U)`` (base ``A`` a leaf)."""
    if e.func == sp.log and len(e.args) == 1 and isinstance(e.args[0], sp.Determinant):
        split = _split_lowrank(e.args[0].arg)
        if split:
            A, facs = split
            if len(facs) == 2:
                U, V = facs
                inner = Identity(A.shape[0]) + V * A.I * U
            else:
                U, Cm, V = facs
                inner = Identity(Cm.shape[0]) + Cm * V * A.I * U
            return sp.log(sp.Determinant(A)) + sp.log(sp.Determinant(inner))
    return None


def r_woodbury(e):
    """``(A + U C V)^-1 -> A^-1 - A^-1 U (C^-1 + V A^-1 U)^-1 V A^-1`` (base ``A`` a leaf)."""
    if isinstance(e, Inverse):
        split = _split_lowrank(e.arg)
        if split:
            A, facs = split
            Ai = A.I
            if len(facs) == 2:
                U, V = facs
                cap = Identity(A.shape[0]) + V * Ai * U
            else:
                U, Cm, V = facs
                cap = Cm.I + V * Ai * U
            return Ai - Ai * (facs[0]) * cap.I * (facs[-1]) * Ai
    return None


STRUCTURAL: list[Rule] = [
    r_adjoint_distribute, r_symmetry, r_inverse_cancel, r_matadd_combine,
]
EXPANSION: list[Rule] = [r_det_lemma, r_woodbury]


# ----------------------------------------------------------------------
# engine
# ----------------------------------------------------------------------
class RewriteEngine:
    """Bottom-up fixpoint rewriter over a list of rules."""

    def __init__(self, rules: list[Rule], max_iter: int = 50) -> None:
        self.rules = rules
        self.max_iter = max_iter

    def _node(self, e, trace):
        if e.args:
            new_args, changed = [], False
            for a in e.args:
                na, ch = self._node(a, trace)
                new_args.append(na)
                changed = changed or ch
            if changed:
                e = e.func(*new_args)
        else:
            changed = False
        for rule in self.rules:
            r = rule(e)
            if r is not None and r != e:
                trace.append(getattr(rule, "__name__", "rule"))
                return r, True
        return e, changed

    def normalize(self, e) -> dict:
        trace: list[str] = []
        converged, it = False, 0
        for it in range(1, self.max_iter + 1):
            e, changed = self._node(e, trace)
            if not changed:
                converged = True
                break
        return {"expr": e, "trace": trace, "iters": it, "converged": converged}


def run_phases(e, phases: list[list[Rule]], max_iter: int = 50) -> dict:
    """Apply ordered rule groups, each to a fixpoint before the next (the strategy)."""
    trace: list[str] = []
    converged = True
    for rules in phases:
        res = RewriteEngine(rules, max_iter).normalize(e)
        e, converged = res["expr"], converged and res["converged"]
        trace += res["trace"]
    return {"expr": e, "trace": trace, "converged": converged}


_STRATEGIES = {
    "normalize": [STRUCTURAL],
    "capacity": [STRUCTURAL, EXPANSION],
}


def simplify_expr(e, strategy: str = "normalize"):
    """Simplify a single matrix expression with the named strategy."""
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; choose from {sorted(_STRATEGIES)}."
        )
    return run_phases(e, _STRATEGIES[strategy])["expr"]


def proves_zero(e) -> bool:
    """True if structural normalization reduces ``e`` to the zero matrix.

    This is the symbolic proof of conditional independence: the cross conditional
    covariance reduces to ``0`` exactly when ``A _||_ B | C``.
    """
    out = run_phases(e, [STRUCTURAL])["expr"]
    z = sp.simplify(out.doit()) if hasattr(out, "doit") else out
    return z == 0 or getattr(z, "is_ZeroMatrix", False)


def simplify_cmi(cmi, strategy: str = "normalize"):
    """Return a copy of a ``SymbolicCMI`` with each term/cross simplified."""
    from symbolic_dag.expr import SymbolicCMI

    terms = [(s, simplify_expr(M, strategy)) for s, M in cmi.logdet_terms]
    cross = simplify_expr(cmi.cross, strategy) if cmi.cross is not None else None
    return SymbolicCMI(
        definitions=cmi.definitions, output=cmi.output, metadata=cmi.metadata,
        A=cmi.A, B=cmi.B, C=cmi.C, logdet_terms=terms, cross=cross,
    )
