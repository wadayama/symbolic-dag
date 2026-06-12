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

from collections import Counter
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
# display rules (cosmetic, value-preserving --- the "display" strategy)
# ----------------------------------------------------------------------
def r_block_collapse(e):
    """Collapse block-matrix algebra (sums/products of blocks) into one block.

    The lazy core deliberately keeps block Schur products unevaluated; for
    *presentation* the collapsed single block is the readable form.
    """
    from sympy.matrices.expressions.blockmatrix import BlockMatrix

    if isinstance(e, (MatMul, MatAdd, Inverse)) and e.has(BlockMatrix):
        try:
            c = sp.block_collapse(e)
        except Exception:
            return None
        if c != e:
            return c
    return None


def r_distribute(e):
    """Distribute a product over a sum (``H (N + T) -> H N + H T``).

    Cancelling pairs hidden inside a product only become visible to
    :func:`r_collect` once distributed. (The gradient engine needed the same
    fix internally: ``matderiv._add_terms`` expands for the same reason.)
    """
    if isinstance(e, MatMul) and any(isinstance(a, MatAdd) for a in e.args):
        x = e.expand()
        if x != e:
            return x
    return None


def r_collect(e):
    """Collect like terms in a sum (``N + T - T -> N``).

    ``MatAdd`` does not combine syntactically equal terms on construction;
    ``doit(deep=False)`` does, without touching the term subtrees.
    """
    if isinstance(e, MatAdd):
        z = e.doit(deep=False)
        if z != e:
            return z
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


def r_sylvester(e):
    """``log det(I + X1 X2 ... Xk) -> log det(I + canonical cyclic rotation)``.

    The Sylvester / Weinstein--Aronszajn identity ``det(I_m + A B) = det(I_n + B A)``
    generalises to ``det(I + X1 ... Xk) = det(I + X_{i} ... Xk X1 ... X_{i-1})`` for
    any cyclic shift. Rotating to the lexicographically smallest shift gives a
    *canonical*, idempotent normal form (so two differently-arranged capacity
    log-dets become syntactically equal). Value-preserving; expansion phase only.
    """
    if e.func == sp.log and len(e.args) == 1 and isinstance(e.args[0], sp.Determinant):
        M = e.args[0].arg
        if isinstance(M, MatAdd) and len(M.args) == 2:
            idents = [a for a in M.args if isinstance(a, Identity)]
            others = [a for a in M.args if not isinstance(a, Identity)]
            if len(idents) == 1 and len(others) == 1 and isinstance(others[0], MatMul):
                coeff, mm = others[0].as_coeff_mmul()
                facs = list(mm.args)
                if len(facs) >= 2:
                    rotations = [tuple(facs[i:] + facs[:i]) for i in range(len(facs))]
                    best = min(rotations, key=lambda r: tuple(str(x) for x in r))
                    if tuple(facs) != best:
                        inner = Identity(best[0].shape[0]) + coeff * MatMul(*best)
                        return sp.log(sp.Determinant(inner))
    return None


STRUCTURAL: list[Rule] = [
    r_adjoint_distribute, r_symmetry, r_inverse_cancel, r_matadd_combine,
]
EXPANSION: list[Rule] = [r_det_lemma, r_woodbury, r_sylvester]
DISPLAY: list[Rule] = [r_block_collapse, r_distribute, r_collect]


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
    # presentation cleanup: normalize first (the proven phasing), then a joint
    # fixpoint where distribution / collection / block collapse interleave with
    # the structural rules (a distribution exposes an inverse-cancel, which
    # exposes a collect, ...). Value-preserving; opt-in, so the default
    # "normalize" behaviour (and proves_zero) is untouched.
    "display": [STRUCTURAL, STRUCTURAL + DISPLAY],
}


def simplify_expr(e, strategy: str = "normalize"):
    """Simplify a single matrix expression with the named strategy."""
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; choose from {sorted(_STRATEGIES)}."
        )
    return run_phases(e, _STRATEGIES[strategy])["expr"]


def _signed_logdet_pairs(expr):
    """Parse a scalar ``+/- log det(M) +/- ...`` sum into ``[(sign, M), ...]``.

    Returns ``None`` if any term is not a plain signed log-det (the caller
    falls back to matrix-level simplification).
    """
    if expr == 0:
        return []
    pairs = []
    for t in (expr.as_ordered_terms() if isinstance(expr, sp.Add) else [expr]):
        sign = 1
        if isinstance(t, sp.Mul):
            c, rest = t.as_coeff_Mul()
            if c not in (1, -1):
                return None
            sign, t = int(c), rest
        if (
            t.func == sp.log
            and len(t.args) == 1
            and isinstance(t.args[0], sp.Determinant)
        ):
            pairs.append((sign, t.args[0].arg))
        else:
            return None
    return pairs


def simplify_logdet_terms(
    terms: list[tuple[int, sp.Basic]], strategy: str = "normalize"
) -> "list[tuple[int, sp.Basic]] | None":
    """Simplify signed log-det terms at the SCALAR level, where the log-det rules live.

    :func:`simplify_cmi` maps :func:`simplify_expr` over the bare matrices, so
    the log-det-level EXPANSION rules (determinant lemma, Sylvester) can never
    fire there --- they match ``log(det(.))`` nodes. This function runs the
    strategy on each ``log det M_k`` term instead (so ``"capacity"`` can split
    ``log det(N + H Q H^H)`` into ``log det N + log det(I + .)``), flattens the
    results in order, and drops syntactically cancelling ``+/- log det M``
    pairs (which is how ``log det N`` cancels to leave the capacity form).

    Returns the new ``[(sign, matrix), ...]`` list (order-preserving), or
    ``None`` if some term did not stay a pure signed log-det sum --- the caller
    should then fall back to the matrix-level path.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; choose from {sorted(_STRATEGIES)}."
        )
    flat: list[tuple[int, sp.Basic]] = []
    for s, M in terms:
        out = run_phases(sp.log(sp.Determinant(M)), _STRATEGIES[strategy])["expr"]
        pairs = _signed_logdet_pairs(out)
        if pairs is None:
            return None
        flat.extend((s * ps, PM) for ps, PM in pairs)
    net: Counter = Counter()
    for s, M in flat:
        net[M] += s
    result, seen = [], set()
    for s, M in flat:
        if M in seen:
            continue
        seen.add(M)
        n = net[M]
        result.extend([(1 if n > 0 else -1, M)] * abs(n))
    return result


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
