"""Symbolic matrix / Wirtinger differentiation of CMI.

``sympy``'s generic differentiation cannot differentiate ``log det M`` with
respect to a matrix variable (it returns the zero matrix), so --- exactly as the
block simplification rules had to be supplied by hand --- the gradient of CMI in
the block-symbolic form needs its own matrix-calculus engine. It has the same
recursive-structural shape as the rewrite engine:

  1. :func:`differential` --- the matrix differential by the product / sum /
     inverse / adjoint rules, treating ``F`` and ``F^H`` as independent
     (Wirtinger).
  2. :func:`wirtinger_grad_logdet` --- uses ``d log det M = tr(M^-1 dM)`` and
     extracts the coefficient of ``dF^H`` by the cyclic property of the trace
     (``tr(P dF^H Q) = tr(QP dF^H)``), giving the closed-form Wirtinger gradient.

Convention: the gradient is identified from ``dI = tr(G dF^H)`` as
``dI/dF^* = G``. A numerical library's autograd (PyTorch / cmi-dag) returns
``2G``; that factor is verified in the test suite, not assumed.

Scope (first milestone): single matrix variable, log-det objectives. The CMI
gradient is taken via the two-term form ``log det Sigma_{B|C} - log det
Sigma_{B|AC}`` with a single-node outer set ``B`` (or ``A``), so the terms are
single matrices and no block differentiation is needed. Both-multi-node
gradients and trace objectives are deferred.
"""

from __future__ import annotations

import sympy as sp
from sympy import (
    Adjoint,
    Identity,
    Inverse,
    MatAdd,
    MatMul,
    MatrixExpr,
    MatrixSymbol,
    ZeroMatrix,
)

from symbolic_dag.assumptions import apply_hermitian


def differential(e: MatrixExpr, F: MatrixSymbol, dF: MatrixSymbol) -> MatrixExpr:
    """Matrix differential of ``e`` w.r.t. ``F`` (with ``F`` and ``F^H`` independent)."""
    if e == F:
        return dF
    if e == Adjoint(F):
        return Adjoint(dF)
    if isinstance(e, (MatrixSymbol, Identity, ZeroMatrix)):
        return ZeroMatrix(*e.shape)  # constant
    if isinstance(e, MatAdd):
        return MatAdd(*[differential(a, F, dF) for a in e.args])
    if isinstance(e, MatMul):
        coeff, mm = e.as_coeff_mmul()
        args = list(mm.args)
        terms = []
        for i, a in enumerate(args):
            da = differential(a, F, dF)
            if getattr(da, "is_ZeroMatrix", False):
                continue
            terms.append(coeff * MatMul(*(args[:i] + [da] + args[i + 1:])))
        return MatAdd(*terms) if terms else ZeroMatrix(*e.shape)
    if isinstance(e, Adjoint):
        return Adjoint(differential(e.arg, F, dF))
    if isinstance(e, Inverse):
        return -e * differential(e.arg, F, dF) * e
    raise TypeError(
        f"differential: unsupported node {type(e).__name__} "
        "(block differentiation is not supported in this milestone)."
    )


def _add_terms(e: MatrixExpr) -> list[MatrixExpr]:
    e = e.doit()
    if getattr(e, "is_ZeroMatrix", False):
        return []
    return list(e.args) if isinstance(e, MatAdd) else [e]


def _coeff_of_conj(term: MatrixExpr, dF: MatrixSymbol):
    """If ``term`` contains ``dF^H``, return its cyclic coefficient; else ``None``."""
    term = term.doit()
    if isinstance(term, MatMul):
        scal, mm = term.as_coeff_mmul()
        facs = list(mm.args)
    else:
        scal, facs = sp.Integer(1), [term]
    if Adjoint(dF) not in facs:
        return None
    k = facs.index(Adjoint(dF))
    return (scal * MatMul(*(facs[k + 1:] + facs[:k]))).doit()


def wirtinger_grad_logdet(
    M: MatrixExpr, F: MatrixSymbol, dF: MatrixSymbol
) -> MatrixExpr:
    """Closed-form Wirtinger gradient ``d(log det M)/dF^*``, derived symbolically."""
    Minv = Inverse(M)
    G = None
    for t in _add_terms(differential(M, F, dF)):
        coeff = _coeff_of_conj((Minv * t).doit(), dF)
        if coeff is not None:
            G = coeff if G is None else (G + coeff).doit()
    return ZeroMatrix(*F.shape) if G is None else G


def wirtinger_grad_cmi(cmi, var: MatrixSymbol) -> MatrixExpr:
    """Closed-form Wirtinger gradient ``dI/dvar^*`` of a ``SymbolicCMI``.

    Computed from the single-matrix two-term (entropy) form

        I(A;B|C) = log det Sigma_{B|C} - log det Sigma_{B|AC},
        dI/dvar^* = grad(log det Sigma_{B|C}) - grad(log det Sigma_{B|AC}),

    where the conditional covariances are built by *sequential single-node
    conditioning* (:func:`symbolic_dag.information.conditional_covariance_seq`),
    so they are single matrix expressions for **arbitrary** conditioning ``C`` ---
    no block differentiation is needed. This works whenever ``A`` or ``B`` is a
    single node (the single one is taken as the outer set); both-multi-node is
    deferred. The Hermitian assumption is imposed and the result is structurally
    simplified.

    A numerical library's autograd returns ``2 *`` this gradient.

    Raises:
        NotImplementedError: if both ``A`` and ``B`` are multi-node.
    """
    from symbolic_dag.information import conditional_covariance_seq
    from symbolic_dag.rewrite import simplify_expr

    K = cmi.metadata.get("K")
    if K is None:
        raise ValueError("SymbolicCMI is missing its K-blocks (metadata['K']).")
    A, B, C = list(cmi.A), list(cmi.B), list(cmi.C)
    if len(B) == 1:
        outer, inner = B[0], A
    elif len(A) == 1:
        outer, inner = A[0], B
    else:
        raise NotImplementedError(
            "Wirtinger gradient requires A or B to be a single node "
            f"(got |A|={len(A)}, |B|={len(B)}); both-multi-node is deferred."
        )

    M1 = conditional_covariance_seq(K, outer, sorted(C))             # Sigma_{B|C}
    M2 = conditional_covariance_seq(K, outer, sorted(inner + C))     # Sigma_{B|AC}
    dF = MatrixSymbol("d" + var.name, *var.shape)
    G = (
        wirtinger_grad_logdet(M1, var, dF) - wirtinger_grad_logdet(M2, var, dF)
    ).doit()
    return simplify_expr(apply_hermitian(G), "normalize")
