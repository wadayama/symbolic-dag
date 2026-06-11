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

from symbolic_dag.assumptions import HermitianMatrix, apply_hermitian


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
    # Distribute products over sums so that any ``dF`` / ``dF^H`` ends up as a
    # top-level factor of each additive term. The ``Inverse`` differential rule
    # ``-e d(arg) e`` leaves an inner ``MatAdd`` factor (when ``F`` sits inside a
    # conditioning inverse); without this expansion the conjugate differential
    # would stay buried in a product and the coefficient extraction would miss it.
    try:
        e = e.expand()
    except (AttributeError, TypeError):  # pragma: no cover - defensive
        pass
    if getattr(e, "is_ZeroMatrix", False):
        return []
    return list(e.args) if isinstance(e, MatAdd) else [e]


def _coeff_of(term: MatrixExpr, factor) -> "MatrixExpr | None":
    """Cyclic coefficient ``C`` such that ``term = tr(C · factor)``, or ``None``.

    Uses ``tr(P · factor · Q) = tr(Q P · factor)``; ``factor`` is ``dF`` (plain,
    holomorphic) or ``Adjoint(dF)`` (the conjugate differential).
    """
    term = term.doit()
    if isinstance(term, MatMul):
        scal, mm = term.as_coeff_mmul()
        facs = list(mm.args)
    else:
        scal, facs = sp.Integer(1), [term]
    if factor not in facs:
        return None
    k = facs.index(factor)
    return (scal * MatMul(*(facs[k + 1:] + facs[:k]))).doit()


def _coeff_of_conj(term: MatrixExpr, dF: MatrixSymbol):
    """If ``term`` contains ``dF^H``, return its cyclic coefficient; else ``None``."""
    return _coeff_of(term, Adjoint(dF))


def _hermitian_grad(
    M: MatrixExpr, Q: MatrixSymbol, dQ: MatrixSymbol, *, logdet: bool
) -> MatrixExpr:
    """Gradient ``G`` (``df = tr(G dQ)``) for a HERMITIAN variable ``Q``.

    Since ``Q^H = Q``, the differential has a single ``dQ`` (not an independent
    ``dQ^H``): we impose ``Adjoint(Q) -> Q`` first (``apply_hermitian``), then
    extract the coefficient of the plain ``dQ``. With ``logdet`` the ``M^{-1}``
    prefactor of ``d log det M = tr(M^{-1} dM)`` is included; otherwise it is the
    trace objective ``d tr(M) = tr(dM)``. (No autograd factor of 2 here: the
    Hermitian gradient is identified directly from ``df = tr(G dQ)``.)
    """
    M = apply_hermitian(M)
    prefactor = Inverse(M) if logdet else None
    G = None
    for t in _add_terms(differential(M, Q, dQ)):
        ext = (prefactor * t).doit() if prefactor is not None else t
        coeff = _coeff_of(ext, dQ)
        if coeff is not None:
            G = coeff if G is None else (G + coeff).doit()
    return ZeroMatrix(*Q.shape) if G is None else G


def wirtinger_grad_logdet(
    M: MatrixExpr, F: MatrixSymbol, dF: MatrixSymbol
) -> MatrixExpr:
    """Closed-form gradient of ``log det M`` w.r.t. ``F``, derived symbolically.

    For a plain ``MatrixSymbol`` ``F`` this is the Wirtinger gradient
    ``d(log det M)/dF^*`` (``dI = tr(G dF^H)``; autograd returns ``2G``). For a
    :class:`HermitianMatrix` covariance ``Q`` it is the **Hermitian** gradient
    ``df = tr(G dQ)`` (no factor of 2) --- e.g. the capacity gradient
    ``d log det(N + H Q H^H)/dQ = H^H (N + H Q H^H)^{-1} H``.
    """
    if isinstance(F, HermitianMatrix):
        return _hermitian_grad(M, F, dF, logdet=True)
    Minv = Inverse(M)
    G = None
    for t in _add_terms(differential(M, F, dF)):
        coeff = _coeff_of_conj((Minv * t).doit(), dF)
        if coeff is not None:
            G = coeff if G is None else (G + coeff).doit()
    return ZeroMatrix(*F.shape) if G is None else G


def wirtinger_grad_trace(
    M: MatrixExpr, F: MatrixSymbol, dF: MatrixSymbol
) -> MatrixExpr:
    """Closed-form Wirtinger gradient ``d(tr M)/dF^*``, derived symbolically.

    Uses ``d tr(M) = tr(dM)`` and extracts the coefficient of ``dF^H`` by the
    cyclic property of the trace --- the trace analogue of
    :func:`wirtinger_grad_logdet`, without the ``M^{-1}`` prefactor. The primary
    use is an **MMSE / LMMSE** objective ``tr(Sigma_{X|Y})`` (an estimation-error
    covariance). For a plain ``MatrixSymbol`` autograd returns ``2 *`` this; for a
    :class:`HermitianMatrix` covariance ``Q`` it is the Hermitian gradient
    ``d tr(M)/dQ`` with ``df = tr(G dQ)`` (no factor of 2).
    """
    if isinstance(F, HermitianMatrix):
        return _hermitian_grad(M, F, dF, logdet=False)
    G = None
    for t in _add_terms(differential(M, F, dF)):
        coeff = _coeff_of_conj(t, dF)
        if coeff is not None:
            G = coeff if G is None else (G + coeff).doit()
    return ZeroMatrix(*F.shape) if G is None else G


def trace_grad(M: MatrixExpr, var: MatrixSymbol) -> MatrixExpr:
    """Closed-form Wirtinger gradient ``d(tr M)/dvar^*`` of a matrix expression.

    Convenience wrapper over :func:`wirtinger_grad_trace`: it creates the
    differential symbol, imposes the Hermitian assumption, and structurally
    simplifies the result. For an MMSE design, pass the estimation-error
    covariance ``M = Sigma_{X|Y}`` (e.g. from
    :func:`symbolic_dag.information.mmse_error_covariance`); ``tr(M)`` is the
    scalar MMSE and this returns ``d(MMSE)/dvar^*``. Autograd returns ``2 *`` it.
    """
    from symbolic_dag.rewrite import simplify_expr

    dF = MatrixSymbol("d" + var.name, *var.shape)
    G = wirtinger_grad_trace(M, var, dF).doit()
    return simplify_expr(apply_hermitian(G), "normalize")


def wirtinger_grad_cmi(cmi, var: MatrixSymbol) -> MatrixExpr:
    """Closed-form Wirtinger gradient ``dI/dvar^*`` of a ``SymbolicCMI``.

    Computed from the single-matrix two-term (entropy) form

        I(A;B|C) = log det Sigma_{B|C} - log det Sigma_{B|AC},
        dI/dvar^* = grad(log det Sigma_{B|C}) - grad(log det Sigma_{B|AC}),

    where the conditional covariances are built by *sequential single-node
    conditioning* (:func:`symbolic_dag.information.conditional_covariance_seq`),
    so they are single matrix expressions for **arbitrary** conditioning ``C`` ---
    no block differentiation is needed.

    When ``A`` or ``B`` is a single node, that node is the outer set directly.
    When **both** are multi-node, the chain rule of mutual information

        I(A;B|C) = sum_i I(a_i; B | a_{<i}, C)            (chain over the smaller set)

    expands the gradient into a sum of single-node terms (each ``a_i`` is the outer
    set, the rest joins the conditioning), so the same single-node machinery
    applies and block differentiation is still avoided. The Hermitian assumption is
    imposed and the result is structurally simplified.

    A numerical library's autograd returns ``2 *`` this gradient.
    """
    from symbolic_dag.information import conditional_covariance_seq
    from symbolic_dag.rewrite import simplify_expr

    K = cmi.metadata.get("K")
    if K is None:
        raise ValueError("SymbolicCMI is missing its K-blocks (metadata['K']).")
    A, B, C = list(cmi.A), list(cmi.B), list(cmi.C)
    dF = MatrixSymbol("d" + var.name, *var.shape)
    if len(B) == 1:
        outer, inner = B[0], A
    elif len(A) == 1:
        outer, inner = A[0], B
    else:
        # Both multi-node: chain rule over the smaller set. Each term has a
        # single-node outer set, so the single-node log-det gradient applies.
        chain, other = (B, A) if len(B) <= len(A) else (A, B)
        chain, other, C_s = sorted(chain), sorted(other), sorted(C)
        G = ZeroMatrix(*var.shape)
        for i, node in enumerate(chain):
            pre = chain[:i]
            M1 = conditional_covariance_seq(K, node, sorted(pre + C_s))
            M2 = conditional_covariance_seq(K, node, sorted(pre + other + C_s))
            G = G + wirtinger_grad_logdet(M1, var, dF) - wirtinger_grad_logdet(M2, var, dF)
        return simplify_expr(apply_hermitian(G.doit()), "normalize")

    M1 = conditional_covariance_seq(K, outer, sorted(C))             # Sigma_{B|C}
    M2 = conditional_covariance_seq(K, outer, sorted(inner + C))     # Sigma_{B|AC}
    G = (
        wirtinger_grad_logdet(M1, var, dF) - wirtinger_grad_logdet(M2, var, dF)
    ).doit()
    return simplify_expr(apply_hermitian(G), "normalize")
