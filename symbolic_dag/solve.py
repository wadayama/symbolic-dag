"""Closed-form solving of *linear* matrix stationarity (KKT) equations.

A capacity stationarity ``dI/dF* = 0`` is generally nonlinear in ``F`` (``F`` sits
inside an inverse), so it is left to the analyst. But the **MMSE / linear-estimator**
stationarity is *linear* in the variable: the objective is quadratic, so its
Wirtinger gradient is affine and the optimum has a closed form. :func:`solve_stationary`
solves exactly those directly-invertible shapes; the canonical instance is the
LMMSE / Wiener filter ``W·Σ_Y − Σ_XY = 0 → W = Σ_XY·Σ_Y⁻¹``.
"""

from __future__ import annotations

import sympy as sp
from sympy import Identity, MatAdd, MatMul, MatrixExpr, MatrixSymbol, Mul, ZeroMatrix


def _split_term(t: MatrixExpr, var: MatrixSymbol):
    """Split a var-term into ``(scalar, [left mats], [right mats])`` or ``None``.

    ``None`` means ``var`` is nested (inside an adjoint/inverse) or appears more
    than once -- i.e. the term is not linear in ``var`` as a bare factor.
    """
    facs = list(t.args) if isinstance(t, MatMul) else [t]
    mats = [f for f in facs if isinstance(f, MatrixExpr)]
    scals = [f for f in facs if not isinstance(f, MatrixExpr)]
    if sum(1 for m in mats if m == var) != 1:
        return None
    k = mats.index(var)
    left, right = mats[:k], mats[k + 1:]
    # var must be a *bare* factor: it may not also appear nested (e.g. inside an
    # inverse) in the surrounding matrix factors, NOR inside a scalar coefficient
    # (e.g. ``Trace(... var ...)``), else the term is nonlinear in var.
    if any(m.has(var) for m in left + right):
        return None
    if any(s.has(var) for s in scals):
        return None
    coeff = Mul(*scals) if scals else sp.Integer(1)
    return coeff, left, right


def solve_stationary(equation, var: MatrixSymbol) -> MatrixExpr:
    """Solve a linear matrix stationarity equation ``equation == 0`` for ``var``.

    ``equation`` (an ``Eq`` or a bare expression, e.g. a Wirtinger gradient
    ``dJ/dvar*``) must be **affine in ``var``**: a sum of constant terms and terms
    ``L · var · R`` with ``var`` a bare factor appearing once. The directly
    invertible shapes are solved:

    * right-linear ``var·M + S = 0``      → ``var = -S·M⁻¹``
    * left-linear  ``M·var + S = 0``      → ``var = -M⁻¹·S``
    * two-sided    ``L·var·R + S = 0`` (single term) → ``var = -L⁻¹·S·R⁻¹``

    The result is Hermitian-reduced and structurally simplified.

    Raises:
        NotImplementedError: if ``var`` is nested (adjoint/inverse) or the equation
            is genuinely two-sided with several terms (a Sylvester equation, which
            has no closed-form matrix solution for symbolic dimensions).
        ValueError: if ``equation`` does not contain ``var``.
    """
    from symbolic_dag.assumptions import apply_hermitian
    from symbolic_dag.rewrite import simplify_expr

    if isinstance(equation, sp.Equality):
        equation = equation.lhs - equation.rhs
    e = equation.doit().expand()
    terms = list(e.args) if isinstance(e, MatAdd) else [e]

    const, var_terms = [], []
    for t in terms:
        if not t.has(var):
            const.append(t)
            continue
        split = _split_term(t, var)
        if split is None:
            raise NotImplementedError(
                "solve_stationary: `var` appears nested (adjoint / inverse / "
                "scalar coefficient such as a Trace) or more than once; only "
                "equations affine and linear in `var` are supported."
            )
        var_terms.append(split)
    if not var_terms:
        raise ValueError("equation does not contain the variable.")

    S = MatAdd(*const).doit() if const else ZeroMatrix(*var.shape)
    rows, cols = var.shape

    all_left_empty = all(not L for _, L, _ in var_terms)
    all_right_empty = all(not R for _, _, R in var_terms)

    if all_left_empty:  # var · M + S = 0
        M = MatAdd(*[c * (MatMul(*R) if R else Identity(cols))
                     for c, _, R in var_terms]).doit()
        sol = (-S) * M.I
    elif all_right_empty:  # M · var + S = 0
        M = MatAdd(*[c * (MatMul(*L) if L else Identity(rows))
                     for c, L, _ in var_terms]).doit()
        sol = M.I * (-S)
    elif len(var_terms) == 1:  # L · var · R + S = 0
        c, L, R = var_terms[0]
        Lm = MatMul(*L).doit() if L else Identity(rows)
        Rm = MatMul(*R).doit() if R else Identity(cols)
        sol = (sp.Integer(-1) / c) * (Lm.I * S * Rm.I)
    else:
        raise NotImplementedError(
            "solve_stationary: two-sided multi-term (Sylvester) equation has no "
            "closed-form matrix solution for symbolic dimensions."
        )

    return simplify_expr(apply_hermitian(sol.doit()), "normalize")
