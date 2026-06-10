"""LaTeX export --- the readable hand-off of the closed forms to the analyst.

The library's responsibility ends at producing closed-form CMI and Wirtinger
gradients; the subsequent regime / KKT / structural analysis is human work. This
module makes that hand-off ergonomic.

Two levels of detail for the CMI:

- the **structural** form (default) writes it with named conditional
  covariances, ``I = log|Sigma_{A|C}| + log|Sigma_{B|C}| - log|Sigma_{AB|C}|`` ---
  compact and readable, the natural starting point for analysis;
- the **expanded** form substitutes the explicit Schur-complement matrix
  expressions.

The Wirtinger gradient and stationarity (KKT) condition are always rendered as
the explicit closed-form matrix expressions the engine derives.
"""

from __future__ import annotations

import sympy as sp
from sympy import Determinant, MatrixExpr


def _set(nodes) -> str:
    return ",".join(map(str, nodes))


def _cond_cov_symbol(U, C) -> str:
    u = _set(U)
    return rf"\Sigma_{{{u}\mid {_set(C)}}}" if C else rf"\Sigma_{{{u}}}"


def cmi_to_latex(cmi, *, expand: bool = False, simplify: str | None = "normalize") -> str:
    """LaTeX for the CMI.

    Args:
        cmi: A :class:`symbolic_dag.expr.SymbolicCMI`.
        expand: If False (default), render with named conditional covariances
            ``Sigma_{A|C}`` etc.; if True, substitute the explicit matrix
            expressions (simplified by ``simplify``).
        simplify: Rewrite strategy for the expanded matrices (``"normalize"``,
            ``"capacity"``, or ``None``).
    """
    A, B, C = cmi.A, cmi.B, cmi.C
    a, b = _set(A), _set(B)
    lhs = f"I(V_{{{a}}}; V_{{{b}}}" + (rf" \mid V_{{{_set(C)}}})" if C else ")")

    if not expand:
        AB = sorted(tuple(A) + tuple(B))
        rhs = (
            rf"\log\left|{_cond_cov_symbol(A, C)}\right|"
            rf"+\log\left|{_cond_cov_symbol(B, C)}\right|"
            rf"-\log\left|{_cond_cov_symbol(AB, C)}\right|"
        )
        return f"{lhs} = {rhs}"

    terms = cmi.logdet_terms
    if simplify is not None:
        from symbolic_dag.rewrite import simplify_expr

        terms = [(s, simplify_expr(M, simplify)) for s, M in terms]
    rhs = "".join(
        (r"+\log" if s > 0 else r"-\log") + sp.latex(Determinant(M)) for s, M in terms
    ).lstrip("+")
    return f"{lhs} = {rhs}"


def expr_to_latex(M: MatrixExpr) -> str:
    """LaTeX for a matrix expression (gradient, Schur term, ...)."""
    return sp.latex(M)


def report(cmi, var: sp.MatrixSymbol | None = None, *, expand: bool = False) -> str:
    """A LaTeX ``align*`` block: the CMI and (if ``var`` given) its gradient and KKT.

    The CMI is shown in structural form (named conditional covariances) unless
    ``expand=True``; the gradient and stationarity are the explicit closed forms.
    """
    cmi_line = cmi_to_latex(cmi, expand=expand).replace(" = ", " &= ", 1)
    lines = [cmi_line]
    if var is not None:
        G = cmi.wirtinger_grad(var)
        v = sp.latex(var)
        lines.append(rf"\frac{{\partial I}}{{\partial {v}^{{*}}}} &= {sp.latex(G)}")
        lines.append(rf"\text{{(KKT)}}\quad {sp.latex(G)} &= 0")
    return "\\begin{align*}\n" + " \\\\\n".join(lines) + "\n\\end{align*}"
