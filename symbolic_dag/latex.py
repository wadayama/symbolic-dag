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


def _set(nodes, names=None) -> str:
    if names:
        return ",".join(names.get(n, str(n)) for n in nodes)
    return ",".join(map(str, nodes))


def _cond_cov_symbol(U, C, names=None) -> str:
    u = _set(U, names)
    return rf"\Sigma_{{{u}\mid {_set(C, names)}}}" if C else rf"\Sigma_{{{u}}}"


def _det_of_symbol(content: str, det_style: str) -> str:
    """Wrap a (string) matrix symbol as a determinant: ``|X|`` or ``det(X)``."""
    if det_style == "det":
        return rf"\det\left({content}\right)"
    return rf"\left|{content}\right|"


def _det_of_expr(M, det_style: str) -> str:
    """Wrap a matrix expression as a determinant: ``|X|`` or ``det(X)``."""
    if det_style == "det":
        return rf"\det\left({sp.latex(M)}\right)"
    return sp.latex(Determinant(M))


def _check_det_style(det_style: str) -> None:
    if det_style not in ("bars", "det"):
        raise ValueError(
            f"unknown det_style {det_style!r}; choose 'bars' (|X|) or 'det' (det(X))."
        )


def cmi_to_latex(
    cmi,
    *,
    expand: bool = False,
    simplify: str | None = "normalize",
    det_style: str = "bars",
) -> str:
    """LaTeX for the CMI.

    Args:
        cmi: A :class:`symbolic_dag.expr.SymbolicCMI`.
        expand: If False (default), render with named conditional covariances
            ``Sigma_{A|C}`` etc.; if True, substitute the explicit matrix
            expressions (simplified by ``simplify``).
        simplify: Rewrite strategy for the expanded form (``"normalize"``,
            ``"capacity"``, ``"display"``, or ``None``). Applied at the scalar
            log-det level (:func:`symbolic_dag.rewrite.simplify_logdet_terms`),
            so the log-det rules --- determinant lemma, Sylvester --- actually
            fire; ``"capacity"`` on a two-term CMI yields ``log det(I + .)``.
        det_style: ``"bars"`` (default) renders ``|X|``; ``"det"`` renders
            ``det(X)``.

    If the CMI was built by :class:`symbolic_dag.builder.GaussianDAG`, its node
    names are used (``V_X`` instead of ``V_0``); otherwise indices are shown.
    A CMI in two-term form (:meth:`symbolic_dag.expr.SymbolicCMI.two_term`)
    renders structurally as ``log|S_{B|C}| - log|S_{B|AC}|``.
    """
    _check_det_style(det_style)
    A, B, C = cmi.A, cmi.B, cmi.C
    names = cmi.metadata.get("node_names")
    a, b = _set(A, names), _set(B, names)
    lhs = f"I(V_{{{a}}}; V_{{{b}}}" + (rf" \mid V_{{{_set(C, names)}}})" if C else ")")

    if not expand:
        if cmi.metadata.get("form") == "two_term_logdet":
            U, other = (B, A) if cmi.metadata.get("two_term_side", "B") == "B" else (A, B)
            rhs = (
                r"\log" + _det_of_symbol(_cond_cov_symbol(U, C, names), det_style)
                + r"-\log"
                + _det_of_symbol(
                    _cond_cov_symbol(U, sorted(tuple(other) + tuple(C)), names),
                    det_style,
                )
            )
            return f"{lhs} = {rhs}"
        AB = sorted(tuple(A) + tuple(B))
        rhs = (
            r"\log" + _det_of_symbol(_cond_cov_symbol(A, C, names), det_style)
            + r"+\log" + _det_of_symbol(_cond_cov_symbol(B, C, names), det_style)
            + r"-\log" + _det_of_symbol(_cond_cov_symbol(AB, C, names), det_style)
        )
        return f"{lhs} = {rhs}"

    terms = cmi.logdet_terms
    if simplify is not None:
        from symbolic_dag.rewrite import simplify_expr, simplify_logdet_terms

        scalar = simplify_logdet_terms(terms, simplify)
        terms = (
            scalar
            if scalar is not None
            else [(s, simplify_expr(M, simplify)) for s, M in terms]
        )
    rhs = "".join(
        (r"+\log" if s > 0 else r"-\log") + _det_of_expr(M, det_style)
        for s, M in terms
    ).lstrip("+")
    return f"{lhs} = {rhs or '0'}"


def expr_to_latex(M: MatrixExpr) -> str:
    """LaTeX for a matrix expression (gradient, Schur term, ...)."""
    return sp.latex(M)


def report(
    cmi,
    var: sp.MatrixSymbol | None = None,
    *,
    expand: bool = False,
    det_style: str = "bars",
) -> str:
    """A LaTeX ``align*`` block: the CMI and (if ``var`` given) its gradient and KKT.

    The CMI is shown in structural form (named conditional covariances) unless
    ``expand=True``; the gradient and stationarity are the explicit closed forms.
    """
    cmi_line = cmi_to_latex(cmi, expand=expand, det_style=det_style).replace(
        " = ", " &= ", 1
    )
    lines = [cmi_line]
    if var is not None:
        G = cmi.wirtinger_grad(var)
        v = sp.latex(var)
        lines.append(rf"\frac{{\partial I}}{{\partial {v}^{{*}}}} &= {sp.latex(G)}")
        lines.append(rf"\text{{(KKT)}}\quad {sp.latex(G)} &= 0")
    return "\\begin{align*}\n" + " \\\\\n".join(lines) + "\n\\end{align*}"
