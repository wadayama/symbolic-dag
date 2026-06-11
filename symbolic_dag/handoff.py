"""Hand-off / pretty type-setting of the closed forms for the *next* step.

The library produces closed-form CMI, Wirtinger gradients and KKT conditions; the
subsequent regime / structural / asymptotic analysis is human (or another tool's)
work. This module makes that hand-off frictionless, in three directions:

- :func:`render_pdf` — typeset a result to a **standalone PDF** (and optional PNG)
  via ``pdflatex``, so the analyst sees the rendered math immediately;
- :func:`to_mathematica` — emit a **Wolfram Language** string (``Dot`` /
  ``ConjugateTranspose`` / ``Inverse`` / ``Det`` ...), to continue in Mathematica;
- :func:`to_markdown` — a **Markdown** summary with LaTeX math, readable by both
  humans and LLMs.

All three are thin exporters over the existing symbolic results; the LaTeX string
layer lives in :mod:`symbolic_dag.latex`.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import sympy as sp
from sympy import (
    Adjoint,
    BlockMatrix,
    Identity,
    Inverse,
    MatAdd,
    MatMul,
    MatPow,
    MatrixExpr,
    MatrixSymbol,
    Transpose,
    ZeroMatrix,
)

from symbolic_dag.assumptions import HermitianMatrix

# ----------------------------------------------------------------------
# Wolfram Language (Mathematica) export
# ----------------------------------------------------------------------
def _wl_symbol(name: str) -> str:
    """A matrix symbol name as a Wolfram symbol (``F_1`` -> ``Subscript[F, 1]``)."""
    base, _, sub = name.partition("_")
    return f"Subscript[{base}, {sub.replace('_', ', ')}]" if sub else base


def _wl(e, scalar: bool = False) -> str:
    """Recursively render a ``sympy`` matrix expression as Wolfram Language.

    With ``scalar=True`` the matrices are treated as ``1x1`` scalars: ``Dot`` ->
    ``Times``, ``ConjugateTranspose`` -> ``Conjugate``, ``Inverse`` -> reciprocal,
    ``IdentityMatrix`` -> ``1`` --- a form ready for ``Integrate`` / ``Expectation``
    (e.g. an ergodic average over fading), where the matrix form is not.
    """
    if isinstance(e, (HermitianMatrix, MatrixSymbol)):
        return _wl_symbol(e.name)
    if isinstance(e, Identity):
        return "1" if scalar else f"IdentityMatrix[{e.rows}]"
    if isinstance(e, ZeroMatrix):
        return "0" if scalar else f"ConstantArray[0, {{{e.rows}, {e.cols}}}]"
    if isinstance(e, Adjoint):
        inner = _wl(e.arg, scalar)
        return f"Conjugate[{inner}]" if scalar else f"ConjugateTranspose[{inner}]"
    if isinstance(e, Transpose):
        inner = _wl(e.arg, scalar)
        return inner if scalar else f"Transpose[{inner}]"
    if isinstance(e, Inverse):
        inner = _wl(e.arg, scalar)
        return f"Power[{inner}, -1]" if scalar else f"Inverse[{inner}]"
    if isinstance(e, MatPow):
        exp = sp.mathematica_code(e.exp)
        head = "Power" if scalar else "MatrixPower"
        return f"{head}[{_wl(e.base, scalar)}, {exp}]"
    if isinstance(e, BlockMatrix):
        if scalar:
            raise ValueError("scalar=True does not support a BlockMatrix term.")
        rows = [
            "{" + ", ".join(_wl(e.blocks[i, j]) for j in range(e.blockshape[1])) + "}"
            for i in range(e.blockshape[0])
        ]
        return "ArrayFlatten[{" + ", ".join(rows) + "}]"
    if isinstance(e, MatAdd):
        return "Plus[" + ", ".join(_wl(a, scalar) for a in e.args) + "]"
    if isinstance(e, MatMul):
        coeff, mm = e.as_coeff_mmul()
        head = "Times" if scalar else "Dot"
        body = f"{head}[" + ", ".join(_wl(a, scalar) for a in mm.args) + "]"
        return body if coeff == 1 else f"Times[{sp.mathematica_code(coeff)}, {body}]"
    return sp.mathematica_code(e)  # plain scalar


def to_mathematica(
    obj,
    var: MatrixSymbol | None = None,
    *,
    scalar: bool = False,
    simplify: str | None = "normalize",
) -> str:
    """Wolfram Language string for a matrix expression or a ``SymbolicCMI``.

    Args:
        obj: a ``MatrixExpr`` (rendered directly), or a
            :class:`symbolic_dag.expr.SymbolicCMI`. For a CMI, the scalar
            ``Sum_k sign_k Log[Det[...]]`` is emitted; if ``var`` is given, its
            Wirtinger gradient ``dI/dvar*`` is emitted instead.
        var: differentiate the CMI w.r.t. this symbol (gradient hand-off).
        scalar: treat matrices as ``1x1`` scalars (``Det`` is dropped, ``Dot`` ->
            ``Times`` etc.) --- the form to feed Wolfram's ``Integrate`` /
            ``Expectation`` for a scalar/eigenvalue ergodic average.
        simplify: rewrite strategy applied to the CMI's matrices before export
            (``"normalize"`` / ``"capacity"`` / ``None``).

    Matrix-symbol names are mapped to ``Subscript[...]`` (e.g. ``Sigma_0`` ->
    ``Subscript[Sigma, 0]``); single-letter Wolfram built-ins (``N``, ``D``, ``E``,
    ``I``, ``K``, ``O``) may need renaming on the Mathematica side. Round-trip a
    Wolfram result back with :func:`from_mathematica`.
    """
    from symbolic_dag.expr import SymbolicCMI

    if isinstance(obj, SymbolicCMI):
        if var is not None:
            return _wl(obj.wirtinger_grad(var), scalar)
        if scalar:
            # Two-term entropy form  log det Sigma_{B|C} - log det Sigma_{B|AC}
            # (single-node outer) -- block-free, so it flattens to a clean scalar.
            from symbolic_dag.assumptions import apply_hermitian
            from symbolic_dag.information import conditional_covariance_seq
            from symbolic_dag.rewrite import simplify_expr

            A, B, C = list(obj.A), list(obj.B), list(obj.C)
            if len(B) == 1:
                outer, inner = B[0], A
            elif len(A) == 1:
                outer, inner = A[0], B
            else:
                raise ValueError(
                    "scalar=True needs A or B to be a single node "
                    f"(got |A|={len(A)}, |B|={len(B)})."
                )
            K = obj.metadata["K"]
            M1 = conditional_covariance_seq(K, outer, sorted(C))
            M2 = conditional_covariance_seq(K, outer, sorted(inner + C))
            if simplify is not None:
                M1 = simplify_expr(apply_hermitian(M1), simplify)
                M2 = simplify_expr(apply_hermitian(M2), simplify)
            return (
                f"Plus[Log[{_wl(M1, scalar=True)}], "
                f"Times[-1, Log[{_wl(M2, scalar=True)}]]]"
            )
        terms = obj.logdet_terms
        if simplify is not None:
            from symbolic_dag.rewrite import simplify_expr

            terms = [(s, simplify_expr(M, simplify)) for s, M in terms]
        parts = []
        for s, M in terms:
            term = f"Log[Det[{_wl(M)}]]"
            parts.append(term if s > 0 else f"Times[-1, {term}]")
        return "Plus[" + ", ".join(parts) + "]"
    return _wl(obj, scalar)


def from_mathematica(s: str):
    """Parse a Wolfram Language (scalar) expression string back into ``sympy``.

    Closes the loop after Wolfram does the heavy symbolic work (e.g. an ergodic
    integral): the returned ``sympy`` expression can be evaluated / cross-checked
    numerically. A thin wrapper over ``sympy.parsing.mathematica.parse_mathematica``
    that additionally maps the special functions common in these results
    (``ExpIntegralE`` -> ``expint``, ``ExpIntegralEi`` -> ``Ei``, ``Gamma`` ->
    ``uppergamma``/``gamma`` ...) to their ``sympy`` equivalents, so the result is
    numerically evaluable. It handles **scalar** expressions, not matrix algebra
    (``Dot`` / ``ConjugateTranspose``).
    """
    from sympy import Ei, erf, erfc, expint, gamma, uppergamma
    from sympy.core.function import AppliedUndef
    from sympy.parsing.mathematica import parse_mathematica

    expr = parse_mathematica(s)
    table = {
        "ExpIntegralE": lambda *a: expint(*a),
        "ExpIntegralEi": lambda *a: Ei(*a),
        "Gamma": lambda *a: uppergamma(*a) if len(a) == 2 else gamma(*a),
        "Erf": lambda *a: erf(*a),
        "Erfc": lambda *a: erfc(*a),
    }
    reps = {
        f: table[f.func.__name__](*f.args)
        for f in expr.atoms(AppliedUndef)
        if f.func.__name__ in table
    }
    return expr.xreplace(reps) if reps else expr


# ----------------------------------------------------------------------
# Markdown (human- and LLM-readable)
# ----------------------------------------------------------------------
def to_markdown(cmi, var: MatrixSymbol | None = None, *, expand: bool = True) -> str:
    """A Markdown summary of the CMI (and its gradient / KKT if ``var`` given).

    Math is rendered as LaTeX in ``$$`` blocks (read well by humans and LLMs); the
    structural and, if ``expand``, the explicit closed forms are both shown.
    """
    from symbolic_dag.latex import cmi_to_latex

    def _s(nodes):
        return "{" + ", ".join(map(str, nodes)) + "}"

    lines = [
        f"## Conditional mutual information  I(V_A; V_B | V_C)",
        "",
        f"- **A** = {_s(cmi.A)}, **B** = {_s(cmi.B)}, **C** = {_s(cmi.C)}",
        "",
        "**Structural form**",
        "",
        f"$$ {cmi_to_latex(cmi, expand=False)} $$",
    ]
    if expand:
        lines += ["", "**Closed form (explicit conditional covariances)**", "",
                  f"$$ {cmi_to_latex(cmi, expand=True)} $$"]
    if var is not None:
        G = cmi.wirtinger_grad(var)
        v = sp.latex(var)
        lines += [
            "", f"**Wirtinger gradient**  ∂I/∂{var.name}*", "",
            rf"$$ \frac{{\partial I}}{{\partial {v}^{{*}}}} = {sp.latex(G)} $$",
            "", "**Stationarity (KKT)**", "",
            rf"$$ {sp.latex(G)} = 0 $$",
            "",
            "> The numerical convention: a PyTorch/cmi-dag autograd returns "
            "twice this Wirtinger gradient.",
        ]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# PDF / PNG rendering
# ----------------------------------------------------------------------
_TEX_TEMPLATE = r"""\documentclass[border=10pt]{standalone}
\usepackage{amsmath,amssymb}
\begin{document}
$\displaystyle
\begin{aligned}
%s
\end{aligned}$
\end{document}
"""


def _aligned_body(cmi, var, expand) -> str:
    from symbolic_dag.latex import cmi_to_latex

    lines = [cmi_to_latex(cmi, expand=expand).replace(" = ", " &= ", 1)]
    if var is not None:
        G = cmi.wirtinger_grad(var)
        v = sp.latex(var)
        lines.append(rf"\frac{{\partial I}}{{\partial {v}^{{*}}}} &= {sp.latex(G)}")
        lines.append(rf"\text{{(KKT)}}\quad {sp.latex(G)} &= 0")
    return " \\\\[4pt]\n".join(lines)


def render_pdf(
    obj,
    path: str,
    *,
    var: MatrixSymbol | None = None,
    expand: bool = False,
    png: bool = False,
    dpi: int = 200,
) -> str:
    """Typeset a result to a standalone PDF (optionally also a PNG); return the PDF path.

    Args:
        obj: a :class:`symbolic_dag.expr.SymbolicCMI` (rendered as the CMI and, if
            ``var`` given, its gradient + KKT) or a ready LaTeX ``aligned`` body
            string.
        path: output ``.pdf`` path (the ``.tex`` is written alongside it).
        var: include the gradient / KKT for this symbol.
        expand: expand the CMI to explicit conditional covariances.
        png: also produce a PNG next to the PDF (needs ``pdftocairo`` from poppler).
        dpi: PNG resolution.

    Raises:
        RuntimeError: if ``pdflatex`` (or ``pdftocairo`` for PNG) is unavailable, or
            compilation fails.
    """
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise RuntimeError(
            "pdflatex not found; install a LaTeX distribution (e.g. TeX Live / MacTeX) "
            "to render PDFs. (to_markdown / to_mathematica need no external tools.)"
        )
    from symbolic_dag.expr import SymbolicCMI

    body = _aligned_body(obj, var, expand) if isinstance(obj, SymbolicCMI) else str(obj)
    path = os.path.abspath(path)
    outdir = os.path.dirname(path) or "."
    stem = os.path.splitext(os.path.basename(path))[0]
    tex_path = os.path.join(outdir, stem + ".tex")
    with open(tex_path, "w") as f:
        f.write(_TEX_TEMPLATE % body)

    proc = subprocess.run(
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error",
         "-output-directory", outdir, tex_path],
        capture_output=True, text=True,
    )
    pdf_path = os.path.join(outdir, stem + ".pdf")
    if not os.path.exists(pdf_path):
        tail = "\n".join(proc.stdout.splitlines()[-15:])
        raise RuntimeError(f"pdflatex failed to produce a PDF:\n{tail}")

    for ext in (".aux", ".log"):
        aux = os.path.join(outdir, stem + ext)
        if os.path.exists(aux):
            os.remove(aux)

    if png:
        pdftocairo = shutil.which("pdftocairo")
        if pdftocairo is None:
            raise RuntimeError("pdftocairo (poppler) not found; cannot produce a PNG.")
        subprocess.run(
            [pdftocairo, "-png", "-r", str(dpi), "-singlefile", pdf_path,
             os.path.join(outdir, stem)],
            check=True, capture_output=True,
        )
    return pdf_path
