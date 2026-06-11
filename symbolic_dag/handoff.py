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


def _wl(e) -> str:
    """Recursively render a ``sympy`` matrix expression as Wolfram Language."""
    if isinstance(e, (HermitianMatrix, MatrixSymbol)):
        return _wl_symbol(e.name)
    if isinstance(e, Identity):
        return f"IdentityMatrix[{e.rows}]"
    if isinstance(e, ZeroMatrix):
        return f"ConstantArray[0, {{{e.rows}, {e.cols}}}]"
    if isinstance(e, Adjoint):
        return f"ConjugateTranspose[{_wl(e.arg)}]"
    if isinstance(e, Transpose):
        return f"Transpose[{_wl(e.arg)}]"
    if isinstance(e, Inverse):
        return f"Inverse[{_wl(e.arg)}]"
    if isinstance(e, MatPow):
        return f"MatrixPower[{_wl(e.base)}, {sp.mathematica_code(e.exp)}]"
    if isinstance(e, BlockMatrix):
        rows = [
            "{" + ", ".join(_wl(e.blocks[i, j]) for j in range(e.blockshape[1])) + "}"
            for i in range(e.blockshape[0])
        ]
        return "ArrayFlatten[{" + ", ".join(rows) + "}]"
    if isinstance(e, MatAdd):
        return "Plus[" + ", ".join(_wl(a) for a in e.args) + "]"
    if isinstance(e, MatMul):
        coeff, mm = e.as_coeff_mmul()
        dot = "Dot[" + ", ".join(_wl(a) for a in mm.args) + "]"
        return dot if coeff == 1 else f"Times[{sp.mathematica_code(coeff)}, {dot}]"
    return sp.mathematica_code(e)  # plain scalar


def to_mathematica(obj, var: MatrixSymbol | None = None, *, simplify: str | None = "normalize") -> str:
    """Wolfram Language string for a matrix expression or a ``SymbolicCMI``.

    Args:
        obj: a ``MatrixExpr`` (rendered directly), or a
            :class:`symbolic_dag.expr.SymbolicCMI`. For a CMI, the scalar
            ``Sum_k sign_k Log[Det[...]]`` is emitted; if ``var`` is given, its
            Wirtinger gradient ``dI/dvar*`` is emitted instead.
        var: differentiate the CMI w.r.t. this symbol (gradient hand-off).
        simplify: rewrite strategy applied to the CMI's matrices before export
            (``"normalize"`` / ``"capacity"`` / ``None``).

    Matrix-symbol names are mapped to ``Subscript[...]`` (e.g. ``Sigma_0`` ->
    ``Subscript[Sigma, 0]``); single-letter Wolfram built-ins (``N``, ``D``, ``E``,
    ``I``, ``K``, ``O``) may need renaming on the Mathematica side.
    """
    from symbolic_dag.expr import SymbolicCMI

    if isinstance(obj, SymbolicCMI):
        if var is not None:
            return _wl(obj.wirtinger_grad(var))
        terms = obj.logdet_terms
        if simplify is not None:
            from symbolic_dag.rewrite import simplify_expr

            terms = [(s, simplify_expr(M, simplify)) for s, M in terms]
        parts = [
            f"Log[Det[{_wl(M)}]]" if s > 0 else f"Times[-1, Log[Det[{_wl(M)}]]]"
            for s, M in terms
        ]
        return "Plus[" + ", ".join(parts) + "]"
    return _wl(obj)


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
