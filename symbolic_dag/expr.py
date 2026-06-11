"""Lazy K-recursive symbolic objects: ``RecursiveExpr`` and ``SymbolicCMI``.

The central design choice (from the design note and the smoke tests) is that a
conditional mutual information is NOT returned as one giant expanded formula but
as a *lazy* object: a small set of matrix intermediates plus a list of signed
log-determinant terms. The block determinants and inverses are held symbolically
and only acted on when asked --- simplified (the rewrite engine), differentiated
(the Wirtinger engine), or evaluated numerically.

Complex / Wirtinger convention (matching ``cmi-dag``): the CMI carries no factor
of one-half,

    I(V_A; V_B | V_C) = log det Sigma_{A|C} + log det Sigma_{B|C}
                        - log det Sigma_{AB|C},

(in nats). The cross conditional covariance ``Sigma_{AB|C}`` is kept too: it is
the zero matrix exactly iff ``A`` and ``B`` are conditionally independent given
``C``, which is how d-separation is proved symbolically.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from sympy import Determinant, MatrixExpr


def _to_numpy(M: MatrixExpr, subs: Mapping) -> np.ndarray:
    """Evaluate a (possibly block / symbolic-dimension) matrix at concrete values."""
    r = M.subs(subs).doit()
    if not isinstance(r, sp.MatrixBase):
        r = r.as_explicit()
    return np.array(r.tolist(), dtype=complex)


def _logdet(M: np.ndarray) -> float:
    return float(np.linalg.slogdet(M)[1])  # log|det|; real for HPD


@dataclass
class RecursiveExpr:
    """A straight-line program of matrix intermediates with a scalar output.

    Attributes:
        definitions: Ordered ``(symbol, matrix-expr)`` intermediate definitions
            (the lazy K-recursive form). May be empty for small gadgets.
        output: The scalar output expression (e.g. a sum of log-dets).
        metadata: Free-form annotations (form name, op counts, ...).
    """

    definitions: list[tuple[sp.Symbol, MatrixExpr]] = field(default_factory=list)
    output: sp.Expr | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class SymbolicCMI(RecursiveExpr):
    """Symbolic conditional mutual information ``I(V_A; V_B | V_C)``.

    Carries the signed log-determinant terms (so it can be evaluated and
    differentiated termwise) and the cross conditional covariance used for the
    d-separation proof. Built by
    :func:`symbolic_dag.information.conditional_mutual_information_from_k`.
    """

    A: tuple[int, ...] = ()
    B: tuple[int, ...] = ()
    C: tuple[int, ...] = ()
    # signed log-det terms: I = sum_k sign_k * log det M_k
    logdet_terms: list[tuple[int, MatrixExpr]] = field(default_factory=list)
    # cross conditional covariance Sigma_{AB|C} (zero iff A _||_ B | C)
    cross: MatrixExpr | None = None

    # ---- scalar form -------------------------------------------------
    def to_expr(self) -> sp.Expr:
        """The CMI as a scalar sympy expression: sum of signed ``log(det(.))``."""
        return sp.Add(
            *[sign * sp.log(Determinant(M)) for sign, M in self.logdet_terms]
        )

    # ---- numeric evaluation / verification ---------------------------
    def evaluate(self, subs: Mapping) -> float:
        """Evaluate the CMI (nats) at a concrete substitution of the symbols.

        ``subs`` must map every free matrix symbol to a concrete ``sympy.Matrix``
        (Hermitian PD for covariances), and the dimension symbol to an ``int`` if
        a symbolic dimension was used.
        """
        return float(
            sum(sign * _logdet(_to_numpy(M, subs)) for sign, M in self.logdet_terms)
        )

    def numeric_check(
        self, subs: Mapping, reference: float, *, atol: float = 1e-9
    ) -> tuple[bool, float]:
        """Compare :meth:`evaluate` against a ``reference`` value (e.g. cmi-dag).

        Returns ``(passed, abs_error)``.
        """
        err = abs(self.evaluate(subs) - reference)
        return bool(err < atol), err

    # ---- the symbolic operations (wired to the engines) --------------
    def simplify(self, strategy: str = "normalize") -> "SymbolicCMI":
        """Simplify each log-det term and the cross block with the rewrite engine.

        ``strategy`` selects the rule phase set (see
        :mod:`symbolic_dag.rewrite`). Returns a new ``SymbolicCMI``.
        """
        from symbolic_dag.rewrite import simplify_cmi

        return simplify_cmi(self, strategy=strategy)

    def is_conditionally_independent(self) -> bool:
        """Prove ``A _||_ B | C`` by reducing the cross block to the zero matrix."""
        from symbolic_dag.rewrite import proves_zero

        return self.cross is not None and proves_zero(self.cross)

    def wirtinger_grad(self, var: sp.MatrixSymbol) -> MatrixExpr:
        """Closed-form Wirtinger gradient ``dI/dvar^*`` (see :mod:`symbolic_dag.matderiv`)."""
        from symbolic_dag.matderiv import wirtinger_grad_cmi

        return wirtinger_grad_cmi(self, var)

    def stationarity(self, var: sp.MatrixSymbol) -> sp.Equality:
        """The KKT/stationarity condition ``dI/dvar^* = 0`` as a sympy ``Eq``."""
        from sympy import Eq, ZeroMatrix

        G = self.wirtinger_grad(var)
        return Eq(G, ZeroMatrix(*var.shape))

    # ---- PyTorch-based numerical verification (the cmidag extra) ------
    def torch_value(self, subs: Mapping, dim: int):
        """Evaluate the CMI as a differentiable ``torch`` scalar (see :mod:`symbolic_dag.verify`)."""
        from symbolic_dag.verify import torch_value

        return torch_value(self, subs, dim)

    def check(self, dim: int, **kwargs) -> dict:
        """Numerically verify the CMI value at random points with PyTorch.

        Returns ``{"passed", "max_abs_err", "samples"}``. Requires the ``cmidag``
        extra (torch); for a torch-free check use :func:`symbolic_dag.numpy_cmi`.
        """
        from symbolic_dag.verify import check

        return check(self, dim, **kwargs)

    # ---- hand-off / pretty type-setting ------------------------------
    def to_mathematica(self, var: sp.MatrixSymbol | None = None, **kwargs) -> str:
        """Wolfram Language string for the CMI (or its gradient if ``var`` given)."""
        from symbolic_dag.handoff import to_mathematica

        return to_mathematica(self, var, **kwargs)

    def to_markdown(self, var: sp.MatrixSymbol | None = None, **kwargs) -> str:
        """Markdown summary (LaTeX math) of the CMI and, if ``var`` given, its gradient/KKT."""
        from symbolic_dag.handoff import to_markdown

        return to_markdown(self, var, **kwargs)

    def to_pdf(self, path: str, var: sp.MatrixSymbol | None = None, **kwargs) -> str:
        """Typeset the CMI (and gradient/KKT if ``var``) to a standalone PDF; return its path."""
        from symbolic_dag.handoff import render_pdf

        return render_pdf(self, path, var=var, **kwargs)

    def check_gradient(self, var: sp.MatrixSymbol, dim: int, **kwargs) -> dict:
        """Verify the Wirtinger gradient against PyTorch autograd (``autograd == 2*grad``)."""
        from symbolic_dag.verify import check_gradient

        return check_gradient(self, var, dim, **kwargs)

    # ---- LaTeX hand-off ----------------------------------------------
    def to_latex(self, *, expand: bool = False, simplify: str | None = "normalize") -> str:
        """Render the CMI as LaTeX (see :mod:`symbolic_dag.latex`)."""
        from symbolic_dag.latex import cmi_to_latex

        return cmi_to_latex(self, expand=expand, simplify=simplify)

    def report(self, var: sp.MatrixSymbol | None = None, *, expand: bool = False) -> str:
        """LaTeX ``align*`` block: the CMI and (if ``var`` given) its gradient and KKT."""
        from symbolic_dag.latex import report

        return report(self, var, expand=expand)


@dataclass
class LogDetQuantity(RecursiveExpr):
    """A scalar information quantity ``sum_k w_k log det M_k + sum_j v_j tr(T_j) + c``.

    The lazy container shared by the non-CMI quantities (conditional entropy,
    total correlation, Gaussian KL divergence): weighted log-determinant terms,
    optional weighted trace terms, and a scalar constant (which may involve the
    symbolic dimension, e.g. ``n log(pi e)``). All matrices are kept block-free
    so the Wirtinger engine applies termwise.
    """

    logdet_terms: list[tuple[sp.Expr, MatrixExpr]] = field(default_factory=list)
    trace_terms: list[tuple[sp.Expr, MatrixExpr]] = field(default_factory=list)
    constant: sp.Expr = sp.Integer(0)

    # ---- scalar form -------------------------------------------------
    def to_expr(self) -> sp.Expr:
        """The quantity as a scalar sympy expression."""
        return sp.Add(
            *[w * sp.log(Determinant(M)) for w, M in self.logdet_terms],
            *[w * sp.Trace(M) for w, M in self.trace_terms],
            self.constant,
        )

    # ---- numeric evaluation -------------------------------------------
    def evaluate(self, subs: Mapping) -> float:
        """Numeric value (nats) at a concrete substitution (NumPy)."""
        val = sum(
            float(sp.sympify(w).subs(subs)) * _logdet(_to_numpy(M, subs))
            for w, M in self.logdet_terms
        )
        val += sum(
            float(sp.sympify(w).subs(subs)) * float(np.trace(_to_numpy(M, subs)).real)
            for w, M in self.trace_terms
        )
        return float(val + float(sp.sympify(self.constant).subs(subs)))

    def torch_value(self, subs: Mapping, dim: int):
        """The quantity as a differentiable real ``torch`` scalar (nats)."""
        import torch

        from symbolic_dag.verify import to_torch

        scalar_subs = {
            k: v for k, v in subs.items() if not isinstance(k, sp.MatrixSymbol)
        }

        def w_val(w):
            w = sp.sympify(w)
            return float(w.subs(scalar_subs) if scalar_subs else w)

        val = torch.zeros((), dtype=torch.float64)
        for w, M in self.logdet_terms:
            val = val + w_val(w) * torch.linalg.slogdet(to_torch(M, subs, dim))[1]
        for w, M in self.trace_terms:
            val = val + w_val(w) * torch.trace(to_torch(M, subs, dim)).real
        c = sp.sympify(self.constant)
        if scalar_subs:
            c = c.subs(scalar_subs)
        # any remaining free symbol in the constant is the (symbolic) dimension
        c = c.subs({s: dim for s in c.free_symbols})
        return val + float(c)

    # ---- symbolic operations ------------------------------------------
    def wirtinger_grad(self, var: sp.MatrixSymbol) -> MatrixExpr:
        """Closed-form Wirtinger gradient, termwise over log-det and trace terms."""
        from sympy import MatrixSymbol, ZeroMatrix

        from symbolic_dag.assumptions import apply_hermitian
        from symbolic_dag.matderiv import wirtinger_grad_logdet, wirtinger_grad_trace
        from symbolic_dag.rewrite import simplify_expr

        dF = MatrixSymbol("d" + var.name, *var.shape)
        G = ZeroMatrix(*var.shape)
        for w, M in self.logdet_terms:
            G = G + sp.sympify(w) * wirtinger_grad_logdet(M, var, dF)
        for w, M in self.trace_terms:
            G = G + sp.sympify(w) * wirtinger_grad_trace(M, var, dF)
        return simplify_expr(apply_hermitian(G.doit()), "normalize")

    def check_gradient(
        self, var: sp.MatrixSymbol, dim: int, *, seed: int = 0, atol: float = 1e-7
    ) -> dict:
        """Verify the gradient against PyTorch autograd (``autograd == 2 * grad``)."""
        import torch

        from symbolic_dag.assumptions import HermitianMatrix
        from symbolic_dag.verify import random_point_for_symbols, to_torch

        if isinstance(var, HermitianMatrix):
            raise NotImplementedError(
                f"check_gradient uses the Wirtinger convention (autograd == 2*grad), "
                f"which does not apply to the Hermitian variable {var.name!r}; use "
                "hermitian_grad_check (df = tr(G dQ)) instead."
            )
        syms = {
            a for M in [m for _, m in self.logdet_terms + self.trace_terms]
            for a in sp.preorder_traversal(M) if isinstance(a, sp.MatrixSymbol)
        }
        syms.add(var)
        subs = random_point_for_symbols(syms, dim, seed=seed, requires_grad=var)
        G = self.wirtinger_grad(var)
        val = self.torch_value(subs, dim)
        val.backward()
        subs_ng = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in subs.items()}
        err = float((subs[var].grad - 2.0 * to_torch(G, subs_ng, dim)).abs().max())
        return {"passed": bool(err < atol), "max_abs_err": err}
