"""Composite information objectives: weighted sums of CMIs.

Multi-terminal design problems are usually posed on a *linear combination* of
conditional mutual informations --- a sum rate, a weighted rate, a rate-region
facet (cf. ``cmi-dag``'s rate-function evaluator)

    f = sum_k  alpha_k * I(V_{A_k}; V_{B_k} | V_{C_k}).

:class:`CompositeCMI` carries the ``(weight, SymbolicCMI)`` terms and exposes the
same surface as a single CMI: closed-form Wirtinger gradient (by linearity),
numeric evaluation, PyTorch verification, and LaTeX. Directed information is the
special case with unit weights and a causal-conditioning pattern
(:func:`directed_information_from_k`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import sympy as sp
from sympy import MatrixExpr, MatrixSymbol

from symbolic_dag.expr import SymbolicCMI


def _weight_value(w, subs) -> float:
    """Evaluate a (possibly symbolic) scalar weight under scalar entries of ``subs``."""
    w = sp.sympify(w)
    scalar_subs = {
        k: v for k, v in (subs or {}).items() if not isinstance(k, MatrixSymbol)
    }
    val = w.subs(scalar_subs) if scalar_subs else w
    return float(val)


@dataclass
class CompositeCMI:
    """A weighted sum of CMIs ``f = sum_k w_k I(A_k; B_k | C_k)`` (lazy).

    Attributes:
        terms: list of ``(weight, SymbolicCMI)``; weights may be numbers or
            scalar ``sympy`` expressions (substituted at evaluation time).
        metadata: free-form annotations (e.g. ``{"form": "directed_information"}``).
    """

    terms: list[tuple[sp.Expr, SymbolicCMI]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # ---- numeric evaluation -------------------------------------------
    def evaluate(self, subs) -> float:
        """Numeric value (nats): ``sum_k w_k * I_k`` at a concrete substitution."""
        return float(
            sum(_weight_value(w, subs) * cmi.evaluate(subs) for w, cmi in self.terms)
        )

    def torch_value(self, subs, dim: int):
        """The objective as a differentiable ``torch`` scalar (see ``verify``)."""
        val = None
        for w, cmi in self.terms:
            t = _weight_value(w, subs) * cmi.torch_value(subs, dim)
            val = t if val is None else val + t
        return val

    # ---- symbolic operations ------------------------------------------
    def wirtinger_grad(self, var: MatrixSymbol) -> MatrixExpr:
        """Closed-form ``df/dvar^*`` by linearity: ``sum_k w_k * dI_k/dvar^*``."""
        from symbolic_dag.assumptions import apply_hermitian
        from symbolic_dag.rewrite import simplify_expr

        G = None
        for w, cmi in self.terms:
            term = sp.sympify(w) * cmi.wirtinger_grad(var)
            G = term if G is None else G + term
        if G is None:
            raise ValueError("CompositeCMI has no terms.")
        return simplify_expr(apply_hermitian(G.doit()), "normalize")

    def stationarity(self, var: MatrixSymbol) -> sp.Equality:
        """The KKT/stationarity condition ``df/dvar^* = 0``."""
        from sympy import Eq, ZeroMatrix

        return Eq(self.wirtinger_grad(var), ZeroMatrix(*var.shape))

    # ---- verification ---------------------------------------------------
    def check_gradient(
        self, var: MatrixSymbol, dim: int, *, seed: int = 0, atol: float = 1e-7
    ) -> dict:
        """Verify the gradient against PyTorch autograd (``autograd == 2 * grad``).

        Weights must be numeric for this check.
        """
        import torch

        from symbolic_dag.verify import (
            _matrix_symbols,
            random_point_for_symbols,
            to_torch,
        )

        syms = set()
        for _, cmi in self.terms:
            syms |= _matrix_symbols(cmi)
        subs = random_point_for_symbols(syms, dim, seed=seed, requires_grad=var)
        G = self.wirtinger_grad(var)
        val = self.torch_value(subs, dim)
        val.backward()
        subs_ng = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in subs.items()}
        err = float((subs[var].grad - 2.0 * to_torch(G, subs_ng, dim)).abs().max())
        return {"passed": bool(err < atol), "max_abs_err": err}

    # ---- hand-off --------------------------------------------------------
    def to_latex(self) -> str:
        """``f = w_1 I(...) + w_2 I(...) + ...`` (information sets only)."""
        from symbolic_dag.latex import cmi_to_latex

        parts = []
        for w, cmi in self.terms:
            lhs = cmi_to_latex(cmi).split(" = ")[0]
            w = sp.sympify(w)
            if w == 1:
                parts.append(lhs)
            else:
                parts.append(rf"{sp.latex(w)}\, {lhs}")
        return " + ".join(parts)


def composite_cmi(
    terms: Sequence[tuple[sp.Expr, SymbolicCMI]],
) -> CompositeCMI:
    """Build a weighted-sum objective ``sum_k w_k I(A_k; B_k | C_k)``.

    Args:
        terms: ``(weight, SymbolicCMI)`` pairs; weights are numbers or scalar
            ``sympy`` expressions.
    """
    terms = list(terms)
    if not terms:
        raise ValueError("composite_cmi needs at least one (weight, cmi) term.")
    for w, cmi in terms:
        if not isinstance(cmi, SymbolicCMI):
            raise TypeError(
                f"composite_cmi terms must be (weight, SymbolicCMI); got {type(cmi).__name__}."
            )
    return CompositeCMI(terms=terms, metadata={"form": "weighted_cmi_sum"})


def directed_information_from_k(
    K: dict[tuple[int, int], MatrixExpr],
    X_seq: Sequence[int],
    Y_seq: Sequence[int],
    C: Sequence[int] = (),
) -> CompositeCMI:
    """Directed information ``I(X^n -> Y^n | C)`` (Massey), as a sum of CMIs.

        I(X^n -> Y^n | C) = sum_{i=1}^{n} I(X^i ; Y_i | Y^{i-1}, C),

    where ``X^i = {X_1..X_i}`` and ``Y^{i-1} = {Y_1..Y_{i-1}}``. Each term is a
    genuine :class:`SymbolicCMI`, so the gradient/verification machinery applies
    termwise. ``X_seq`` and ``Y_seq`` are equal-length sequences of node indices
    in causal order.
    """
    from symbolic_dag.information import conditional_mutual_information_from_k

    X_seq, Y_seq, C = list(X_seq), list(Y_seq), list(C)
    if len(X_seq) != len(Y_seq) or not X_seq:
        raise ValueError(
            f"X_seq and Y_seq must be non-empty and of equal length; got "
            f"{len(X_seq)} and {len(Y_seq)}."
        )
    terms: list[tuple[sp.Expr, SymbolicCMI]] = []
    for i in range(len(Y_seq)):
        A = X_seq[: i + 1]
        cond = Y_seq[:i] + C
        terms.append(
            (sp.Integer(1),
             conditional_mutual_information_from_k(K, A=A, B=[Y_seq[i]], C=cond))
        )
    return CompositeCMI(terms=terms, metadata={"form": "directed_information", "K": K})
