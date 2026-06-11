"""(internal, experimental) Fisher information / Cramer-Rao bound for ISAC.

NOT part of the public API: it is intentionally **not** exported from
``symbolic_dag`` and **not** described in the README, so the public concept stays
"symbolic conditional mutual information for linear Gaussian DAGs". It is the
*sensing* dual of that comms layer --- both the CMI/MMSE and the Fisher/CRB are
log-det / trace / inverse of a linear-Gaussian covariance --- and is meant for
**ISAC** (joint communication + sensing) waveform design: reproduce, *from the
DAG model*, the sensing closed forms (Slepian-Bangs FIM, Cramer-Rao / posterior
CRB, and the design gradient ``dCRB/dF*``) that ISAC papers derive by hand.

It reuses the existing engine: covariances come from the K-recursion, the design
gradient reuses :func:`symbolic_dag.matderiv.trace_grad`, and numeric checks reuse
:func:`symbolic_dag.verify.to_torch`. Import explicitly::

    from symbolic_dag._fisher import (
        fisher_information_matrix, cramer_rao_bound, crb_trace, crb_grad,
    )

See ``docs/internal/fisher-crb.md`` for the spec and worked ISAC example.
"""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp
from sympy import (
    Abs,
    Adjoint,
    Inverse,
    MatrixExpr,
    MatrixSymbol,
    Trace,
    ZeroMatrix,
    conjugate,
    re,
)

from symbolic_dag.assumptions import HermitianMatrix, apply_hermitian
from symbolic_dag.matderiv import trace_grad
from symbolic_dag.rewrite import simplify_expr
from symbolic_dag.verify import to_torch


# ----------------------------------------------------------------------
# Slepian-Bangs Fisher information
# ----------------------------------------------------------------------
def fisher_information_matrix(
    R: MatrixExpr,
    dR: Sequence[MatrixExpr] = (),
    dmu: Sequence[MatrixExpr] = (),
    *,
    prior: MatrixExpr | None = None,
) -> sp.Matrix:
    """Slepian-Bangs FIM ``J`` (``P x P`` sympy Matrix) for ``y ~ CN(mu(t), R(t))``.

    ``J[i,j] = tr(R^-1 dR_i R^-1 dR_j) + (dmu_i^H R^-1 dmu_j + dmu_j^H R^-1 dmu_i)``
    (the second, mean term is the Hermitian ``2 Re{...}``), optionally plus an
    additive ``prior`` FIM (Bayesian / posterior CRB).

    Args:
        R: the observation covariance (a ``MatrixExpr``, e.g. from the K-recursion).
        dR: the covariance derivatives ``(dR/dt_1, ..., dR/dt_P)``.
        dmu: the mean derivatives ``(dmu/dt_1, ..., dmu/dt_P)`` (column vectors).
        prior: optional ``P x P`` additive prior information matrix.
    """
    Rinv = Inverse(R)
    P = max(len(dR), len(dmu))
    if P == 0:
        raise ValueError("provide dR and/or dmu (the parameter derivatives).")

    def entry(i: int, j: int):
        t = sp.Integer(0)
        if dR:
            t += Trace(Rinv * dR[i] * Rinv * dR[j])
        if dmu:
            t += Trace(Adjoint(dmu[i]) * Rinv * dmu[j])
            t += Trace(Adjoint(dmu[j]) * Rinv * dmu[i])
        return t

    J = sp.Matrix(P, P, entry)
    if prior is not None:
        J = J + sp.Matrix(prior)
    return J


# ----------------------------------------------------------------------
# Cramer-Rao bound
# ----------------------------------------------------------------------
def _inverse_with_opaque_traces(J: sp.Matrix) -> sp.Matrix:
    """``J^-1`` with each ``Trace(...)`` held as an opaque scalar symbol.

    ``sympy``'s matrix inverse tries to ``evalf`` pivots, which fails on ``Trace``
    atoms; scalarising them first (and substituting back) keeps it purely symbolic.
    """
    J = sp.Matrix(J)
    traces = set().union(*(e.atoms(Trace) for e in J)) if len(J) else set()
    syms = {T: sp.Dummy() for T in traces}
    back = {s: T for T, s in syms.items()}
    Jinv = J.applyfunc(lambda e: e.xreplace(syms)).inv()
    return Jinv.applyfunc(lambda e: sp.together(e).xreplace(back))


def cramer_rao_bound(J: sp.Matrix, indices=None):
    """``CRB = J^-1`` (full matrix), or its ``(indices, indices)`` component/block.

    A single integer ``indices=i`` returns ``(J^-1)[i,i]`` --- the CRB on parameter
    ``i`` with the others as nuisances (the Schur-complement / posterior-CRB form).
    """
    Jinv = _inverse_with_opaque_traces(J)
    if indices is None:
        return Jinv
    if isinstance(indices, int):
        return Jinv[indices, indices]
    idx = list(indices)
    return Jinv[idx, idx]


def crb_trace(J: sp.Matrix, indices: Sequence[int] | None = None):
    """``tr(J^-1)`` over all parameters, or over a subset ``indices``."""
    Jinv = _inverse_with_opaque_traces(J)
    if indices is None:
        return sum(Jinv[i, i] for i in range(Jinv.rows))
    return sum(Jinv[i, i] for i in indices)


# ----------------------------------------------------------------------
# Design gradient  dCRB/dF*  (reuses trace_grad)
# ----------------------------------------------------------------------
def _expand_conj_traces(e):
    """Rewrite ``|Trace|^2`` and ``conjugate(Trace)`` into bare ``Trace`` atoms.

    ``conjugate(tr(M)) = tr(M^H)`` and ``|tr(M)|^2 = tr(M) tr(M^H)``, so all the
    ``F``-dependence ends up inside ``Trace`` atoms (which ``trace_grad`` handles).
    """
    e = sp.sympify(e)
    e = e.replace(
        lambda x: x.is_Pow and isinstance(x.base, Abs) and isinstance(x.base.args[0], Trace) and x.exp == 2,
        lambda x: x.base.args[0] * Trace(Adjoint(x.base.args[0].arg)),
    )
    e = e.replace(
        lambda x: isinstance(x, conjugate) and isinstance(x.args[0], Trace),
        lambda x: Trace(Adjoint(x.args[0].arg)),
    )
    return e


def crb_grad(metric, var: MatrixSymbol) -> MatrixExpr:
    """Wirtinger gradient ``d(metric)/dvar*`` of a scalar CRB metric.

    ``metric`` is any scalar built from ``Trace(A_k R_X(var))`` atoms (e.g.
    ``cramer_rao_bound(J, i)`` or ``crb_trace(J)``). By the chain rule
    ``d(metric)/dvar* = sum_k (d metric / d t_k) * d t_k/dvar*`` with
    ``t_k = Trace(...)``, ``d t_k/dvar* = trace_grad(...)`` and ``d metric/d t_k`` a
    scalar derivative. A numerical library's autograd returns ``2 *`` this.
    """
    metric = _expand_conj_traces(metric)
    traces = [T for T in metric.atoms(Trace) if T.has(var)]
    syms = {T: sp.Dummy() for T in traces}
    scalar = metric.xreplace(syms)
    back = {s: T for T, s in syms.items()}
    G = None
    for T, s in syms.items():
        gk = trace_grad(apply_hermitian(T.arg), var)          # d Trace(M)/dvar*
        coeff = sp.diff(scalar, s).xreplace(back)              # d metric / d t_k
        term = coeff * gk
        G = term if G is None else G + term
    if G is None:
        return ZeroMatrix(*var.shape)
    return simplify_expr(apply_hermitian(G.doit()), "normalize")


# ----------------------------------------------------------------------
# Numeric verification (self-contained; reuses to_torch for the matrix parts)
# ----------------------------------------------------------------------
def _to_torch_scalar(e, subs, dim):
    """Evaluate a scalar expression over ``Trace`` atoms to a (differentiable) torch scalar."""
    import torch

    e = sp.sympify(e)
    if isinstance(e, Trace):
        return torch.trace(to_torch(e.arg, subs, dim))
    if isinstance(e, conjugate):
        return _to_torch_scalar(e.args[0], subs, dim).conj()
    if isinstance(e, Abs):
        return _to_torch_scalar(e.args[0], subs, dim).abs()
    if isinstance(e, re):
        return _to_torch_scalar(e.args[0], subs, dim).real
    if e.is_Add:
        out = None
        for a in e.args:
            t = _to_torch_scalar(a, subs, dim)
            out = t if out is None else out + t
        return out
    if e.is_Mul:
        out = None
        for a in e.args:
            t = _to_torch_scalar(a, subs, dim)
            out = t if out is None else out * t
        return out
    if e.is_Pow:
        base = _to_torch_scalar(e.base, subs, dim)
        ex = e.exp
        if ex == -1:
            return 1 / base
        return base ** (int(ex) if ex.is_Integer else float(ex))
    if e.is_number:
        return torch.tensor(complex(e), dtype=torch.complex128)
    raise TypeError(f"_to_torch_scalar: unsupported {type(e).__name__}: {e}")


def _eval_grad_matrix(G, subs, dim):
    """Evaluate a matrix expression whose scalar coefficients may contain ``Trace``."""
    from sympy import MatAdd, MatMul

    G = G.doit()
    if isinstance(G, MatAdd):
        out = None
        for a in G.args:
            t = _eval_grad_matrix(a, subs, dim)
            out = t if out is None else out + t
        return out
    if isinstance(G, MatMul):
        mats = [a for a in G.args if isinstance(a, MatrixExpr)]
        scals = [a for a in G.args if not isinstance(a, MatrixExpr)]
        out = None
        for a in mats:
            t = to_torch(a, subs, dim)
            out = t if out is None else out @ t
        for sc in scals:
            out = _to_torch_scalar(sc, subs, dim) * out
        return out
    return to_torch(G, subs, dim)


def _random_point(symbols, dim, seed, requires_grad=None):
    import torch

    g = torch.Generator().manual_seed(seed)
    C = torch.complex128

    def rc():
        return torch.complex(
            torch.randn(dim, dim, dtype=torch.float64, generator=g),
            torch.randn(dim, dim, dtype=torch.float64, generator=g),
        )

    def hpd():
        A = rc()
        return A @ A.mH + dim * torch.eye(dim, dtype=C)

    subs = {}
    for s in symbols:
        t = hpd() if isinstance(s, HermitianMatrix) else rc()
        if requires_grad is not None and s == requires_grad:
            t = t.clone().requires_grad_(True)
        subs[s] = t
    return subs


def crb_value(metric, subs, dim) -> float:
    """Numeric value of a (real) scalar CRB metric at ``subs`` (torch)."""
    return float(_to_torch_scalar(metric, subs, dim).real)


def crb_grad_check(metric, var: MatrixSymbol, dim: int, *, seed: int = 0, atol: float = 1e-6) -> dict:
    """Check ``crb_grad`` against PyTorch autograd (``autograd == 2 * crb_grad``)."""
    import torch

    syms = {a for a in sp.preorder_traversal(sp.sympify(metric)) if isinstance(a, MatrixSymbol)}
    syms.add(var)
    subs = _random_point(syms, dim, seed, requires_grad=var)
    G = crb_grad(metric, var)
    m = _to_torch_scalar(metric, subs, dim).real
    m.backward()
    subs_ng = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in subs.items()}
    err = float((subs[var].grad - 2.0 * _eval_grad_matrix(G, subs_ng, dim)).abs().max())
    return {"passed": bool(err < atol), "max_abs_err": err}
