"""PyTorch-based numerical verification of symbolic results.

The numerical sibling ``cmi-dag`` is built on PyTorch, so the user-facing way to
*numerically check* a symbolic result is aligned with PyTorch here too. The key
primitive is :func:`to_torch`, which lowers a ``sympy`` matrix expression to a
``torch`` tensor; on top of it the symbolic CMI becomes an ordinary
**differentiable** ``torch`` scalar, so:

- :func:`SymbolicCMI.torch_value` evaluates the CMI numerically (complex128,
  nats, no one-half factor --- the cmi-dag convention);
- :meth:`SymbolicCMI.check` cross-checks that value against an independent
  Schur-complement computation from the same K-blocks;
- :meth:`SymbolicCMI.check_gradient` checks the closed-form Wirtinger gradient
  against **PyTorch autograd** of the very same CMI (autograd returns twice the
  Wirtinger gradient --- the same factor cmi-dag uses).

PyTorch is a core dependency, so these helpers work out of the box after
``uv sync``. (An independent, torch-free :func:`symbolic_dag.numpy_cmi` oracle is
also available, and is what the core test suite uses for an extra cross-check.)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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
    ZeroMatrix,
)

from symbolic_dag.assumptions import HermitianMatrix


def _torch():
    import torch  # core dependency

    return torch


# ----------------------------------------------------------------------
# sympy matrix expression -> torch tensor
# ----------------------------------------------------------------------
def to_torch(e: MatrixExpr, subs: Mapping[MatrixSymbol, "object"], dim: int):
    """Evaluate a ``sympy`` matrix expression to a complex128 ``torch`` tensor.

    Args:
        e: A matrix expression over ``MatrixSymbol`` leaves (products, sums,
            ``Inverse``, ``Adjoint``, ``Identity``, ``ZeroMatrix``,
            ``BlockMatrix``).
        subs: Maps each matrix symbol to a ``torch`` tensor (complex128). A leaf
            tensor may carry ``requires_grad=True`` to enable autograd.
        dim: Concrete value of the (possibly symbolic) matrix dimension, used to
            materialise ``Identity`` / ``ZeroMatrix``.

    Returns:
        A ``torch`` tensor; differentiable through any ``requires_grad`` leaf.
    """
    torch = _torch()
    C = torch.complex128
    if isinstance(e, (HermitianMatrix, MatrixSymbol)):
        return subs[e]
    if isinstance(e, Identity):
        return torch.eye(dim, dtype=C)
    if isinstance(e, ZeroMatrix):
        r = e.shape[0] if isinstance(e.shape[0], int) else dim
        c = e.shape[1] if isinstance(e.shape[1], int) else dim
        return torch.zeros(r, c, dtype=C)
    if isinstance(e, Adjoint):
        return to_torch(e.arg, subs, dim).mH
    if isinstance(e, Inverse):
        return torch.linalg.inv(to_torch(e.arg, subs, dim))
    if isinstance(e, MatPow):
        base = to_torch(e.base, subs, dim)
        p = int(e.exp)
        if p < 0:
            base, p = torch.linalg.inv(base), -p
        return torch.linalg.matrix_power(base, p)
    if isinstance(e, MatAdd):
        out = None
        for a in e.args:
            t = to_torch(a, subs, dim)
            out = t if out is None else out + t
        return out
    if isinstance(e, MatMul):
        coeff, mm = e.as_coeff_mmul()
        out = None
        for a in mm.args:
            t = to_torch(a, subs, dim)
            out = t if out is None else out @ t
        c = complex(coeff)
        return out if c == 1 else c * out
    if isinstance(e, BlockMatrix):
        rows = [
            torch.cat([to_torch(b, subs, dim) for b in e.blocks.row(i)], dim=1)
            for i in range(e.blockshape[0])
        ]
        return torch.cat(rows, dim=0)
    raise TypeError(f"to_torch: unsupported node {type(e).__name__}")


# ----------------------------------------------------------------------
# random points and independent torch CMI
# ----------------------------------------------------------------------
def _matrix_symbols(cmi) -> set[MatrixSymbol]:
    syms: set[MatrixSymbol] = set()
    for M in cmi.metadata["K"].values():
        syms |= {a for a in sp.preorder_traversal(M) if isinstance(a, MatrixSymbol)}
    return syms


def random_torch_point(
    cmi, dim: int, *, seed: int = 0, requires_grad: MatrixSymbol | None = None
) -> dict:
    """Random complex128 tensors for every symbol of ``cmi`` (covariances HPD).

    ``HermitianMatrix`` symbols get a random Hermitian positive-definite matrix;
    other matrix symbols get a random complex matrix. If ``requires_grad`` is a
    symbol, that tensor is made a differentiable leaf.
    """
    torch = _torch()
    C = torch.complex128
    g = torch.Generator().manual_seed(seed)

    def rc():
        return torch.complex(
            torch.randn(dim, dim, dtype=torch.float64, generator=g),
            torch.randn(dim, dim, dtype=torch.float64, generator=g),
        )

    def hpd():
        A = rc()
        return A @ A.mH + dim * torch.eye(dim, dtype=C)

    subs = {}
    for s in _matrix_symbols(cmi):
        t = hpd() if isinstance(s, HermitianMatrix) else rc()
        if requires_grad is not None and s == requires_grad:
            t = t.clone().requires_grad_(True)
        subs[s] = t
    return subs


def _k_to_torch(cmi, subs, dim) -> dict[tuple[int, int], "object"]:
    return {key: to_torch(M, subs, dim) for key, M in cmi.metadata["K"].items()}


def _torch_cmi_from_k(K, A, B, C):
    """Independent CMI from torch K-blocks (Schur complement + slogdet)."""
    torch = _torch()

    def getK(a, b):
        return K[(a, b)] if a >= b else K[(b, a)].mH

    def assemble(rows, cols):
        return torch.cat(
            [torch.cat([getK(r, c) for c in cols], dim=1) for r in rows], dim=0
        )

    def cond(U, Cc):
        U, Cc = sorted(U), sorted(Cc)
        Kuu = assemble(U, U)
        if not Cc:
            return Kuu
        return Kuu - assemble(U, Cc) @ torch.linalg.solve(
            assemble(Cc, Cc), assemble(Cc, U)
        )

    sld = lambda M: torch.linalg.slogdet(M)[1]
    A, B, C = sorted(A), sorted(B), sorted(C)
    return sld(cond(A, C)) + sld(cond(B, C)) - sld(cond(sorted(A + B), C))


# ----------------------------------------------------------------------
# user-facing checks
# ----------------------------------------------------------------------
def torch_value(cmi, subs: Mapping, dim: int):
    """The CMI as a differentiable real ``torch`` scalar (nats)."""
    torch = _torch()
    val = torch.zeros((), dtype=torch.float64)
    for sign, M in cmi.logdet_terms:
        val = val + sign * torch.linalg.slogdet(to_torch(M, subs, dim))[1]
    return val


def check(cmi, dim: int, *, seed: int = 0, samples: int = 4, atol: float = 1e-8) -> dict:
    """Numerically verify the CMI value at random points (PyTorch).

    Compares :func:`torch_value` (from the symbolic log-det terms) against an
    independent Schur-complement CMI computed from the same K-blocks. Returns
    ``{"passed": bool, "max_abs_err": float, "samples": int}``.
    """
    max_err = 0.0
    for s in range(samples):
        subs = random_torch_point(cmi, dim, seed=seed + s)
        v = float(torch_value(cmi, subs, dim).real)
        ref = float(_torch_cmi_from_k(_k_to_torch(cmi, subs, dim),
                                      list(cmi.A), list(cmi.B), list(cmi.C)).real)
        max_err = max(max_err, abs(v - ref))
    return {"passed": bool(max_err < atol), "max_abs_err": max_err, "samples": samples}


def check_gradient(
    cmi, var: MatrixSymbol, dim: int, *, seed: int = 0, atol: float = 1e-7
) -> dict:
    """Verify the closed-form Wirtinger gradient against PyTorch autograd.

    Evaluates the symbolic gradient ``dI/dvar^*`` at a random point and checks
    ``autograd == 2 * symbolic_gradient`` (PyTorch's Wirtinger convention).
    Returns ``{"passed": bool, "max_abs_err": float}``.
    """
    torch = _torch()
    G = cmi.wirtinger_grad(var)
    subs = random_torch_point(cmi, dim, seed=seed, requires_grad=var)
    val = torch_value(cmi, subs, dim)
    val.backward()
    autograd = subs[var].grad
    subs_ng = {k: (v.detach() if hasattr(v, "detach") else v) for k, v in subs.items()}
    Gn = to_torch(G, subs_ng, dim)
    err = float((autograd - 2.0 * Gn).abs().max())
    return {"passed": bool(err < atol), "max_abs_err": err}
