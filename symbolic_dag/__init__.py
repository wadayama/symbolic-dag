"""symbolic-dag: symbolic conditional mutual information for linear Gaussian DAGs.

The symbolic sibling of ``cmi-dag``. The primary API mirrors ``cmi-dag``'s
index-based functional surface (``compute_k_blocks_multiroot``,
``conditional_mutual_information_from_k``) but is complex-symbolic: gains and
covariances are ``sympy`` matrix expressions, and the CMI is returned as a lazy
:class:`SymbolicCMI` that can be simplified (the rewrite engine), differentiated
(the Wirtinger engine), and evaluated/cross-checked numerically. A thin
named-node builder :class:`GaussianDAG` is provided for readability.

Conventions match ``cmi-dag``: complex (Wirtinger), nats, no one-half factor;
``^H`` is ``sympy.Adjoint``. A numerical library's autograd returns twice the
Wirtinger gradient produced here.
"""

from __future__ import annotations

from symbolic_dag.assumptions import HermitianMatrix, hermitian
from symbolic_dag.builder import GaussianDAG
from symbolic_dag.expr import RecursiveExpr, SymbolicCMI
from symbolic_dag.information import (
    conditional_covariance,
    conditional_covariance_seq,
    conditional_mutual_information_from_k,
    mmse_error_covariance,
)
from symbolic_dag.krecursion import compute_k_blocks_multiroot, get_K, hermitianize
from symbolic_dag.latex import cmi_to_latex, report
from symbolic_dag.matderiv import (
    trace_grad,
    wirtinger_grad_cmi,
    wirtinger_grad_logdet,
    wirtinger_grad_trace,
)
from symbolic_dag.numeric import numpy_cmi, numpy_k_blocks
from symbolic_dag.rewrite import proves_zero, simplify_expr
from symbolic_dag.verify import random_torch_point, to_torch

__all__ = [
    "GaussianDAG",
    "HermitianMatrix",
    "RecursiveExpr",
    "SymbolicCMI",
    "cmi_to_latex",
    "compute_k_blocks_multiroot",
    "conditional_covariance",
    "conditional_covariance_seq",
    "conditional_mutual_information_from_k",
    "get_K",
    "report",
    "hermitian",
    "hermitianize",
    "mmse_error_covariance",
    "numpy_cmi",
    "numpy_k_blocks",
    "proves_zero",
    "random_torch_point",
    "simplify_expr",
    "to_torch",
    "trace_grad",
    "wirtinger_grad_cmi",
    "wirtinger_grad_logdet",
    "wirtinger_grad_trace",
]
