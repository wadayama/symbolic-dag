"""Bridge to the actual numerical ``cmi-dag`` library, for cross-validation.

``cmi-dag`` is the sibling repository (not a PyPI dependency). To cross-check the
symbolic results against it we import it directly, adding its repository root to
``sys.path`` (``torch`` itself is a core dependency). When the repository is
absent the helpers report unavailability and the cross-check tests skip cleanly;
the rest of the suite still runs.

Set ``SYMBOLIC_DAG_CMIDAG_PATH`` to override the default sibling-repo location.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from functools import cache

import numpy as np

# Default: a ``cmi-dag`` checkout sitting next to this repository. Override with
# the ``SYMBOLIC_DAG_CMIDAG_PATH`` environment variable to point anywhere else.
# When neither resolves to a real directory the cross-check tests skip cleanly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CMIDAG = os.path.join(os.path.dirname(_REPO_ROOT), "cmi-dag")


@cache
def _load():
    """Import (torch, cmi_dag) or return None if unavailable."""
    root = os.environ.get("SYMBOLIC_DAG_CMIDAG_PATH", _DEFAULT_CMIDAG)
    try:
        import torch  # noqa: F401

        if root not in sys.path and os.path.isdir(root):
            sys.path.insert(0, root)
        import cmi_dag  # noqa: F401

        return torch, cmi_dag
    except Exception:
        return None


def cmidag_available() -> bool:
    return _load() is not None


def _t(M: np.ndarray):
    import torch

    return torch.tensor(np.asarray(M), dtype=torch.complex128)


def cmidag_cmi(
    num_nodes: int,
    roots: Sequence[int],
    parents: Mapping[int, Sequence[int]],
    edge_mats: Mapping[tuple[int, int], np.ndarray],
    root_covs: Mapping[int, np.ndarray],
    noise_covs: Mapping[int, np.ndarray],
    A: Sequence[int],
    B: Sequence[int],
    C: Sequence[int] = (),
) -> float:
    """CMI value from the actual cmi-dag (nats, complex)."""
    _, cmi_dag = _load()
    K = cmi_dag.compute_k_blocks_multiroot(
        num_nodes=num_nodes, roots=list(roots),
        parents={j: list(p) for j, p in parents.items()},
        edge_mats={k: _t(v) for k, v in edge_mats.items()},
        root_covs={k: _t(v) for k, v in root_covs.items()},
        noise_covs={k: _t(v) for k, v in noise_covs.items()},
    )
    return float(
        cmi_dag.conditional_mutual_information_from_k(
            K, A=list(A), B=list(B), C=list(C)
        ).item().real
    )


def cmidag_grad(
    num_nodes: int,
    roots: Sequence[int],
    parents: Mapping[int, Sequence[int]],
    static_edges: Mapping[tuple[int, int], np.ndarray],
    root_covs: Mapping[int, np.ndarray],
    noise_covs: Mapping[int, np.ndarray],
    A: Sequence[int],
    B: Sequence[int],
    C: Sequence[int],
    F_key: tuple[int, int],
    H_for_F: np.ndarray,
    F_value: np.ndarray,
) -> np.ndarray:
    """Autograd dI(A;B|C)/dF for a general gadget where edge ``F_key = H_for_F @ F``.

    Returns ``F.grad`` (PyTorch Wirtinger convention = 2x the symbolic gradient).
    """
    torch, cmi_dag = _load()
    F = torch.tensor(np.asarray(F_value), dtype=torch.complex128).clone().requires_grad_(True)
    em = {k: _t(v) for k, v in static_edges.items()}
    em[F_key] = _t(H_for_F) @ F
    K = cmi_dag.compute_k_blocks_multiroot(
        num_nodes=num_nodes, roots=list(roots),
        parents={j: list(p) for j, p in parents.items()}, edge_mats=em,
        root_covs={k: _t(v) for k, v in root_covs.items()},
        noise_covs={k: _t(v) for k, v in noise_covs.items()},
    )
    cmi_dag.conditional_mutual_information_from_k(
        K, A=list(A), B=list(B), C=list(C)
    ).backward()
    return F.grad.detach().numpy()


def cmidag_precoder_grad(
    H: np.ndarray, F: np.ndarray, Sigma0: np.ndarray, R: np.ndarray
) -> np.ndarray:
    """Autograd dI/dF for the precoder gadget Y=(HF)X0+X1+N, I(X0;Y|X1).

    Returns ``F.grad`` (PyTorch Wirtinger convention = 2x the symbolic gradient).
    """
    torch, cmi_dag = _load()
    d = H.shape[0]
    C = torch.complex128
    Iden = torch.eye(d, dtype=C)
    Ft = torch.tensor(np.asarray(F), dtype=C).clone().requires_grad_(True)
    K = cmi_dag.compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): _t(H) @ Ft, (2, 1): Iden},
        root_covs={0: _t(Sigma0), 1: Iden}, noise_covs={2: _t(R)},
    )
    val = cmi_dag.conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])
    val.backward()
    return Ft.grad.detach().numpy()
