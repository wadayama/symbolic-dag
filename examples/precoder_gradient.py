"""Closed-form Wirtinger gradient / KKT of a MIMO precoder CMI.

Derives, symbolically, the gradient of ``I(X0; Y | X1)`` for the precoder gadget
``Y = (H F) X0 + X1 + N`` with respect to the precoder ``F`` --- the kind of
gradient usually worked out by hand --- and checks it against PyTorch autograd
(which returns 2x the Wirtinger gradient) in one call.

Run with:  uv run python examples/precoder_gradient.py
"""

from __future__ import annotations

import sympy as sp
from sympy import Identity, MatrixSymbol

from symbolic_dag import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
    hermitian,
)

d = sp.Symbol("d", positive=True, integer=True)
H, F = MatrixSymbol("H", d, d), MatrixSymbol("F", d, d)
S0, R = hermitian("Sigma_0", d), hermitian("R", d)


def build():
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): H * F, (2, 1): Identity(d)},
        root_covs={0: S0, 1: Identity(d)}, noise_covs={2: R},
    )
    return conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])


def main() -> None:
    I = build()
    G = I.wirtinger_grad(F)
    print("Closed-form Wirtinger gradient  dI/dF^*  (derived symbolically):")
    print("   ", G)
    print("\nStationarity (optimal-precoder condition):")
    print("   ", I.stationarity(F))

    # numeric cross-check: the closed-form gradient vs PyTorch autograd, in one call
    # (autograd returns 2 x the Wirtinger gradient; .check_gradient handles the factor)
    print(f"\ngradient vs PyTorch autograd: {I.check_gradient(F, dim=3)}")
    print(f"value check:                  {I.check(dim=3)}")


if __name__ == "__main__":
    main()
