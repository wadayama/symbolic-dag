"""Multi-node MAC CMI: the sum-rate facet I(X0, X1; Y) and the chain rule.

The 2-user multiple-access channel (MAC) is the smallest multi-root gadget:
roots X0, X1 (independent) and a shared receiver Y. This example builds the
sum-rate facet I(X0, X1; Y) -- a multi-node information set -- verifies the value
with the one-call PyTorch check, and checks the chain rule
I(X0,X1;Y) = I(X0;Y|X1) + I(X1;Y) numerically.

Run with:  uv run python examples/mac_cmi.py
"""

from __future__ import annotations

import sympy as sp
from sympy import MatrixSymbol

from symbolic_dag import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
    hermitian,
    random_torch_point,
)

d = sp.Symbol("d", positive=True, integer=True)
A, B = MatrixSymbol("A", d, d), MatrixSymbol("B", d, d)
SX, SY, SZ = (hermitian(s, d) for s in ("Sigma_X", "Sigma_Y", "Sigma_Z"))


def build():
    # roots X0, X1; Y = A X0 + B X1 + N
    return compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): A, (2, 1): B},
        root_covs={0: SX, 1: SY}, noise_covs={2: SZ},
    )


def main() -> None:
    K = build()
    I_sum = conditional_mutual_information_from_k(K, A=[0, 1], B=[2], C=[])  # I(X0,X1;Y)
    print("Sum-rate facet  I(X0, X1; Y)  log-det terms (lazy form):")
    for sign, M in I_sum.logdet_terms:
        head = str(M).splitlines()[0]
        print(f"   {'+' if sign > 0 else '-'} log det  {head}{' ...' if chr(10) in str(M) else ''}")

    # one-call PyTorch verification of the (multi-node) value
    print(f"\nvalue check (PyTorch): {I_sum.check(dim=2)}")

    # chain rule, at a shared random torch point: I(X0,X1;Y) == I(X0;Y|X1) + I(X1;Y)
    dim = 2
    pt = random_torch_point(I_sum, dim, seed=0)
    v_sum = float(I_sum.torch_value(pt, dim).real)
    v_0g1 = float(conditional_mutual_information_from_k(K, [0], [2], [1]).torch_value(pt, dim).real)
    v_1 = float(conditional_mutual_information_from_k(K, [1], [2], []).torch_value(pt, dim).real)
    print(f"chain rule  I(X0,X1;Y) == I(X0;Y|X1) + I(X1;Y):  {abs(v_sum - (v_0g1 + v_1)):.2e}")


if __name__ == "__main__":
    main()
