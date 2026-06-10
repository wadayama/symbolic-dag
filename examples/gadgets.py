"""Canonical gadgets: build, prove d-separation, read the symbolic CMI.

Run with:  uv run python examples/gadgets.py
"""

from __future__ import annotations

import sympy as sp
from sympy import MatrixSymbol

from symbolic_dag import GaussianDAG, hermitian

d = sp.Symbol("d", positive=True, integer=True)
A = MatrixSymbol("A", d, d)
B = MatrixSymbol("B", d, d)
SX, SY, SZ = (hermitian(s, d) for s in ("Sigma_X", "Sigma_Y", "Sigma_Z"))


def chain():
    G = GaussianDAG()
    G.add_source("X", cov=SX)
    G.add_node("Y", parents={"X": A}, noise=SY)
    G.add_node("Z", parents={"Y": B}, noise=SZ)
    return G


def collider():
    G = GaussianDAG()
    G.add_source("X", cov=SX)
    G.add_source("Y", cov=SY)
    G.add_node("Z", parents={"X": A, "Y": B}, noise=SZ)
    return G


def main() -> None:
    print("=== chain  X -> Y -> Z ===")
    G = chain()
    I_cond = G.cmi(A=["X"], B=["Z"], C=["Y"])  # I(X;Z|Y)
    print("I(X;Z|Y) conditionally independent (proved):",
          I_cond.is_conditionally_independent())
    I_marg = G.cmi(A=["X"], B=["Z"], C=[])      # I(X;Z)
    print("I(X;Z) log-det terms:")
    for sign, M in I_marg.logdet_terms:
        print(f"   {'+' if sign > 0 else '-'} log det  {M}")

    print("\n=== collider  X -> Z <- Y ===")
    Gc = collider()
    print("I(X;Y) marginally independent (proved):",
          Gc.cmi(["X"], ["Y"], []).is_conditionally_independent())
    print("I(X;Y|Z) independent? (opened by collider):",
          Gc.cmi(["X"], ["Y"], ["Z"]).is_conditionally_independent())


if __name__ == "__main__":
    main()
