# symbolic-dag tutorials

A four-part walkthrough of the library, from a first symbolic conditional MI to
closed-form Wirtinger gradients cross-checked against the numerical `cmi-dag`.

| # | Topic | File |
| --- | --- | --- |
| 1 | Installation and your first symbolic CMI | [`tutorial-1-installation-and-first-cmi.md`](tutorial-1-installation-and-first-cmi.md) |
| 2 | Proving conditional independence (the rewrite engine) | [`tutorial-2-proving-conditional-independence.md`](tutorial-2-proving-conditional-independence.md) |
| 3 | Closed-form Wirtinger gradients and KKT | [`tutorial-3-wirtinger-gradients-and-kkt.md`](tutorial-3-wirtinger-gradients-and-kkt.md) |
| 4 | The builder and cross-validation against cmi-dag | [`tutorial-4-builder-and-cmidag-crosscheck.md`](tutorial-4-builder-and-cmidag-crosscheck.md) |

Each tutorial is self-contained and includes runnable code snippets. The scripts
under `../examples/` are the polished end-to-end versions.

The primary API deliberately mirrors the numerical sibling
[`cmi-dag`](https://github.com/wadayama/cmi-dag): the same
`compute_k_blocks_multiroot` / `conditional_mutual_information_from_k` call
pattern, the same index conventions. If you already use cmi-dag, your DAG-building
code transfers directly; the difference is that gains and covariances are `sympy`
matrix expressions and the CMI comes back as a symbolic object you can simplify,
differentiate, and prove things about.
