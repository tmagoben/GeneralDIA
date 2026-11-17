"""Independent small, real, two-electron singlet oracle in orthonormalized AOs."""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np


def singlet_spectrum(overlap, hcore, eri, position, nuclear_repulsion):
    """Direct diagonalization on symmetric spatial products, independent of RHF/FCI RDMs.

    AO inputs: dimensionless overlap, one-/two-electron integrals (chemist notation)
    in hartree, and three position matrices in bohr. Off-diagonal transition dipoles
    have no nuclear contribution. The electronic matrices returned here are NOT
    total molecular permanent dipoles. Limited to two electrons and 2-8 real AOs.
    """
    raw = [np.asarray(x) for x in (overlap, hcore, eri, position)]
    if not all(t.dtype.kind in "iuf" and np.isfinite(t).all() for t in raw):
        raise ValueError("oracle requires finite real AO integrals")
    s, h, v, r = (t.astype(float, copy=False) for t in raw)
    n = s.shape[0] if s.ndim == 2 else 0
    if not 2 <= n <= 8 or s.shape != (n, n) or h.shape != s.shape:
        raise ValueError("expected 2-8 atomic orbitals in square overlap/core matrices")
    if v.shape != (n,) * 4 or r.shape != (3, n, n) or not np.isfinite(nuclear_repulsion):
        raise ValueError("invalid ERI, position matrices, or nuclear repulsion")
    if not all(np.allclose(m, m.T, atol=1e-12, rtol=1e-12) for m in (s, h, *r)):
        raise ValueError("overlap, core Hamiltonian, and position operators must be symmetric")
    ev, u = np.linalg.eigh(s)
    if ev.min() < 1e-9:
        raise ValueError("overlap is singular or ill-conditioned")
    x = (u * ev**-0.5) @ u.T
    h = x.T @ h @ x
    v = np.einsum("ap,bq,cr,ds,abcd->pqrs", x, x, x, x, v, optimize=True)
    r = x.T @ r @ x
    eye = np.eye(n)
    pairs = list(combinations_with_replacement(range(n), 2))
    sym = np.zeros((n, n, len(pairs)))
    for k, (a, b) in enumerate(pairs):
        sym[a, b, k] = sym[b, a, k] = 1.0 if a == b else 1 / np.sqrt(2.0)
    sym = sym.reshape(n * n, -1)
    # <pq|V|rs> = (pr|qs) for distinguishable spatial-product indices.
    h2 = np.kron(h, eye) + np.kron(eye, h) + v.transpose(0, 2, 1, 3).reshape(n * n, n * n)
    singlet_h = sym.T @ h2 @ sym
    if not np.allclose(singlet_h, singlet_h.T, atol=1e-11, rtol=1e-11):
        raise ValueError("two-electron Hamiltonian is not symmetric")
    energies, vec = np.linalg.eigh(singlet_h)
    states = sym @ vec
    operators = (np.kron(c, eye) + np.kron(eye, c) for c in r)
    dipoles = np.stack([-states.T @ operator @ states for operator in operators])
    return energies + nuclear_repulsion, dipoles
