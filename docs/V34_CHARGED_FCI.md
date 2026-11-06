# v3.4 follow-on: charged closed-shell singlet FCI references (H3+ preparation)

**Scope:** add a small charged molecular electronic-structure oracle to support
a future *genuine molecular conical-intersection* benchmark. The present
slice does not claim to have measured H3+ Berry phase, conical gap or NACs.

## Contract and conventions

`FCIDipoleBackend(..., charge=q)` now accepts an explicitly integral molecular
charge in electron-charge units (default zero, backwards compatible).
The effective electron count is `N_e = sum(Z) - q`. It must be a positive,
even integer for the existing RHF + spin-adapted singlet FCI adapter; neither
open-shell nor odd-electron systems are admitted. The computed Hamiltonian
still includes nuclear repulsion and the same singlet FCI root ordering.

Dipoles are total molecular dipoles in e·bohr and retain a fixed laboratory
origin of (0,0,0) bohr. For a rigid **active translation** of all molecular
coordinates by a vector \(\Delta\mathbf R\), the charge-specific law is

\[
\boldsymbol\mu(R+\Delta R) =
\boldsymbol\mu(R) + q\,\Delta\mathbf R_{\rm bohr}\,I.
\]

Consequently, the neutral reference is translation invariant, but the ionic
**diagonal dipoles** shift by the total charge times displacement; transition
dipoles do not acquire an identity shift. The existing neutral
`SharedMolecularOperators` dipole model remains neutral-only, intentionally;
this change is solely an electronic-structure reference adapter extension.

Each cross-geometry overlap continues to use physical cross-AO integrals,
the orbital overlap `C_L^T S_AO(L,R) C_R`, and determinants/FCI coefficient
contractions. The new H3+ test independently reconstructs one-alpha,
one-beta two-electron overlaps as
`einsum('ab,ac,bd,cd->', bra.conj(), S, S, ket)`.
It compares magnitudes since independent FCI diagonalizations may choose
different signs for the real electronic roots.

## Molecular control

Neutral H3 (three electrons) is rejected under this singlet contract, whereas
H3+ (three protons, charge +1, two electrons) is supported. Charge metadata,
RHF/FCI electronic energies, dipoles, physical overlaps and positive-electron
input checks are tested. The initial geometries are **intentionally slightly
asymmetric** to avoid ambiguous degenerate root labels in translation and
overlap tests.

## Next scientific gate

At the equilateral D3h geometry, the first two H3+ *excited singlets* form
an E' pair and symmetry-breaking in-plane modes can produce a molecular
CI. That result must still be reproduced in a chosen finite basis and
geometry in a separate benchmark, including root isolation, physical cyclic
overlaps, analytic symmetry controls, two-dimensional gap splitting and
mesh/radius dependence. No CI claim follows merely from the charged adapter.

Independent starting literature:

- Physical Chemistry Chemical Physics (2024), DOI 10.1039/D4CP00391H,
  Section 3.1: H3+ electronic E' degenerate excited singlets.
- J. Chem. Phys. 141, 204306 (2014), DOI 10.1063/1.4901986:
  singlet H3+ CI surfaces and nonadiabatic coupling context.

## Reproduce

```bash
python -m pip install -e ".[dev,pyscf]"
pytest tests/test_fci_charged.py tests/test_fci_dipoles.py tests/test_fci_loop.py -q
```

The dedicated PySCF job executes these tests without dependency skips; the
regular core suite may skip PySCF tests when that optional backend is absent.

Further milestones are tracked in issue #9.
