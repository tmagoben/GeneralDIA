# v3.4 slice 2 — derivative-coupling numerator certification

**Scope:** independent numerical derivative consistency with an optional reference
operator. This is not molecular nonadiabatic-coupling or trajectory certification.

## Scientific and numerical contract

For nondegenerate states in one declared fixed finite-state basis,

\[
N^a_{ij}(R)=\langle\phi_i(R)|\partial_{R_a}H(R)|\phi_j(R)\rangle,
\qquad d^a_{ij}(R)=N^a_{ij}(R)/(E_j-E_i).
\]

The numerator can remain finite as a gap closes, while division becomes
singular or electronic state identities become ambiguous. At exact degeneracy,
the numerator *matrix* is gauge covariant but individual entries depend on
the electronic frame. Unresolved state pairs are flagged rather than silently
assigned a zero-valued physical coupling.

**certify_coupling_numerators(model, atomic_numbers, positions, ...)** compares
two independent routes to the Hamiltonian derivative:

- PyTorch autograd of the complete real-symmetric or complex-Hermitian matrix.
- Centered five-point finite differences of Hamiltonian matrices,
  \( \partial_a H \approx [H(R-2h)-8H(R-h)+8H(R+h)-H(R+2h)]/(12h) \).

Both are projected with the **same central-geometry eigenvectors**. The routine
never differentiates neighboring eigenvector phases or divides by energy gaps.

Inputs must be float64 Cartesian positions and float64 or complex128 Hamiltonians.
The caller supplies displacement h in coordinate units and numeric tolerances.
Nonfinite, non-Hermitian and shape-inconsistent matrices are rejected.

An optional **reference_derivative** tensor of shape (atoms, 3, states, states)
supplies an independent target in the **same fixed diabatic basis**. Both
the autograd and finite-difference derivatives must match that reference.
Without it, passing means *internal numerical consistency only*. An isospectral
but incorrect Hamiltonian could pass the internal comparison. Under a constant
diabatic unitary basis change, transform the reference by the same unitary.

A geometry-dependent electronic basis needs additional connection/covariant
derivative terms. This fixed-basis formula cannot be blindly applied to
independently optimized AO or MO coefficients.

## Analytic positive and negative controls

At x=0, H(x)=0.07*x*sigma_z+1e-9*sigma_x has an energy gap of 2e-9, below the
default gap floor of 1e-6. The off-diagonal numerator magnitude is 0.07,
**not zero**, but its state-resolved coupling is marked unresolved.

A detached-gradient defect gives plausible energies but incorrect autograd
derivatives: five-point differences reject it. An isospectral Hamiltonian
with wrong operator derivatives may pass internal numerical checks but fails
an explicitly declared same-basis reference oracle. Complex phases,
constant electronic-basis gauges, multi-atom/multi-axis derivatives,
fourth-order finite-difference convergence, and exact degeneracies are
also covered.

## Reproduce

~~~bash
pytest tests/test_certification.py -q
python examples/12_coupling_numerator_certification.py
~~~

The example emits machine-readable JSON with the gap floor, unresolved pairs,
reference residual, and pass/fail. Set GITHUB_SHA to link it to a code revision.

## Claim limits and remaining gates

This does not validate physical molecular overlaps, local conical-intersection
geometry, truncated-state completeness, or nonadiabatic trajectories.
