# Limitations

## Molecular representation

The default model uses all pair distances and a sum aggregation. It has no angular,
many-body, periodic, charge-state, or spin-state features. Pair distances cannot
distinguish enantiomers. The implementation processes one molecule at a time and
scales quadratically with atom count.

`SharedMolecularOperators` adds a centered polar-vector dipole head for neutral
systems. Its vectors lie in the span of centered nuclear positions, excluding
out-of-plane dipoles for planar geometries. Equivalent atomic environments share
weights, forcing zero dipoles for homonuclear diatomics, including physically
allowed transitions. It is a minimal reference model; complex mode does not impose
SOC, time-reversal, or Kramers constraints.

## Diabatic identifiability

Adiabatic energies do not select a unique diabatic Hamiltonian. Gradient targets
constrain eigenvalue variation but do not remove all gauge freedom. Off-diagonal
state-sensitive targets need phase and subspace alignment across geometries.

The connected-path tracker can apply supplied overlap information and diagnose weak
or tied assignments. It does not generate electronic-structure overlaps or select a
unique global diabatic gauge.

`MolecularPathDataset.split(family_key=...)` keeps supplied molecular families
together. `assert_disjoint_paths()` checks IDs, family labels, and duplicate
ordered-atom distance matrices across partitions. Family provenance remains the
caller's responsibility. This geometric screen does not identify atom permutations,
chemical graph identity, or every source of statistical correlation.

## Degeneracies

Individual eigenvectors become gauge-sensitive near degeneracy. The derivative
coupling utility suppresses divisions below a user-set gap threshold and can return a
mask. It does not construct a smooth degenerate-subspace gauge.

The tracker aligns a degenerate block when equal-dimensional blocks are present at
both adjacent geometries. A block that splits or merges is reported as ambiguous; the
code does not continue with invented individual root identities.

The second-best assignment margin requires repeated assignment solves and is designed
for the small finite-state manifolds in GeneralDIA, not hundreds of electronic roots.

## Training scale

The reference trainer uses one geometry per optimizer step and does not provide graph
batches, distributed training, mixed precision, early stopping, schedulers, or data
streaming. It serves small experiments and reference implementations.

The one-geometry loss compares ascending adiabatic energy ranks. The separate
`path_observable_loss()` and `train_path_model()` support tracked state character.
Target and prediction block partitions must agree; initial models with incompatible
degeneracies require a suitable initialization, not relaxed ambiguity checks.

The legacy derivative path objective compares each Cartesian component's diagonal-block eigenvalues
and off-block singular values. It loses relative information between components and
blocks and is not generally invariant to spatial rotation. The new dipole and joint
terms compare full Cartesian trace/Gram/cross tensors and do have that invariance.
These remain partial moments: neither objective proves joint gauge equivalence or
identifies a unique latent Hamiltonian. First-order gradients are supported within a fixed accepted
partition. Higher derivatives through the block-frame operation are unsupported;
assignment changes and repeated operator singular values can be nonsmooth.

## Visual reports

The HTML report communicates stored tracking evidence and contains no molecular
viewer or live electronic-structure backend. It visualizes the supplied path and
thresholds; it does not certify that the overlaps are physically correct.

## Electronic structure

The bundled PySCF symbol table supports elements H through Ca. The SA-CASSCF adapter
uses equal state weights and assumes users selected a valid active space. Production
datasets need restart handling, state tracking, and calculation-level failure logs.

The FCI dipole path adapter supplies neutral singlet energies, dipoles, and physical
cross-geometry overlaps in small orbital spaces. It supplies no gradients or
coupling numerators. Its default eight-orbital cap is a guard for small reference
experiments, not a promise of inexpensive scaling for arbitrary electron counts.

## Quantum backends

Pauli expansion requires a state dimension equal to a power of two and costs
$O(4^n)$ Pauli terms for $n$ qubits. The PennyLane and Qiskit adapters target the
ground state with a small hardware-efficient ansatz. They do not implement excited
states, subspace-search VQE, noise models, error mitigation, or fermionic encodings.

## Scientific validation

The synthetic example verifies software integration. It does not establish chemical
accuracy or suitability for nonadiabatic dynamics. Each application needs external
reference data and tests designed for its geometry domain.

The LiH FCI/STO-3G comparison provides limited molecular integration evidence with
one seed and held-out bond lengths within the same molecule. It does not validate
unseen-family generalization, chemical accuracy, conical intersections, Berry
phases, or molecular derivative/dipole joint supervision.
