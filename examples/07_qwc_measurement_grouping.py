from generaldia.quantum import measurement_plan

# Eq. (7) from Verteletskyi, Yen, and Izmaylov, J. Chem. Phys. 152, 124114 (2020),
# translated into GeneralDIA's left-to-right Pauli-label order for four qubits.
paper_terms = {
    "ZIII": 1.0,
    "ZZII": 1.0,
    "ZZZI": 1.0,
    "ZZZZ": 1.0,
    "IIXX": 1.0,
    "YIXX": 1.0,
    "YYXX": 1.0,
}

for method in ("largest_first", "exact"):
    plan = measurement_plan(paper_terms, method=method)
    assert plan.n_measurement_settings == 2
    print(f"{method}: {plan.n_measurement_settings} measurement settings")
    for index, group in enumerate(plan.groups, start=1):
        print(f"  group {index}: labels={group.labels}, basis={group.basis}")
        print(f"    local basis changes={group.basis_changes}")
