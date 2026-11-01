# Cross-runtime acceptance-criteria audit

Audit: `LING_OBSERVABILITY_ATTEMPT05_CROSS_RUNTIME_ELIGIBILITY_AUDIT_V1`  
Frozen decision time: `2026-10-10T12:31:20+08:00`

## Decision

`CROSS_RUNTIME_CRITERION_AUDIT = EXISTING_CRITERIA_NOT_APPLICABLE`

`CROSS_RUNTIME_MERGE_ELIGIBILITY = FAIL`

`MERGE_FAILURE_REASON = APPLICABILITY_NOT_ESTABLISHED`

`ATTEMPT05_STATUS = QUALIFICATION_FAIL_AND_STOP`

This is a qualification failure for combining two runtime populations. It is **not** evidence that driver 580.178.04 numerically diverges, and it is **not** a result about the observability hypothesis.

## Scope and method

The audit was read-only over the original protocol, measurement implementation, premeasurement identity, whole-runtime replay qualification, GPU3 expansion qualification, same-GPU FP references, canonical/finalizer code, preserved PASS/FAIL receipts, and the Attempt04 reboot receipt. No historical file was modified or reserialized. No historical canonical was rerun. No formal unit was started.

A search of the frozen historical worktrees found no record of driver `580.178.04`, and no gate that defines admission of a new-driver unit into the old-driver 32K panel. The only frozen driver-specific GPU3 qualification asserts exactly `580.173.02`.

## What the frozen evidence proves

1. **Identity:** scientific source/assets, documents, rotations and runtime dependencies are byte-identical to their frozen identities.
2. **Old-runtime reproducibility:** GPU6 and GPU3 each passed same-driver fresh-process checks on the short qualification fixtures.
3. **Implementation parity:** K3, JPROP, basis semantics, initial pulse, FP pairing, persistent-QDQ semantics and Future-KL harness wiring passed their original local/short-fixture checks.
4. **Full-horizon FP integrity on GPU3:** same-GPU FP capture replays bitwise through 32768 tokens.
5. **Old canonical integrity:** the 80 existing units have unique scientific keys and verified canonical/sidecar bytes.

## What the frozen evidence does not prove

No premeasurement criterion simultaneously covers:

- the same GPU UUID under drivers `580.173.02` and `580.178.04`;
- the complete 32768-token recurrent trajectory;
- repeated quantize/dequantize/writeback feedback;
- accumulated state error and K3 readout;
- downstream Future-KL;
- an admission rule for `80 old + 48 new` as one scientific panel.

The historical `1e-4` and `1e-6` thresholds are basis orthogonality and functional-basis tolerances. They are not state, endpoint, KL, rank, or cross-driver tolerances and cannot be repurposed.

## Post-reboot short probe disclosure

One GPU6 nonformal qualification process had already been started before the present criterion-first directive. It completed naturally without signal/kill. It recorded driver `580.178.04` on the same GPU6 UUID and exact equality for the frozen short fixture. It produced zero formal units. This observation is preserved in the evidence matrix but is not used to turn the audit into PASS: the fixture is only FP128/anchor64-end128 plus short H/L6 coverage and no frozen rule establishes that this is sufficient for a 32K mixed-runtime panel. No additional probe was launched.

## Historical state preserved

- Attempt04 remains permanent `FAIL_AND_STOP` due to external reboot.
- Existing canonical inventory remains 80/128: Primary 41/64, Secondary 39/64, complete pairs 39/64.
- Missing units remain 48: GPU6 19 and GPU3 29.
- The two interrupted Secondary units have no canonical and are not resumed from partial state.
- All historical PASS and FAIL receipts remain unchanged.

## Conclusion

The evidence set contains strong, legitimate criteria, but their documented scope stops short of cross-driver 32K merge validity. Extending short-fixture exact parity, same-runtime determinism, or FP-only full-horizon replay to recurrent-QDQ/Future-KL panel equivalence would be a new scientific rule. Therefore the only fail-closed decision is Case B and Attempt05 stops before formal measurement.
