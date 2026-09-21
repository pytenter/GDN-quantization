# KDA Rotation Result Reinterpretation

Task: `KDA_VALUE_HADAMARD_POSTFIX_SCIENTIFIC_REBASELINE_V1`

## A. Still Valid

- Native R128 versus C128 orientation findings that do not depend on the historical Rotated branch.
- The Value-side rotation theorem and exact coordinate orientation.
- Corrected FP equivalence of Value-side Hadamard.
- Static INT8_R128 quantization improvement measured on the corrected path.

## B. Valid Only As Bug-Propagation Evidence

- Rotated-history damage, prequant donor rescue, history-dominant classification, and prefill localization.
- Packed/sequential, write-back, persistent exposure, and related panels that localized propagation of the bad endpoint state.
- The historical Buggy Hadamard FutureKL trajectory is retained for provenance only.

## C. Invalid As Intrinsic KDA Rotation Mechanism

- Historical claims of intrinsic Value-side rotation instability.
- Historical claims of intrinsic KDA long-horizon rotation failure.
- Historical claims of an intrinsic state-quantization-boundary failure based on Buggy Hadamard FutureKL near 0.12 or above.

Historical artifacts are preserved. Their interpretation is changed because the severe branch contained a redundant prefill-end state basis rotation.
