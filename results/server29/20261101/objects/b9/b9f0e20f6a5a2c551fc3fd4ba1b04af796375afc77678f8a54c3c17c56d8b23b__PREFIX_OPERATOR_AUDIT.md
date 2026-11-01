# Full-shape prefix replay

The exact full-model prefix input was replayed in **10 fresh processes**, with **5 same-process repeats** each. Both operators were internally exact and reproducible across processes, but their output hashes differ consistently. Replay output hashes match the corresponding full-model captures.

Canonical versus candidate max abs: 1.525878906e-05, relative L2: 1.125205311e-07. Against CPU FP64 mathematical reference, canonical relative L2 is 3.793015294e-08 and candidate is 1.054624108e-07. The candidate is deterministic but not canonical-compatible for this model; CPU FP64 was diagnostic only.
