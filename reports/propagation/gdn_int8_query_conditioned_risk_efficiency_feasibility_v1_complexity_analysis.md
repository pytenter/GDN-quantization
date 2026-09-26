# Complexity Analysis

| Candidate | Fidelity | Future-KL | Extra state pass | Extra state bytes | Extra FLOPs | Dynamic metadata | Fusable | Practical |
|---|---:|---:|---:|---:|---:|---:|---|---|
| M0 | baseline | evaluation only | 0 | 0 | 0 | none | YES | YES |
| query_only exact | high dynamic baseline | evaluation only | 0 | 0 | 1048576 | q_t O(d_k) | YES | PARTIAL |
| dyn-q/static-w exact | best query-conditioned exact | evaluation only | 0 | 0 | 1052672 | q_t O(d_k) | YES | PARTIAL |
| q2/static-w | separable no cross-row terms | evaluation only | 0 | 0 | 1572864 | q2 O(d_k) | YES | YES |
| group-q/static-w | coarser q2 | evaluation only | 0 | 0 | 1572864 | group q2 O(d_k/group) | YES | YES |
| head-q/static-w | head energy only | evaluation only | 0 | 0 | 1572864 | head scalar O(head) | YES | YES |
| static-column | no dynamic q | evaluation only | 0 | 0 | 1048576 | none | YES | YES |
