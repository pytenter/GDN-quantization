# LING_L6_L7_PAIRED_TRAJECTORY_CLOSURE_V1

This is a strict read-only offline analysis. No training or free-generation evaluation was run, and no frozen source artifact was changed.

## Outcome integrity

Exact reproduced scores: FP **14/20**, Hadamard/L2 **9/20**, L6 **9/20**, L7 **10/20**.

- L6-H: rescue=['aime26_23']; regress=['aime26_26']; net=0; exact p=1.0. Holm p=1.0.
- L7-H: rescue=['aime26_11', 'aime26_12']; regress=['aime26_26']; net=1; exact p=1.0. Holm p=1.0.
- L7-L6: rescue=['aime26_11', 'aime26_12']; regress=['aime26_23']; net=1; exact p=1.0.

## All-20 paired trajectory summary

Sign convention throughout is condition A minus condition B. Continuous estimates are exploratory, with paired mean/median, sign counts, and deterministic 10,000-replicate bootstrap CIs in `statistics.json`.

### L6-H

- generated_tokens: n=20, mean Δ=-2.885e+04, median Δ=0, signs +/0/−=8/3/9, bootstrap mean-Δ CI=[-66633.87375, 8567.489999999994].
- repeated_4gram_fraction: n=20, mean Δ=0.0173, median Δ=0.04081, signs +/0/−=13/0/7, bootstrap mean-Δ CI=[-0.02685800052396982, 0.05965912970586486].
- first_committed_claim_position: n=12, mean Δ=-9443, median Δ=4883, signs +/0/−=6/0/6, bootstrap mean-Δ CI=[-28458.63958333333, 7990.77708333332].
- num_answer_changes: n=20, mean Δ=0.6, median Δ=0, signs +/0/−=7/5/8, bootstrap mean-Δ CI=[-7.1, 7.15].
- retracted_event_count: n=20, mean Δ=7.35, median Δ=3, signs +/0/−=13/0/7, bootstrap mean-Δ CI=[-0.1, 16.25124999999998].

### L7-H

- generated_tokens: n=20, mean Δ=753.9, median Δ=1260, signs +/0/−=10/5/5, bootstrap mean-Δ CI=[-40298.369999999995, 40123.628749999996].
- repeated_4gram_fraction: n=20, mean Δ=0.01145, median Δ=0.01279, signs +/0/−=12/0/8, bootstrap mean-Δ CI=[-0.021125761924152314, 0.04409039007242319].
- first_committed_claim_position: n=11, mean Δ=-1.302e+04, median Δ=1298, signs +/0/−=6/0/5, bootstrap mean-Δ CI=[-38027.07045454545, 6609.840909090902].
- num_answer_changes: n=20, mean Δ=-2.25, median Δ=0.5, signs +/0/−=10/4/6, bootstrap mean-Δ CI=[-9.45, 3.001249999999982].
- retracted_event_count: n=20, mean Δ=2.05, median Δ=1, signs +/0/−=11/0/9, bootstrap mean-Δ CI=[-4.25, 9.20124999999998].

### L7-L6

- generated_tokens: n=20, mean Δ=2.961e+04, median Δ=3.99e+04, signs +/0/−=11/1/8, bootstrap mean-Δ CI=[-22053.161249999997, 77596.07999999996].
- repeated_4gram_fraction: n=20, mean Δ=-0.00585, median Δ=-0.01646, signs +/0/−=9/0/11, bootstrap mean-Δ CI=[-0.03408142952032033, 0.022030282606832312].
- first_committed_claim_position: n=11, mean Δ=167.7, median Δ=-4598, signs +/0/−=4/0/7, bootstrap mean-Δ CI=[-14759.384090909092, 16825.136363636353].
- num_answer_changes: n=20, mean Δ=-2.85, median Δ=0, signs +/0/−=9/2/9, bootstrap mean-Δ CI=[-9.95, 3.8012499999999814].
- retracted_event_count: n=20, mean Δ=-5.3, median Δ=-6.5, signs +/0/−=7/1/12, bootstrap mean-Δ CI=[-13.3, 3.05].

## Outcome/trajectory association

- L6: among 7 prespecified directional checks, 7 place rescue deltas in a more favorable direction than regression deltas. With only 1 rescues and 1 regressions, this is descriptive only.
  Switch-direction checks: termination 1/2, length 1/2, stable commitment 1/2 coherent.
- L7: among 7 prespecified directional checks, 4 place rescue deltas in a more favorable direction than regression deltas. With only 2 rescues and 1 regressions, this is descriptive only.
  Switch-direction checks: termination 3/3, length 3/3, stable commitment 2/3 coherent.

## Switch-case findings

- aime26_11: H=wrong, 261456 tokens, LENGTH_CEILING; L6=abstain, 261456 tokens, LENGTH_CEILING; L7=correct, 74923 tokens, EXPLICIT_EOS_OR_STOP. See `switch_case_forensics.md` for compact claim timelines.
- aime26_12: H=abstain, 261443 tokens, LENGTH_CEILING; L6=abstain, 261443 tokens, LENGTH_CEILING; L7=correct, 47257 tokens, EXPLICIT_EOS_OR_STOP. See `switch_case_forensics.md` for compact claim timelines.
- aime26_23: H=abstain, 261540 tokens, LENGTH_CEILING; L6=correct, 96753 tokens, EXPLICIT_EOS_OR_STOP; L7=abstain, 261540 tokens, LENGTH_CEILING. See `switch_case_forensics.md` for compact claim timelines.
- aime26_26: H=correct, 214389 tokens, EXPLICIT_EOS_OR_STOP; L6=wrong, 194930 tokens, EXPLICIT_EOS_OR_STOP; L7=abstain, 261515 tokens, LENGTH_CEILING. See `switch_case_forensics.md` for compact claim timelines.

q26 L7 independently verifies file SHA256 `8a08632e751aaddbf18fbefcb78001bbffa689e0e2c119af4161e8be8dbc89bf`, 261515 generated tokens, length ceiling=True, incorrect=True, abstain=True.

## Geometry/checkpoint sanity

L6/L7 rotations were independently loaded. Per-layer orthogonality, NaN/Inf, determinant sign/log-absolute-determinant, parameter norm, exact checkpoint/file hashes, and same-basis L6↔L7 Frobenius distances are in `rotation_sanity.json`. Distances are provenance/sanity only, not reasoning metrics.

## Hardware and provenance caveats

- Effective assignment SHA256: `e2781d2ecaf81f09d0c36b78438bb4d88c8236cebb5bd7a55c72c53a85bb3cf3`.
- FP is taken from one exact frozen canonical baseline artifact set. The known FP cross-hardware bitwise parity failure is preserved; no cross-hardware FP outputs are pooled as interchangeable observations.
- H q11/q12 retain their migrated 4090 generation identity; the remaining frozen H samples and FP baseline samples retain their exact artifact paths and generation metadata.
- L6/L7 condition-level INT8 parity passed in the source audit, permitting the frozen mixed-hardware canonical assignment.
- Commitment token positions are estimates from normalized character fraction to stored token count; official scoring uses the original validated V4 parser and is not based on those estimates.
- Established 81,920-token segmented oscillation/loop/semantic-degeneration indicators are `NOT_AVAILABLE`; they were not silently redefined for 262,144 tokens.

## Diagnostic labels

- `PROVENANCE_GATE = PASS`
- `SCORER_ARTIFACT_EXPLANATION = NOT_SUPPORTED`
- `L7_STRUCTURED_TRAJECTORY_SIGNAL = SUPPORTIVE`
- `L6_STRUCTURED_TRAJECTORY_SIGNAL = INCONCLUSIVE`
- `CURRENT_RESULT_INTERPRETATION = STRUCTURED_BUT_UNDERPOWERED`

The score ordering alone is not used as evidence. Exact p=1.0 is not interpreted as method identity. The small number of discordant pairs sharply limits inference.

`NEXT_ACTION = CANONICAL_60_EVALUATION`

No next action was executed.
