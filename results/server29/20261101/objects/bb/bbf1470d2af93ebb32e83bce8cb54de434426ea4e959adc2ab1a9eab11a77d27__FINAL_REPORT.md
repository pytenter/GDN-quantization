# QWEN_GDN_SHARED_OBJECTIVE_PARETO_AND_DEV_SCREEN_V1 — Final Report

## Executive conclusion

The primary result is **INSTABILITY_NO_STABLE_SELECTION_SIGNAL**. Existing learned rotations can beat canonical Hadamard on the original shared objectives, but candidate ordering changes across metrics and between the frozen SCREEN and CHECK splits. The preregistered C5 u63 stability hypothesis was falsified. The evidence therefore does not justify selecting a historical checkpoint for later end-to-end evaluation.

## 1. Did any learned checkpoint improve the original shared objectives relative to Hadamard?

Yes. The exact-protocol reevaluation of canonical Hadamard produced State relative-MSE **0.008496120241034077** and Functional relative-MSE **0.04767651924339589** over the same 16 documents and 128 captures. Two historical checkpoints strictly dominate it on both lower-is-better objectives:

- C5 u82 (see derived Pareto table).
- C5 u113: State **0.0081005504871427** and Functional **0.0469885527490987**, corresponding to **-4.655887%** State and **-1.442988%** Functional versus Hadamard.

Hadamard strictly dominates 14 learned checkpoints; 24 more are tradeoffs. These results establish shared-surrogate improvement only, not reasoning or free-generation improvement.

## 2. Is canonical C5/C6 checkpoint selection Pareto-optimal?

- **C5 u63: YES.** State **0.0079161851317621**, Functional **0.0495092943710915**. It remains Pareto-optimal after adding Hadamard.
- **C6 u125: NO.** State **0.008251136421677**, Functional **0.0490164864968392**. It is dominated in the combined historical table and is not Pareto-optimal after adding Hadamard.

The global historical frontier is occupied by C5 u63, u69, u82, and u113. No C6 checkpoint dominates canonical C5 u63. C5 u75, u82, u88, and u113 dominate canonical C6 u125.

## 3. What is special about C5 u113?

C5 u113 is the clearest historical shared-objective alternative:

- versus canonical C5 u63: State is **2.329% worse**, while Functional is **5.091% better**;
- versus canonical C6 u125: it is better on both State and Functional;
- versus Hadamard: State is **4.655887% lower** and Functional is **1.442988% lower**.

It is globally Pareto-optimal. This makes u113 scientifically interesting, but not sufficient for checkpoint selection: its independent metrics did not transfer with a single stable direction.

## 4. Are there deployable historical alternatives?

Yes. All five Phase-C conditions had verified provenance:

- Hadamard: canonical theta=0/H128 identity baseline.
- C5 u63: canonical frozen rotation SHA256 `0266c5065267cf452d72cfe29a2a4d9a24fdfb6a89d7bec04b4d796c86f857c9`; theta SHA256 `54416ccaf8b02cfa8d0ce5235f0b12c7db888c2ab8f07d44565eecc4c45c40d4`.
- C6 u125: canonical frozen rotation SHA256 `67e3731213ad3fb0c088c781ddd912aaaad5af75f9532b289442690a7b09f39c`; theta SHA256 `8312d59e593181e8b36de01058e492171434e3d0d9b0fb92aa6a979bdd151520`.
- C5 u113: noncanonical research export from theta SHA256 `ecdc13b64c65757b9b50c85e4a85352185564f3dd62dcb3e91758a6cf5638a11`; frozen rotation SHA256 `b625e99dd7f57a87040e517d263555e5366de6bba1678d221ec3cba287fafe023`.
- C6 u119: noncanonical research export from theta SHA256 `a580b847729130faee208be0beb95a2179a18d5b7efd3cda57dadb9beec29c36`; frozen rotation SHA256 `7e325dde5fefd34b9ea71b7a0ffbf9196a275abb451208003196b173be58b2f41`.

The two noncanonical matrices were produced with the audited deterministic canonical Cayley export and passed 24-layer orthogonality/hash checks. They remain explicitly noncanonical. The independent MATH-500 subset is self-contained and hash-frozen; its archived source Arrow path was not present on this host, which is a provenance limitation but did not change the frozen JSON records.

## 5. Did validation advantages transfer on independent dev?

The frozen panel used eight deterministically hash-ordered non-AIME MATH-500 tasks: four SCREEN and four CHECK. Each condition independently preserved KV/recurrent state, and every non-FP step performed canonical C128 QDQ/writeback. Values below are candidate-minus-Hadamard means; lower is better.

| Split | Candidate | State | Functional | Final hidden | Full-vocab KL | Fixed-token NLL | Documents better (State / Func / Hidden / KL / NLL) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SCREEN | C5 u63 | -0.00020636 | -0.00034481 | -0.00025119 | -0.00017704 | +0.04051183 | 4/4 / 3/4 / 4/4 / 4/4 / 1/4 |
| SCREEN | C5 u113 | -0.00003417 | -0.00011416 | -0.00007018 | +0.00041054 | +0.01881467 | 2/4 / 2/4 / 3/4 / 2/4 / 1/4 |
| SCREEN | C6 u119 | +0.00015410 | -0.00041544 | -0.00029321 | -0.00036269 | +0.02402994 | 3/4 / 3/4 / 3/4 / 3/4 / 1/4 |
| SCREEN | C6 u125 | -0.00017382 | +0.00011033 | +0.00011670 | -0.00024305 | +0.02388411 | 4/4 / 3/4 / 1/4 / 2/4 / 0/4 |
| CHECK | C5 u63 | -0.00022497 | +0.00033041 | +0.00045517 | -0.00036077 | +0.01861688 | 4/4 / 2/4 / 1/4 / 3/4 / 2/4 |
| CHECK | C5 u113 | -0.00031273 | -0.00004971 | -0.00032094 | -0.00121543 | +0.00847521 | 3/4 / 2/4 / 3/4 / 3/4 / 1/4 |
| CHECK | C6 u119 | +0.00046033 | -0.00009771 | +0.00005748 | +0.00090805 | -0.01359742 | 2/4 / 3/4 / 3/4 / 3/4 / 3/4 |
| CHECK | C6 u125 | +0.00028609 | +0.00169664 | +0.00081806 | -0.00003547 | -0.00003678 | 3/4 / 2/4 / 3/4 / 3/4 / 3/4 |

The SCREEN hypothesis ranked C5 u63 safest and required it to improve State, final-hidden error, and KL on at least 3/4 CHECK documents. It improved CHECK State on 4/4 and KL on 3/4, but final-hidden error on only **1/4**, with mean delta **+0.00045517** and maximum positive regression **0.00287644**. Therefore the hypothesis is falsified.

C5 u113 looks stronger on CHECK than SCREEN for State/hidden/KL, but its fixed-token NLL is worse on 3/4 and its FP-reference margin degrades on 3/4. C6 u119 and u125 exhibit large document-specific regressions; for example C6 u119's maximum CHECK KL regression is **0.00673906**, and C6 u125's maximum CHECK Functional regression is **0.00693045**.

## 6. Where does the advantage disappear?

There is no single universal boundary. C5 u113's shared State/Functional advantage already becomes heterogeneous at the independent local metrics. For C5 u63, State transfers while Functional and final-hidden behavior reverse on CHECK. Across candidates, full-vocabulary KL, fixed-token NLL, and FP-reference margin can move in opposing directions. The detectable failure is therefore **cross-document and cross-metric stability from local surrogate through persistent downstream/full-model behavior**, rather than one uniquely failing layer or scalar metric.

These are teacher-forced proxy diagnostics over eight documents and at most 2048 recurrent updates; they do not cover 81920-token reasoning and are not free-generation accuracy.

## 7. Primary next-step category

**INSTABILITY_NO_STABLE_SELECTION_SIGNAL**

Checkpoint selection is not supported because the preregistered direction reverses on CHECK and no candidate is consistently superior across the hierarchy. A pure surrogate-to-downstream-mismatch label would overstate the result: some downstream metrics improve, but the direction is unstable rather than uniformly absent.

## 8. What single factor should the next experiment modify?

Modify only the **training target**: replace the local GDN State/out-projection surrogate with a persistent downstream final-hidden representation fidelity objective. Keep data/order, optimizer, learning rate, exposures, H64 numerical recurrence, quantizer/writeback, rotation parameterization/initialization, seed, and precision fixed.

Expected effect: shared-objective gains should survive into full-model representations with fewer severe document-specific regressions. Falsification: on the same frozen panel, the pilot must improve final-hidden error versus Hadamard on at least 3/4 documents in both SCREEN and CHECK and reduce the maximum positive final-hidden regression; otherwise reject the objective change. Estimated compute is at most 4 GPU-hours for one small preregistered pilot. This is more justified than checkpoint sweeping or optimizer tuning because the local objectives were demonstrably improved, while the first preregistered failure occurred at persistent downstream transfer.

## Execution incidents and boundaries

- The first Phase-C smoke stopped on an audit-only assertion that incorrectly treated the cumulative QDQ call counter as per-layer. It produced no retained scientific metrics; evidence is preserved. The assertion was corrected without changing scientific configuration, and the identical smoke passed.
- No AIME examples, correctness, or free generation were used for selection.
- No training, C7, Full60, layer scan, intervention, or post-hoc weighted score was run.

## Final status

```text
PARETO_ANALYSIS = COMPLETE
HADAMARD_SHARED_BASELINE = COMPLETE
ORIGINAL_VALIDATION_PROTOCOL_MATCH = YES
HADAMARD_DOMINATED_BY_ANY_LEARNED = YES
CANONICAL_C5_PARETO_OPTIMAL = YES
CANONICAL_C6_PARETO_OPTIMAL = NO
NONCANONICAL_CANDIDATES_DEPLOYABLE = 2
PHASE_C_GATE = PROCEED
INDEPENDENT_DEV_SCREEN = COMPLETE
PRIMARY_NEXT_STEP_CATEGORY = INSTABILITY_NO_STABLE_SELECTION_SIGNAL
GPU_HOURS_USED = 0.9907508527570301
FULL_DENSITY_FULL60 = CANCELLED_BY_USER
NEW_AIME_FREE_GENERATION = NOT_STARTED
NEW_TRAINING = NOT_STARTED
C7 = NOT_STARTED
```
