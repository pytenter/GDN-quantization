# LING_ERROR_COHERENCE_DIAGNOSIS_V1

Evidence category: PREREGISTERED_EXPLORATORY_MECHANISM_DIAGNOSIS. This document and its JSON companion must be committed before any new measurement, including development closure measurements. Historical parent global FAIL remains FAIL_PRESERVED. The separate prospective 32K confirmation is not repeated or pooled.

## Scope and hardware amendment V1

User authorization: “那就只使用GPU6来完成指令吧”. All four original Discovery documents, in original order, are assigned prospectively to physical RTX3090 GPU6 on nlpg-SYS-4029GP-TRT, UUID GPU-de53f243-b271-521c-d875-d96611ed8574. This supersedes only the previous GPU3/GPU5 assignment for this new experiment. No external process is terminated. FP is captured on GPU6 and all Q comparisons/repeats use GPU6. Cross-hardware FP bitwise FAIL is preserved; no old 4090 FP numeric trace is substituted.

No training, rotation learning, document selection, Confirmation, L9, AIME, free generation, endpoint preparation or automatic follow-on experiment is authorized. FINAL_ACTION=STOP_AFTER_DISCOVERY.

## Data and execution

The exact parent Discovery manifest supplies pg19-test-37702, pg19-test-5396, pg19-test-28988, pg19-test-26493, ranks0..3. The parent 16-rotation manifest supplies exact artifacts and order. Their original hashes, model/shard identities, source files, statistical implementation, sign schedule and runtime lock are recorded by the preparation audit. Full hashes are checked before execution.

Route B is necessary: parent artifacts contain frozen FP drive/chunk endpoints and scalar metrics, not Q/F(S_Q) per-token tensors. Moreover the existing Discovery FP traces are 4090 references. Fresh same-GPU6 capture uses exact inherited Capture semantics and model without any training or generation. Replay uses unchanged parent transition, QDQ and signed K=8 J_PROP implementation.

Each of 64 document/rotation units has one 32768-token rollout, with horizons2048/8192/32768. Primary horizon32768; shorter horizons secondary. State cache is inherited FP32 [18 layers,16 heads,128 key,128 value], not BF16 cache. The model and q/k/v drive remain BF16. QDQ is rowwise symmetric INT8-R128, round/clamp[-127,127], amax clamp1e-12, immediate FP32 recurrent writeback. Native physical state recovered by stored rotated state @R.T.

## Exact finite-transition decomposition

At each token, retain the actual stored rotated Q cache; never recover-and-rerotate it as a replacement for the persistent path. Let Sq_phys=Sq_rot@R.T. F(Sq_phys,u) is the inherited actual native fused non-QDQ transition evaluated on that recovered physical input with native frozen u. F(Sfp,u) is the unchanged native FP path. The actual Q next physical endpoint is the existing rotated-v/BF16/writeback path recovered @R.T.

E=Qnext_phys-FPnext; H=F(Sq_phys,u)-FPnext; I=Qnext_phys-F(Sq_phys,u). These are exact endpoint differences, not Jacobians or a zero-write approximation. I is the task-defined actual-Q-versus-native-F injection. It includes both actual QDQ and numerical rotated-basis/write rounding. It is explicitly NOT relabeled as the parent's pure eta. For audit, separately record pure recovered QDQ injection Qnext_phys-preQ_rot@R.T and basis/rounding discrepancy preQ_rot@R.T-F(Sq_phys,u). Their sum equals I. No recurrence or endpoint is changed to force equivariance.

Differences and reductions use float64 after unchanged FP32 transition and native recovery. Raw energies are ||E||²,||H||²,||I||²,2<H,I>. Primary per-layer normalization is 1/(||FPnext||²+1e-12), exactly the parent J_ACC mathematical normalization. Primary aggregation is equal token/layer weighting on inclusive one-based[1024,h]; all heads/key/value cells are summed within layer. J_ACC uses float64 arithmetic on the same physical endpoints and is checked against inherited FP32 J_ACC with the original 5e-5 atol/rtol parity policy. J_INJ and J_PROP come directly from the unchanged inherited replay_step (including its original FP32 arithmetic and K=8 signs), not redefinitions.

SUM_* are sums of these normalized atomic energies over the primary support; J_ACC and J_NOALIGN divide by token count×18 layers. For physical units, RAW_SUM_* are also saved. Coherence ratios are explicitly normalization-weighted; no ratio-of-unweighted-energy is substituted. Per-token/layer coherence and cosine use raw norms; zero denominator gives0 with no new epsilon. NET_COHERENCE_RATIO=SUM_ALIGNMENT/(SUM_HISTORY+SUM_INJECTION), zero=>0; POS/NEG/ABS and coherence bias follow the task equations. Alignment sign fractions are token×layer fractions with exact-zero classification. Secondary cosine and per-token coherence are descriptive means only.

## Support, segments and audit storage

Segment full supports are one-based[1,2048],[2049,8192],[8193,32768], equivalent to zero-based0–2K,2K–8K,8K–32K. Save both FULL_SEGMENT_DIAGNOSTIC and PRIMARY_SUPPORT_INTERSECTION: the first primary segment is[1024,2048], others unchanged. Full-segment values never replace historical scoring support. History propagates from token1 regardless of inclusion.

Save per-layer/horizon and segment sums; each512-token window has layer-wise weighted and raw energy aggregates, signed counts, coherence/cosine sums, pure-QDQ and basis discrepancy energies, vector/energy closure diagnostics. Per window retain five largest positive and five largest negative normalized alignment token-layer events (ties: smaller token then layer), with energies and diagnostic components. No full all-token Q/state dump. Window tables reproduce aggregate metrics to float64 reduction roundoff and retain closure extrema. Frozen FP drive chunks and their endpoints remain available for full replay audit.

## Gates and deterministic repeats

Before formal Discovery, two fresh independent processes each capture128 historical development tokens and perform the same fixed H-rotation replay on GPU6. Compare all captured drive/end-state tensor SHA hashes and numeric replay/closure records exactly, excluding timestamps/paths. In each chunk FP replay equals captured endpoint bitwise. Actual package versions, Python, source hashes, GPU UUID/name/driver/CUDA, environment and deterministic flags are recorded. Repeat FAIL stops; historical cross-hardware FP FAIL remains recorded separately.

Vector closure: max_abs(E-H-I)<=1e-10 AND per-layer relative_L2<=1e-10 (zero E denominator:0 only if residual0, else infinity/fail). Energy closure: abs(E²-H²-I²-A)<=1e-10×max(1,E²+H²+I²+|A|). These fixed outcome-independent tolerances bound float64 algebra on unchanged endpoints, not model parity. No clipping of coherence/cosine; bound check abs<=1+1e-10. Every unit also checks parent FP32 J_ACC parity using5e-5 atol/rtol and bitwise FP chunk endpoints. All nonfinite, identity, semantic, closure, parity, corruption failures STOP without retry or relaxed threshold.

## Rankings, ties and descriptive labels

Use the parent's Spearman and order functions unchanged. Pair tie tolerance=0, exact equality, concordance=concordant/120 (ties nonconcordant). Decisive rate denominator counts pairs non-tied in both J_NOALIGN and J_ACC; decisive means opposite signs. Do not invent amplification/cancellation-driven pair labels: report NOT_CLASSIFIED because differential signed contributions can be mixed. Report amplification/cancellation magnitudes directly. Do not create rotation-level pair rates.

Report per-doc values and document-level medians; NET ratio doc summaries are medians across16 rotations, then median across4docs. Rotation scatter remains descriptive; no64-point pooled primary correlation or p-value.

Predeclared descriptive rule at32K: all four docs with >=1 decisive pair and >=3 docs with median abs(NET_COHERENCE_RATIO)>=0.01 => DESCRIPTIVE_SIGNAL_PRESENT. No decisive pairs in all four docs and all four medians abs(NET)<0.01 => DESCRIPTIVE_SIGNAL_NOT_PRESENT. If at least one decisive pair exists but all four median abs(NET)<0.01 => DESCRIPTIVE_SIGNAL_WEAK. Otherwise DESCRIPTIVE_SIGNAL_MIXED. The1% magnitude convention is descriptive, not a confirmatory significance threshold. Missing/undefined metrics cause fail-closed, not a favorable label.

CONFIRMATORY_STATUS=NOT_EVALUATED. ENERGY_DECOMPOSITION_IDENTITY_IS_TAUTOLOGICAL=YES. SCIENTIFIC_QUESTION=IS_ALIGNMENT_STRUCTURED_AND_RANK_DECISIVE. Signal-present recommends a separately preregistered prospective confirmation only; other labels recommend considering readout-sensitivity diagnosis only. Neither is executed. No reasoning/AIME/training efficacy claim is permitted.

## Resume, integrity and storage

Each unit identity includes document/token/rotation/model/checkpoint/source/premeasurement/runtime/schema SHA, GPU UUID, trace freeze SHA. Completed unit SHA sidecars must match and identity must match exactly. A started but incomplete unit, partial trace, corrupted artifact or STOP evidence blocks execution; no automatic repair/retry. Atomic exclusive writes; all failures retained. Inputs are rehashed at preflight and per-document boundaries. Source changes after measurement require a separately authorized append-only amendment.

Historical manifests/reports/raw/code and all inherited protected source/model/token/rotation files remain byte-identical. No writes in historical namespaces. Historical experiment membership and full tracked new-worktree cleanliness are audited separately; generated new outputs are allowed only inside this new namespace.

Before new capture require >=53,301,980,416 bytes for four trace sets +2GiB outputs/development +20GiB fixed safety reserve. Per-document capture rechecks remaining required storage. Lack of capacity stops; no history deletion. Current GPU must have <2000MiB occupancy before capture; no concurrent inherited worker may run on GPU6. At completion freeze64raw hashes BEFORE statistics, generate required CSV/JSON/MD reports, verify all inventory entries, and STOP_AFTER_DISCOVERY.
