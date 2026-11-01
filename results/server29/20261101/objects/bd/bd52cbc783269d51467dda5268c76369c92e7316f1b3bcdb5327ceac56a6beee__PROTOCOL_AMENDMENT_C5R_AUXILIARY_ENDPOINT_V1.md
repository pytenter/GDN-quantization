# PROTOCOL AMENDMENT
## QWEN_GDN_A800_CLEANROOM_C5R_C6R_C7_MATCHED_V1
## C5R_AUXILIARY_C6R_C7_PRIMARY_ENDPOINT_V1

### Purpose

Amend the already-running clean-room experiment prospectively.

Do NOT restart the experiment.

Do NOT invalidate or overwrite the original preregistration.

Do NOT change any training configuration, loss, seed, data, model, optimizer, checkpoint-selection rule, or readiness gate.

The amendment changes ONLY the downstream evaluation path after training.

The scientific priority is now:

> C6R vs C7 is the primary matched-objective experiment.

C5R remains an auxiliary clean-room reproduction of the historical State-training lineage.

C5R no longer automatically proceeds to Canonical60 AIME26.

---

# 1. Timing and provenance

The amendment is being issued while the experiment is still in the update-0 / pre-formal-training stage.

Before applying it:

record the exact current state:

- current commit
- current tmux jobs
- whether any optimizer update has occurred
- C5R/C6R/C7 current phase
- completed hard gates
- artifact hashes

Create:

protocol/PROTOCOL_AMENDMENT_C5R_AUXILIARY_ENDPOINT_V1.md

and

protocol/PROTOCOL_AMENDMENT_C5R_AUXILIARY_ENDPOINT_V1.json

Record:

- amendment timestamp
- source commit
- reason
- exact changed workflow
- exact unchanged protocol elements

Do NOT modify or rewrite the original preregistration file.

If any optimizer update unexpectedly already occurred before this amendment is written:

record that fact explicitly.

The amendment still applies only to future endpoint routing and must not retroactively change any completed training or selection decision.

---

# 2. Unchanged training plan

Training remains:

C5R
→
C6R
→
C7

sequentially on the same trusted A800 environment.

All three training arms still run if their hard gates pass.

Nothing changes in:

- trusted A800 qualification
- complete GPU parameter integrity
- teacher caches
- training data
- document order
- tokenizer
- seed
- sequence length
- captures/document
- recurrent semantics
- C128 QDQ/writeback
- H64
- optimizer
- learning rates
- weight decay
- gradient clipping
- update count
- target exposure count
- validation cadence
- checkpoint cadence
- Hadamard initialization
- rotation parameterization

No loss is changed.

C5R remains:

historical recurrent State objective.

C6R remains:

historical local Functional objective
+
0.1 × State.

C7 remains:

Final-Hidden relative-MSE only.

---

# 3. Primary scientific hierarchy

The experiment now has two different roles.

## Primary causal experiment

C6R vs C7

Reason:

Both are trained:

- on the same A800;
- with the same trusted model;
- with the same data;
- with the same train order;
- with the same seed;
- with the same H64 recurrent semantics;
- with the same optimizer;
- with the same LR = 0.001;
- from the same Hadamard initialization.

The primary intended scientific difference is:

C6R:
local Functional + 0.1 State

versus

C7:
Final-Hidden fidelity.


Therefore the central question is:

> Does moving supervision from local GDN function space to full-model Final-Hidden space improve the reasoning endpoint?


## Auxiliary reproduction

C5R

C5R answers:

> Does the historical State-training lineage broadly reproduce in the trusted A800 clean-room?

C5R is NOT required to become a third full endpoint unless its clean-room training exhibits a material shift that requires end-to-end characterization.

---

# 4. C5R training still completes normally

Do NOT truncate or weaken C5R training simply because its AIME endpoint is now conditional.

C5R still must complete:

- gradient smoke
- H64 readiness
- formal 125 updates / 1000 exposures
- historical checkpoint selection rule
- checkpoint artifact freeze
- State/Functional/Final-Hidden validation logging
- Shared16 comparison
- historical reproduction analysis

Save all scheduled C5R candidates and frozen rotations exactly as originally planned.

The only changed question is:

whether selected C5R must proceed to 3090 Canonical60 generation.

---

# 5. C5R reproduction decision gate

After C5R formal training and checkpoint selection, compare it with historical C5 using ONLY non-AIME training/validation evidence.

Historical C5 reference:

selected update:
63

selected exposure:
504

historical Shared16 State MSE:
0.007916185...

historical Shared16 Functional MSE:
0.049509294...

Also use any legitimately comparable historical learning-curve information that is already archived.

Do not invent missing historical Final-Hidden values.


Evaluate:

1. selected update/exposure behavior;
2. State validation trajectory;
3. Functional diagnostic trajectory;
4. selected checkpoint State MSE;
5. selected checkpoint Functional MSE;
6. qualitative optimization shape;
7. numerical stability;
8. rotation behavior as descriptive evidence only.


Create:

analysis/C5R_REPRODUCTION_GATE.md


Assign exactly one:

C5R_REPRODUCTION = CLOSE_REPRODUCTION

or

C5R_REPRODUCTION = MATERIAL_SHIFT

or

C5R_REPRODUCTION = INCONCLUSIVE


Do not choose the category based on AIME results because C5R AIME has not run yet.

Do not create an arbitrary weighted similarity score.

---

# 6. C5R endpoint routing

## If:

C5R_REPRODUCTION = CLOSE_REPRODUCTION

Then:

- freeze C5R selected checkpoint and artifacts;
- include C5R in Shared16 tables;
- preserve it for future use;
- DO NOT deploy C5R for Canonical60;
- DO NOT generate new C5R AIME outputs.

Set:

C5R_AIME26_ROUTE = SKIP_NOT_NEEDED


Historical C5 remains the archived reasoning endpoint:

33/60.


This does NOT mean C5R accuracy is assumed to equal 33/60.

It means an additional C5R reasoning endpoint is not necessary for the main C6R-vs-C7 question.


## If:

C5R_REPRODUCTION = MATERIAL_SHIFT

Then the clean-room reproduction differs enough from historical C5 that its reasoning behavior becomes scientifically relevant.

Proceed with:

C5R frozen-R deployment
→
canonical 3090 smoke
→
C5R Canonical60


Set:

C5R_AIME26_ROUTE = RUN_DUE_TO_MATERIAL_SHIFT


This endpoint is used to quantify the clean-room shift.

It must NOT be used to tune C5R or C6R/C7.


## If:

C5R_REPRODUCTION = INCONCLUSIVE

Default:

do NOT automatically run C5R AIME merely to resolve uncertainty.

Set:

C5R_AIME26_ROUTE = SKIP_INCONCLUSIVE

unless the report demonstrates that the AIME endpoint is specifically required to distinguish a concrete clean-room training confound.

If such a case occurs, stop and record the justification rather than silently adding the run.

---

# 7. C5R-specific failure must not automatically kill C6R/C7

Because C5R is now auxiliary, distinguish:

SHARED_INFRASTRUCTURE_FAILURE

from

C5R_SPECIFIC_FAILURE.


Examples of shared failures:

- A800 parameter integrity failure
- teacher target corruption
- common recurrent runtime failure
- common quantizer failure
- shared data provenance failure

These must stop all downstream training.


Examples of potentially C5R-specific failures:

- State-objective-specific gradient problem
- C5R-only numerical optimization issue
- C5R-specific loss implementation failure

If a failure is demonstrably isolated to C5R and shared infrastructure remains valid:

preserve C5R failure evidence

and allow:

C6R → C7

to continue.


Do not casually classify a shared problem as arm-specific.

Document the evidence.

---

# 8. C6R path is mandatory

After all C6R hard gates pass:

run complete C6R formal training.

Select checkpoint using the exact historical C6 selection protocol.

Freeze:

theta
+
training-host materialized frozen R
+
manifest


Then deploy selected C6R to the canonical 3090 inference host.

Required:

C6R_DEPLOYMENT = PASS


Then run:

C6R Canonical60


This endpoint is mandatory because C6R is the matched control for C7.


No AIME-based tuning.

---

# 9. C7 path is mandatory

After all C7 hard gates pass:

run complete C7 formal training.

Select using:

minimum Shared16 Final-Hidden relative-MSE.


Freeze:

theta
+
training-host materialized frozen R
+
manifest


Deploy C7 to the same canonical 3090 inference runtime.

Required:

C7_DEPLOYMENT = PASS


Then run:

C7 Canonical60.


No AIME-based tuning.

---

# 10. Mandatory new AIME generation

By default, new Canonical60 generations are:

C6R:
60 samples

C7:
60 samples


Total mandatory new samples:

120


C5R contributes another 60 ONLY if:

C5R_AIME26_ROUTE = RUN_DUE_TO_MATERIAL_SHIFT


Historical results are reused:

FP:
52/60

Native INT8:
18/60

Hadamard:
41/60

historical C5:
33/60

historical C6:
36/60


Do NOT regenerate these archived baselines.

---

# 11. Primary endpoint

The PRIMARY comparison is:

C7 vs C6R


Report:

both correct

C6R only

C7 only

both wrong

C7 rescue

C7 regression

net change

accuracy difference

exact paired statistic

question-cluster uncertainty consistent with the existing project convention


This is the cleanest test of the objective change.


Secondary comparisons:

C7 vs Hadamard

C6R vs Hadamard

C7 vs historical C6

C6R vs historical C6

C7 vs FP


If C5R AIME runs:

C5R vs historical C5

C5R vs Hadamard

are secondary clean-room reproduction comparisons.


Do not turn C5R into a co-primary arm.

---

# 12. Historical C6 vs C6R

This comparison remains important.

If C6R is close to historical C6:

this supports the historical C6 lineage being broadly reproducible.


If C6R materially shifts from historical C6:

record:

CLEANROOM_C6_MATERIAL_SHIFT = YES


Do not automatically claim the old vGPU corruption caused the difference.

Possible causes include:

training runtime
numerical trajectory
environment
or other implementation differences.


The clean causal objective test remains:

C7 vs C6R

because both are matched on the A800.

---

# 13. C7 interpretation

## Case A

C7 > C6R
and
C7 > Hadamard

This is the strongest outcome.

Interpretation:

Final-Hidden supervision improves over matched local Functional training and closes/exceeds the fixed-Hadamard endpoint in the observed Canonical60 run.


## Case B

C7 > C6R
but
C7 <= Hadamard

Interpretation:

moving supervision downstream helps relative to C6,
but Final-Hidden fidelity alone does not fully solve the learned-rotation gap.


## Case C

C7 <= C6R

Interpretation:

Final-Hidden objective is not supported as a superior replacement for the matched C6 objective under this protocol.


Do not rescue C7 by changing checkpoint or loss after seeing AIME.

---

# 14. C5R does not determine C7 success

Do NOT require:

C5R reproduction PASS

as a prerequisite for interpreting C7 vs C6R,

provided:

- trusted A800 environment remains valid;
- shared runtime remains valid;
- C6R and C7 complete under identical matched conditions.


C5R is lineage evidence.

C6R vs C7 is the primary causal experiment.

---

# 15. Update artifact structure

Continue using the existing experiment root:

experiments/QWEN_GDN_A800_CLEANROOM_C5R_C6R_C7_MATCHED_V1/


Do NOT create a parallel duplicate experiment.


Add:

protocol/
  PROTOCOL_AMENDMENT_C5R_AUXILIARY_ENDPOINT_V1.md
  PROTOCOL_AMENDMENT_C5R_AUXILIARY_ENDPOINT_V1.json

analysis/
  C5R_REPRODUCTION_GATE.md


Existing directory structure remains unchanged.


If C5R AIME is skipped:

do not create fake result CSVs containing zero rows.

Instead write:

aime26/C5R_NOT_RUN.json

with:

reason
reproduction status
amendment reference


If C5R runs:

use the original planned C5R AIME directory.

---

# 16. Updated final status fields

Add:

C5R_REPRODUCTION =
CLOSE_REPRODUCTION /
MATERIAL_SHIFT /
INCONCLUSIVE /
NOT_COMPLETED


C5R_AIME26_ROUTE =
SKIP_NOT_NEEDED /
RUN_DUE_TO_MATERIAL_SHIFT /
SKIP_INCONCLUSIVE /
NOT_REACHED


C5R_AIME26 =
value/60 /
NOT_RUN


Keep:

C6R_AIME26 =
value/60 or NA

C7_AIME26 =
value/60 or NA


PRIMARY_MATCHED_COMPARISON =
C7_VS_C6R


C7_VS_C6R_NET =
value/NA


C7_VS_HADAMARD_NET =
value/NA


CLEANROOM_C6_MATERIAL_SHIFT =
YES/NO/UNKNOWN


---

# 17. Updated final verdict

Primary verdict should be based on the matched C6R-vs-C7 experiment.

Use one of:

C7_OBJECTIVE_SUPPORTED_AND_BEATS_HADAMARD

C7_OBJECTIVE_SUPPORTED_BUT_BELOW_HADAMARD

C7_OBJECTIVE_NOT_SUPPORTED

C6R_CLEANROOM_MATERIAL_SHIFT

TRAINING_BLOCKED

A800_ENVIRONMENT_REJECTED

INCONCLUSIVE


C5R reproduction status is reported separately and does not replace the primary verdict.

---

# 18. Stop rule

Do NOT automatically run C5R Canonical60 if C5R clean-room training broadly reproduces historical State-training behavior.

Do NOT run C5R AIME merely because GPU resources are available.

Do NOT skip C6R AIME.

Do NOT skip C7 AIME after both arms pass all formal training/deployment gates.

Do NOT add:

new loss terms

new seeds

new datasets

GSM8K

MATH-500

Full60

layer interventions

or C8


The final experiment should answer:

> Under one trusted A800 training environment, does changing the objective from C6R's local Functional supervision to C7's full-model Final-Hidden supervision improve recurrent-state INT8 reasoning?

Everything else is secondary.

---

## Application-time provenance

```json
{
  "amendment": "C5R_AUXILIARY_C6R_C7_PRIMARY_ENDPOINT_V1",
  "timestamp_utc": "2026-10-01T16:21:36.230679+00:00",
  "source_commit": "ad1b9fc19bd5fd5cc9014b7473f31edce22976a3",
  "canonical3090_source_commit": "ae3279701c5e548d43bfce4983e21d36abc5b4ba",
  "source_amendment_sha256": "14ecd15b5788dd22c9614b7cc57354e3f28966212dc6735a1ae601fd8e88722c",
  "original_preregistration_sha256": "009279fb84dd6d45f73bdfc27629a6d28de8d1a6dbfab64b65c83e9c4de5f3c8",
  "trusted_environment_sha256": "d44267ee6bc18acce6985ef18f9e3ce434c1a4cf6cae870bd9300e2cc4ae6816",
  "tmux_jobs": "qwen_a800_inventory: 1 windows (created Thu Oct  1 23:51:03 2026)\nqwen_a800_matched_qualified: 1 windows (created Thu Oct  1 23:11:10 2026)\nqwen_a800_posttraining: 1 windows (created Thu Oct  1 23:18:25 2026)\n",
  "formal_optimizer_updates_by_arm": {
    "C5R": 0,
    "C6R": 0,
    "C7": 0
  },
  "readiness_optimizer_updates_by_arm": {
    "C5R": 0,
    "C6R": 0,
    "C7": 0
  },
  "any_optimizer_update_occurred": false,
  "current_phase_by_arm": {
    "C5R": "UPDATE0_PRE_FORMAL_TRAINING",
    "C6R": "UPDATE0_PRE_FORMAL_TRAINING",
    "C7": "UPDATE0_PRE_FORMAL_TRAINING"
  },
  "completed_hard_gates": {
    "qualification/canary_run1.json": "PASS",
    "qualification/canary_run2.json": "PASS",
    "qualification/canary_run3.json": "PASS",
    "qualification/direct_checkpoint_run1.json": "PASS",
    "qualification/full_model_run1.json": "PASS",
    "qualification/direct_checkpoint_run2.json": "PASS",
    "qualification/full_model_run2.json": "PASS",
    "qualification/direct_checkpoint_run3.json": "PASS",
    "qualification/full_model_run3.json": "PASS",
    "qualification/lm_head_canary.json": "PASS",
    "qualification/fp_runtime_stability.json": "PASS",
    "qualification/teacher_cpu_checkpoint.json": "PASS",
    "qualification/update0_cpu_checkpoint.json": "PASS",
    "readiness/final_hidden_indexing_audit.json": "PASS",
    "readiness/update0_forward_parity.json": "PASS",
    "teacher/cache_files.json": "PASS",
    "teacher/teacher_cache_parity.json": "PASS"
  },
  "new_AIME_samples_at_amendment": 0,
  "reason": "C6R vs C7 primary matched-objective test; C5R auxiliary lineage reproduction; avoid unnecessary extra endpoint.",
  "changed_workflow": {
    "mandatory_endpoint_conditions": [
      "C6R",
      "C7"
    ],
    "mandatory_new_samples": 120,
    "conditional_C5R": "only MATERIAL_SHIFT justified by non-AIME evidence; CLOSE skip; INCONCLUSIVE default skip",
    "C5R_specific_failure": "preserve; only demonstrably isolated failure may be bypassed after shared-integrity evidence review",
    "primary_pair": [
      "C7",
      "C6R"
    ],
    "optional_endpoint_order": [
      "C6R",
      "C7",
      "C5R only if authorized material shift"
    ]
  },
  "unchanged_elements": [
    "qualification",
    "GPU weight integrity",
    "teacher targets",
    "corpus",
    "order",
    "tokenizer",
    "seed",
    "length",
    "captures",
    "recurrence",
    "C128 QDQ/writeback",
    "H64",
    "Adam",
    "LRs",
    "WD0",
    "clip1",
    "125 updates/1000 exposures",
    "validation/checkpoint cadence",
    "Hadamard theta0",
    "rotation parameterization",
    "objectives",
    "checkpoint selection",
    "canonical generation",
    "Frozen V4"
  ],
  "original_preregistration_rewritten": false,
  "current_GPU_worker_interrupted": false,
  "experiment_restarted": false,
  "routing_fallback": "INCONCLUSIVE evidence => SKIP_INCONCLUSIVE; no endpoint used to resolve uncertainty automatically",
  "statistical_families": "fixed six mandatory comparisons Holm; two conditional C5R comparisons separate auxiliary Holm family, declared before all new endpoint outcomes"
}
```

All source artifact hashes and the full process tree are in the companion JSON. Original preregistration and frozen scientific code remain unchanged.
