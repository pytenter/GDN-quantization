# QWEN_GDN_C6R_C7_CANONICAL60_V1

```text
EXPERIMENT = QWEN_GDN_C6R_C7_CANONICAL60_V1
PARENT_READINESS = QWEN_GDN_C6R_C7_4X3090_PERSISTENT_WORKER_READINESS_V2
CANONICAL60_PREMEASUREMENT_FREEZE_GATE = PASS
CANONICAL60_RUNTIME_IDENTITY_GATE = PASS
C6R_ROTATION_SHA = a886588b3613f66e6635e5874dbfaeaee26b850e05b4a1524da0bef952d22da6
C7_ROTATION_SHA = 57e7a11714405806385064f07438e5e2ac6b5952dd421606332b894cef3ee6e8
CONDITION_ORDER = C6R_THEN_C7
WORKER_ASSIGNMENT = RANK_MOD_4_SAME_GPU_ACROSS_CONDITIONS
C6R_RAW_COMPLETE = 60 / 60
C7_RAW_COMPLETE = 60 / 60
TOTAL_NEW_AIME_GENERATIONS = 120 / 120
C6R_RAW_OUTPUT_FREEZE_GATE = PASS
C7_RAW_OUTPUT_FREEZE_GATE = PASS
CANONICAL60_RAW_OUTPUT_CLOSURE_GATE = PASS
FROZEN_V4_IDENTITY_GATE = PASS
SCORING_COMPLETE = YES
C6R_CORRECT = 40 / 60
C7_CORRECT = 33 / 60
C7_RESCUE_C6R = 3
C7_REGRESSION_C6R = 10
BOTH_CORRECT = 30
BOTH_WRONG = 17
C7_MINUS_C6R_ACCURACY = -0.11666666666666667
PRIMARY_EXACT_P = 0.09228515625
PRIMARY_95CI_CLUSTER_BOOTSTRAP = [-0.25, 0.0]
PRIMARY_ENDPOINT_SIGNAL = NEGATIVE_DIRECTION_INCONCLUSIVE
PROTECTED_ARTIFACT_REVALIDATION_GATE = PASS
NEW_TRAINING_STARTED = NO
TRAINING_RERUN = NO
C5R_AIME_GENERATIONS = 0
PUSH_PERFORMED = NO
```

## PRIMARY: matched objective endpoint

C6R is the matched clean-A800 Local-Functional control; C7 is the Final-Hidden objective. Frozen exports are reused without re-export/reselection. C5R reproduction remains INCONCLUSIVE and its AIME endpoint is SKIP.

C6R: 40/60 (66.67%). C7: 33/60 (55.00%). Difference: -11.67 percentage points.

Rescue 3; regression 10; both correct 30; both wrong 17. Two-sided exact discordant binomial p=0.092285156; alpha=0.05; one primary comparison, no multiplicity adjustment.

95% question-cluster percentile CI: [-25.00, 0.00] percentage points. Resample 30 questions carrying both seeds; 10000 draws; seed20261003; numpy default_rng, linear percentiles.

Mechanical signal: NEGATIVE_DIRECTION_INCONCLUSIVE

Rescue sample IDs: aime26_04:seed2, aime26_13:seed1, aime26_22:seed1

Regression sample IDs: aime26_02:seed1, aime26_05:seed2, aime26_09:seed2, aime26_11:seed1, aime26_11:seed2, aime26_12:seed2, aime26_22:seed2, aime26_26:seed2, aime26_27:seed1, aime26_27:seed2

## SECONDARY: historical context

No historical result reader ran until the primary artifacts had been written. These are contextual comparisons, with no secondary p-values or confirmatory claims.

|Condition|Correct/60|Accuracy|
|---|---:|---:|
|FP|52/60|86.67%|
|Native|18/60|30.00%|
|H|41/60|68.33%|
|C5|33/60|55.00%|
|C6|36/60|60.00%|

|Comparison|Difference (pp)|Rescue|Regression|Both correct|Both wrong|
|---|---:|---:|---:|---:|---:|
|C6R_vs_H|-1.67|5|6|35|14|
|C7_vs_H|-13.33|5|13|28|14|

## DESCRIPTIVE: output behavior

All metrics reuse the existing canonical trajectory_record utility; no new model execution. Underlying120 rows and60 paired transitions are retained. No causal mechanism claim.

```json
{
  "behavior": {
    "C6R": {
      "samples": 60,
      "total_generated_tokens": 2085727,
      "mean_generated_tokens": 34762.11666666667,
      "median_generated_tokens": 32040.0,
      "normal_EOS_stop": 59,
      "length_ceiling": 1,
      "abstain": 4,
      "committed_claim_count_median": 2.0,
      "first_committed_claim_median": 79.0,
      "first_committed_claim_char_median": 49368.0,
      "post_commit_fraction_median": 0.19736398144242936,
      "answer_changes_median": 0.0,
      "retractions_median": 2.0,
      "oscillation_proxy_median": 3.0,
      "committed_aba_count_median": 0.0,
      "repeated_4gram_fraction_median": 0.4500434520164205,
      "unique_4gram_ratio_median": 0.5499565479835795,
      "repeated_8gram_fraction_median": 0.14971955447361057,
      "unique_8gram_ratio_median": 0.8502804455263895,
      "repeated_16gram_fraction_median": 0.032196260849418096,
      "unique_16gram_ratio_median": 0.9678037391505818,
      "answer_changes_total": 47,
      "retractions_total": 360
    },
    "C7": {
      "samples": 60,
      "total_generated_tokens": 1786038,
      "mean_generated_tokens": 29767.3,
      "median_generated_tokens": 27844.5,
      "normal_EOS_stop": 60,
      "length_ceiling": 0,
      "abstain": 9,
      "committed_claim_count_median": 2.0,
      "first_committed_claim_median": 67.5,
      "first_committed_claim_char_median": 41336.0,
      "post_commit_fraction_median": 0.25247938178513313,
      "answer_changes_median": 0.0,
      "retractions_median": 2.5,
      "oscillation_proxy_median": 3.0,
      "committed_aba_count_median": 0.0,
      "repeated_4gram_fraction_median": 0.4860662255134144,
      "unique_4gram_ratio_median": 0.5139337744865857,
      "repeated_8gram_fraction_median": 0.1762340454936726,
      "unique_8gram_ratio_median": 0.8237659545063274,
      "repeated_16gram_fraction_median": 0.04299311768539326,
      "unique_16gram_ratio_median": 0.9570068823146067,
      "answer_changes_total": 42,
      "retractions_total": 359
    }
  },
  "transitions": {
    "termination": {
      "normal->normal": 59,
      "ceiling->normal": 1
    },
    "answer": {
      "committed->committed": 49,
      "committed->abstain": 7,
      "abstain->committed": 2,
      "abstain->abstain": 2
    }
  },
  "ceiling_associations": {
    "rescue_C6R_ceiling": 0,
    "regression_C7_ceiling": 0,
    "C7_rescue_H_ceiling": 0,
    "C6R_rescue_H_ceiling": 0
  }
}
```

## Protocol, provenance, incidents and reproducibility

Premeasurement commit: 63eb42db33d1fbbf65450e03e4cf4f9165954c28

Source V2 final commit: cf850cfc06f5384f4e7aa36018fcc07fee45374d. New branch: exp/qwen-gdn-c6r-c7-canonical60-v1. Isolated shared local Git clone (remote Git1.8.3.1 lacks worktree support), with no source checkout modification.

Four independent single-GPU workers, 15 samples each. All C6R workers exit and exact byte-resolution NVML reclaim precede C6R freeze and fresh C7 processes. Same problem/seed uses the same UUID; no dynamic balancing. Manual HF81920 with source-derived sampler, unchanged QDQ/rotation/writeback. Gold-free generation manifest and separate scorer gold manifest. Atomic per-sample final files; all120 final outputs immutable and rehashed. No endpoint-dependent reruns or automatic retries.

Preparation audit disclosure: the source runtime JSON embeds already-frozen historical aggregate baseline fields and these were printed once during CPU source-only review, before any new output existed. No new endpoint scoring or per-sample historical result loading occurred. No protocol was chosen using new outcomes. See provenance/preparation_audit.json.

V1 full in-process unload remains FAIL at8519680 allocated bytes, unresolved. V2 persistent worker readiness is a different lifecycle and does not overwrite V1.

Reproduction entry (single-use; existing errors/outputs must be inspected, not blindly rerun):

```sh
cd /data/zypan/worktrees/qwen-gdn-c6r-c7-canonical60-v1
bash /data/zypan/worktrees/qwen-gdn-c6r-c7-canonical60-v1/experiments/QWEN_GDN_C6R_C7_CANONICAL60_V1/scripts/canonical60_offline.sh
```

## Evidence scope and STOP

AIME26 participated in earlier method development and diagnosis; this is preregistered matched-objective endpoint evidence, not a fully untouched external final benchmark. A fresh reasoning holdout is recommended after a final method freeze. Text failure associations are descriptive, not proof that a mechanism caused accuracy changes.

Potential follow-up only: QWEN_GDN_C6R_C7_ENDPOINT_MECHANISM_V1. NOT_STARTED. No training, C8, replay, new benchmark or push is authorized by this report. STOP and await explicit user decision.
