# L6/L7 recurrent training report

Status: **PASS**

The numerical recurrent state is continuous for all 512 tokens. Gradient graphs are detached every 128 tokens, selected only by the preregistered resource-feasibility probe. Detaching never replaces the student state with a teacher state.

## Semantic gates

- `TRAIN_QDQ_MATCH`: PASS
- `GRADIENT_GATE`: PASS
- `REAL_RECURRENT_WRITEBACK_GATE`: PASS
- `RECURRENT_STATE_PROVENANCE_GATE`: PASS
- `AIME26 overlap`: NO

## Old and new trajectories

```text
Old L4/L5
FP teacher state_t -> single token -> QDQ -> loss -> reset to FP teacher

New L6/L7
student INT8 state_t -> recurrent update -> QDQ -> student INT8 state_t+1
                                                       |
                                                       +-> WRITE BACK -> next token consumes it
```

The provenance probe records teacher/previous/post-QDQ/next-consumed hashes. For every checked transition, the next consumed hash equals the preceding post-QDQ hash; after quantization error appears, the student and teacher hashes differ.

## Completed training

```json
{
  "L6": {
    "best_step": 100,
    "best_validation_primary": 0.003573625930584967,
    "checkpoint_selection": "MIN_NON_AIME_VALIDATION_PRIMARY",
    "gradient_horizon": 128,
    "numerical_recurrent_horizon": 512,
    "objective": "L6_RECURRENT_DENSE_STATE",
    "orthogonality": {
      "max": 1.8477439880371094e-06,
      "median": 1.430511474609375e-06,
      "p95": 1.8477439880371094e-06,
      "per_layer": [
        {
          "layer_id": 0,
          "max_abs_rt_r_minus_i": 1.3113021850585938e-06
        },
        {
          "layer_id": 1,
          "max_abs_rt_r_minus_i": 1.1920928955078125e-06
        },
        {
          "layer_id": 2,
          "max_abs_rt_r_minus_i": 1.3709068298339844e-06
        },
        {
          "layer_id": 4,
          "max_abs_rt_r_minus_i": 1.6689300537109375e-06
        },
        {
          "layer_id": 5,
          "max_abs_rt_r_minus_i": 1.1324882507324219e-06
        },
        {
          "layer_id": 6,
          "max_abs_rt_r_minus_i": 1.3113021850585938e-06
        },
        {
          "layer_id": 8,
          "max_abs_rt_r_minus_i": 1.5497207641601562e-06
        },
        {
          "layer_id": 9,
          "max_abs_rt_r_minus_i": 1.1920928955078125e-06
        },
        {
          "layer_id": 10,
          "max_abs_rt_r_minus_i": 1.5497207641601562e-06
        },
        {
          "layer_id": 12,
          "max_abs_rt_r_minus_i": 1.430511474609375e-06
        },
        {
          "layer_id": 13,
          "max_abs_rt_r_minus_i": 1.8477439880371094e-06
        },
        {
          "layer_id": 14,
          "max_abs_rt_r_minus_i": 9.5367431640625e-07
        },
        {
          "layer_id": 16,
          "max_abs_rt_r_minus_i": 1.6689300537109375e-06
        },
        {
          "layer_id": 17,
          "max_abs_rt_r_minus_i": 1.430511474609375e-06
        },
        {
          "layer_id": 18,
          "max_abs_rt_r_minus_i": 1.430511474609375e-06
        },
        {
          "layer_id": 20,
          "max_abs_rt_r_minus_i": 1.430511474609375e-06
        },
        {
          "layer_id": 21,
          "max_abs_rt_r_minus_i": 1.6689300537109375e-06
        },
        {
          "layer_id": 22,
          "max_abs_rt_r_minus_i": 1.3113021850585938e-06
        }
      ],
      "status": "PASS",
      "worst_layer": 13
    },
    "runtime_seconds": 1642.9600839614868,
    "status": "PASS",
    "steps_completed": 100
  },
  "L7": {
    "best_step": 100,
    "best_validation_primary": 0.0023113065981306136,
    "checkpoint_selection": "MIN_NON_AIME_VALIDATION_PRIMARY",
    "gradient_horizon": 128,
    "numerical_recurrent_horizon": 512,
    "objective": "L7_RECURRENT_DENSE_FUNCTIONAL",
    "orthogonality": {
      "max": 2.0265579223632812e-06,
      "median": 1.4901161193847656e-06,
      "p95": 2.0265579223632812e-06,
      "per_layer": [
        {
          "layer_id": 0,
          "max_abs_rt_r_minus_i": 1.8477439880371094e-06
        },
        {
          "layer_id": 1,
          "max_abs_rt_r_minus_i": 1.4901161193847656e-06
        },
        {
          "layer_id": 2,
          "max_abs_rt_r_minus_i": 1.9073486328125e-06
        },
        {
          "layer_id": 4,
          "max_abs_rt_r_minus_i": 1.6689300537109375e-06
        },
        {
          "layer_id": 5,
          "max_abs_rt_r_minus_i": 1.6093254089355469e-06
        },
        {
          "layer_id": 6,
          "max_abs_rt_r_minus_i": 1.3113021850585938e-06
        },
        {
          "layer_id": 8,
          "max_abs_rt_r_minus_i": 1.430511474609375e-06
        },
        {
          "layer_id": 9,
          "max_abs_rt_r_minus_i": 1.430511474609375e-06
        },
        {
          "layer_id": 10,
          "max_abs_rt_r_minus_i": 1.3113021850585938e-06
        },
        {
          "layer_id": 12,
          "max_abs_rt_r_minus_i": 1.6689300537109375e-06
        },
        {
          "layer_id": 13,
          "max_abs_rt_r_minus_i": 1.8477439880371094e-06
        },
        {
          "layer_id": 14,
          "max_abs_rt_r_minus_i": 1.4901161193847656e-06
        },
        {
          "layer_id": 16,
          "max_abs_rt_r_minus_i": 1.0728836059570312e-06
        },
        {
          "layer_id": 17,
          "max_abs_rt_r_minus_i": 2.0265579223632812e-06
        },
        {
          "layer_id": 18,
          "max_abs_rt_r_minus_i": 1.430511474609375e-06
        },
        {
          "layer_id": 20,
          "max_abs_rt_r_minus_i": 1.3709068298339844e-06
        },
        {
          "layer_id": 21,
          "max_abs_rt_r_minus_i": 1.4901161193847656e-06
        },
        {
          "layer_id": 22,
          "max_abs_rt_r_minus_i": 1.430511474609375e-06
        }
      ],
      "status": "PASS",
      "worst_layer": 17
    },
    "runtime_seconds": 3202.139124393463,
    "status": "PASS",
    "steps_completed": 100
  }
}
```

L4 and L6 use the same audited relative state-reconstruction objective and reduction. L5 and L7 use the same audited local attention out-projection objective and reduction. The intended semantic change is `FP_STATE_RESET_EACH_TOKEN -> REAL_RECURRENT_INT8_WRITEBACK`.
