# Scientific reproducibility qualification — FAIL_AND_STOP

Experiment: `LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1`. Input commit: `08c831985d718feb44ea4a542a7ad6ed9598d54a`.

The actual historical numerical comparison failed; this is not a generated-code classifier rejection. No tolerance was changed.

Future-KL AUC: historical 0.7574607970852441; current 0.8296393686869621; absolute delta 0.07217857160171803; relative delta 9.529017%.

First stored FP-state hash divergence: token 11, layers ['2']. Same development token IDs and same GPU/runtime identity. Root cause is not established.

Full-model H/L6 rotated smoke and 3-process qualification were not reached/completed; no premeasurement commit or Discovery authorization was created.

```json
{
  "EXPERIMENT": "LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1",
  "QUALIFICATION_POLICY": "SCIENTIFIC_REPRODUCIBILITY",
  "STATUS": "FAIL_PRESERVED",
  "FINAL_ACTION": "FAIL_AND_STOP",
  "BLOCKER": "HISTORICAL_NUMERICAL_PARITY_FAILURE",
  "initial_input_commit": "08c831985d718feb44ea4a542a7ad6ed9598d54a",
  "timestamp_utc": "2026-10-07T13:23:01.813981+00:00",
  "FORMAL_MEASUREMENTS": 0,
  "DISCOVERY_PRIMARY_COMPLETE": "0/64",
  "DISCOVERY_SECONDARY_COMPLETE": "0/64",
  "GPU6_FORMAL_AUTHORIZED": "NO",
  "SCIENTIFIC_PREMEASUREMENT_COMMIT": "NOT_CREATED",
  "qualified_fresh_processes": "0/3",
  "gates": {
    "SOURCE_AND_SCIENTIFIC_ASSET_IDENTITY_GATE": "PASS",
    "RUNTIME_ENVIRONMENT_IDENTITY_GATE": "PASS",
    "BASIS_SEMANTICS_GATE": "NOT_EVALUATED",
    "HISTORICAL_K3_PARITY_GATE": "NOT_FULLY_QUALIFIED",
    "HISTORICAL_JPROP_PARITY_GATE": "NOT_EVALUATED",
    "FUTURE_KL_HARNESS_PARITY_GATE": "FAIL",
    "FRESH_PROCESS_NUMERIC_DETERMINISM_GATE": "NOT_EVALUATED",
    "HARDWARE_ASSIGNMENT_GATE": "PASS",
    "STORAGE_CAPACITY_GATE": "PASS"
  },
  "partial_smoke_tests": "Existing small historical K3/kernel fixture tests PASS; this does not establish the full qualification gate.",
  "coverage": {
    "real_model_load": "PASS",
    "FP_model_128_token_capture": "COMPLETE",
    "H_replay_persistent_INT8_128_tokens": "COMPLETE",
    "future_KL_model_fixture": "COMPLETE_BUT_HISTORICALLY_DIFFERENT",
    "full_model_H_rotated_smoke": "NOT_REACHED",
    "full_model_learned_L6_rotated_smoke": "NOT_REACHED"
  },
  "historical_reference": {
    "path": "/data/zypan/worktrees/ling-observability-provenance-amendment-v1/checkout/experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1/smoke/repeat0/result.json",
    "sha256": "9d902526babf110261b86c56533986f5eaf70b394ea1078b4208560f4024829d",
    "size": 910443
  },
  "current_result": {
    "path": "/data/zypan/experiment_runs/ling_observability_scientific_repro_v3/fresh_process_01/smoke/repeat0/result.json",
    "sha256": "16b9f820490b1370a4c567252b6dac2d15562c422fde3564a973cadf7c21a638"
  },
  "STOP_sha256": "e9e2eb0d7bc03e3d21c840e6f0878989430891858a86abf447f880a0c3444f3f",
  "comparisons": {
    "scores": true,
    "target": true,
    "endpoint": true,
    "FP_step_hashes": true
  },
  "development_token_ids_identical": true,
  "development_token_sha256": "473d217bb003242a57029ec75ed9e549457bc45d3033d14e2d2297ce4adcf8bc",
  "first_FP_state_divergence": {
    "one_based_token": 11,
    "zero_based_position": 10,
    "layers": [
      "2"
    ],
    "current": {
      "0": "1fe8c499594b0d62a356066507d5980a3843a080f922051714103445dc2b846a",
      "1": "a1ebc8882b1ea898cbd701cd5cdcf6b682bc990a576794d0d7cb8d277a7e0e52",
      "10": "5879010b3770ab4926058f79dc1f7f871980063252c38f5679c2facc48bbce14",
      "12": "55bbf20b2eb6f59e539d191f90c5567bcea2723a0682ba448983e1b8fbea077a",
      "13": "1dfc48c8e7cb2af1ccd12c8257f34f2a8a258bfc392a93005162b79a05f43800",
      "14": "7ab692b0a0fa28c8c43d8ce56e459f366fb3b361a0b2ca2b8d6d4fc52af9dbfb",
      "16": "df4844cf0ba7e41cb750aa602aed679746b8a27029e6f763c99502a374f4f4a0",
      "17": "9630eb131767581f03e7bdfad6cf7e000ef59616d9a7d9ada0b4f013fb96c3a4",
      "18": "beaff40f7ff3a4c9709d9e090d32d8484d8526974bf857960533df7585496168",
      "2": "7e700d44e37e59ef8eac427417c0a48ad2db8c9abdd29e040481e3e67ebd17bb",
      "20": "2b27fc82bf2f4c871c3444ce12063b96fbfa5cd113e8dc2aeab974716753dfe7",
      "21": "485f848dcb0e542579a59780c119dc627b39ab34ddfcdaf8bf506040cc3b146f",
      "22": "e55b0ae44a3a5cb0c983ff2f83701018d50c5fc8001fbf8701a83a7b5bb898df",
      "4": "a2640e370e375d477a322fb5d4c87b94f86aff84c48d1d55ae3db3bd8c29b02f",
      "5": "7b5b80fabd9e074af072562767fad3b934ea8330bcb365e0bc30e1adb4e92244",
      "6": "bddd4bfd8f9795e77750c5e304d95665ad42a0ce13030443b5a8377869c9619e",
      "8": "a6fd57924349eb1accb592cdcfc744a3fb1c78e48a33692e63e1d9761ccc8dc9",
      "9": "44d2657f1c7ef1ae56dd67d9e1ee5ff226fc549c416bc3492fcacac24b8479c6"
    },
    "historical": {
      "0": "1fe8c499594b0d62a356066507d5980a3843a080f922051714103445dc2b846a",
      "1": "a1ebc8882b1ea898cbd701cd5cdcf6b682bc990a576794d0d7cb8d277a7e0e52",
      "10": "5879010b3770ab4926058f79dc1f7f871980063252c38f5679c2facc48bbce14",
      "12": "55bbf20b2eb6f59e539d191f90c5567bcea2723a0682ba448983e1b8fbea077a",
      "13": "1dfc48c8e7cb2af1ccd12c8257f34f2a8a258bfc392a93005162b79a05f43800",
      "14": "7ab692b0a0fa28c8c43d8ce56e459f366fb3b361a0b2ca2b8d6d4fc52af9dbfb",
      "16": "df4844cf0ba7e41cb750aa602aed679746b8a27029e6f763c99502a374f4f4a0",
      "17": "9630eb131767581f03e7bdfad6cf7e000ef59616d9a7d9ada0b4f013fb96c3a4",
      "18": "beaff40f7ff3a4c9709d9e090d32d8484d8526974bf857960533df7585496168",
      "2": "e01f59955061d91b22488ebcd5b1dc03af6009d59e097b1a6ef9d906558c59b0",
      "20": "2b27fc82bf2f4c871c3444ce12063b96fbfa5cd113e8dc2aeab974716753dfe7",
      "21": "485f848dcb0e542579a59780c119dc627b39ab34ddfcdaf8bf506040cc3b146f",
      "22": "e55b0ae44a3a5cb0c983ff2f83701018d50c5fc8001fbf8701a83a7b5bb898df",
      "4": "a2640e370e375d477a322fb5d4c87b94f86aff84c48d1d55ae3db3bd8c29b02f",
      "5": "7b5b80fabd9e074af072562767fad3b934ea8330bcb365e0bc30e1adb4e92244",
      "6": "bddd4bfd8f9795e77750c5e304d95665ad42a0ce13030443b5a8377869c9619e",
      "8": "a6fd57924349eb1accb592cdcfc744a3fb1c78e48a33692e63e1d9761ccc8dc9",
      "9": "44d2657f1c7ef1ae56dd67d9e1ee5ff226fc549c416bc3492fcacac24b8479c6"
    }
  },
  "future_KL": {
    "historical": 0.757460797085244,
    "current": 0.8296393686869621,
    "absolute_delta": 0.07217857160171803,
    "relative_delta": 0.0952901745931481
  },
  "scores": {
    "current": {
      "64": {
        "E_anchor_sha256": "839d70263e06ebfae9d3059118b50d537d0b817d41ecc707f6fcc5a5bd7be439",
        "HISTORICAL_J_PROP": null,
        "J_ACC": null,
        "J_INJ": null,
        "LOCAL_K3_ACCUMULATED": 2.513036991776486e-05,
        "POR": 0.0009463450227882475,
        "POR_GAIN": 7.588326197695137e-05,
        "PULSE_PROP_ENERGY_AUC": 697.4791462445447,
        "RAW_ACC_ERROR": 12.471064080978602,
        "future_tokens": 64,
        "top_K3_events": [
          {
            "k3_energy": 3.880757278422025e-05,
            "state_energy": 9.224871267143918,
            "token": 120
          },
          {
            "k3_energy": 2.4811298271591135e-05,
            "state_energy": 12.22128995266178,
            "token": 69
          },
          {
            "k3_energy": 2.1252333545569988e-05,
            "state_energy": 12.271708055091928,
            "token": 68
          },
          {
            "k3_energy": 1.9781554557603556e-05,
            "state_energy": 9.157090133635425,
            "token": 121
          },
          {
            "k3_energy": 1.9399357065257795e-05,
            "state_energy": 11.203660590130877,
            "token": 92
          }
        ]
      }
    },
    "historical": {
      "64": {
        "E_anchor_sha256": "c39d65ae6d42ddd15c1f307603fe08b96172fc4e99a26398c7c451ad4daf1f94",
        "HISTORICAL_J_PROP": null,
        "J_ACC": null,
        "J_INJ": null,
        "LOCAL_K3_ACCUMULATED": 2.513036991282448e-05,
        "POR": 0.0009452134145642127,
        "POR_GAIN": 7.579252327036822e-05,
        "PULSE_PROP_ENERGY_AUC": 697.481726867481,
        "RAW_ACC_ERROR": 12.47106408098373,
        "future_tokens": 64,
        "top_K3_events": [
          {
            "k3_energy": 3.864547006697683e-05,
            "state_energy": 9.22283476341499,
            "token": 120
          },
          {
            "k3_energy": 2.4811298272165013e-05,
            "state_energy": 12.221289952665023,
            "token": 69
          },
          {
            "k3_energy": 2.1252333543847123e-05,
            "state_energy": 12.271708055095477,
            "token": 68
          },
          {
            "k3_energy": 1.9692015854307224e-05,
            "state_energy": 9.155907053336069,
            "token": 121
          },
          {
            "k3_energy": 1.9381247764285003e-05,
            "state_energy": 11.203497548913028,
            "token": 92
          }
        ]
      }
    }
  },
  "endpoint": {
    "current": {
      "FP": "d8b74e13528f2d735b8d9e0cc48d368d7350e65313c15c0999e29f0be0483b42",
      "HISTORICAL_J_PROP": null,
      "J_ACC": null,
      "J_INJ": null,
      "LOCAL_K3_ACCUMULATED_32K": 4.310835574977715e-05,
      "Q": "ccd1fdfe881c59f8dc6d6d3a4db3151f483a06f8f609bd0726d69d28c9a2b35e",
      "RAW_ACC_ERROR_32K": 32.062053362915904
    },
    "historical": {
      "FP": "9b91211c1e216111e05ec9a8c71468efaef0b8ebb46938681774d6319f943cdc",
      "HISTORICAL_J_PROP": null,
      "J_ACC": null,
      "J_INJ": null,
      "LOCAL_K3_ACCUMULATED_32K": 4.0791091271020536e-05,
      "Q": "307e99a73bc8268b999c09e51e4fae81790dc31df98fe865f55ee53e1f0a8b15",
      "RAW_ACC_ERROR_32K": 32.14276595541895
    }
  },
  "FP_anchor": {
    "current": "575315c415235e608d19fc209815a8b3a4d2d7abf609286125c07f254b51eef4",
    "historical": "a8f4a2b157ce999ce6bd78c45499df3537c67bd4ca62787eb8c7746a7a63253f"
  },
  "hardware_and_runtime_identity_equal": true,
  "hardware_identity": {
    "CUDA": "12.8",
    "cudnn_benchmark": false,
    "cudnn_deterministic": false,
    "deterministic_algorithms": false,
    "driver": "580.173.02",
    "environment": {
      "CUBLAS_WORKSPACE_CONFIG": null,
      "CUDA_VISIBLE_DEVICES": "6",
      "FLA_CONFIG_DIR": "/data/zypan/worktrees/ling-long-horizon-propagation-confirm-v1/inherited/data01/user2/worktrees/ling-l6-l7-objective-horizon-diagnosis-v1/experiments/LING_FUSED_RMSNORM_GATED_DETERMINISM_CLOSURE_V1/runtime/historical/fixed_fla_configs",
      "LING_GATED_RMSNORM_FIXED_CONFIG": "BT16_W8",
      "PYTHONHASHSEED": "0"
    },
    "gpu_uuid": "GPU-de53f243-b271-521c-d875-d96611ed8574",
    "host": "nlpg-SYS-4029GP-TRT",
    "matmul_tf32": false,
    "physical_gpu": 6,
    "processes": [],
    "python": "3.11.16 (main, Sep  2 2026, 23:39:52) [GCC 15.3.0]",
    "torch": "2.7.1+cu128"
  },
  "free_bytes": 801385336832,
  "required_bytes": 23622320128,
  "root_cause": "NOT_ESTABLISHED; no claim that the clean cache caused the numerical difference",
  "native_generated_full_closure_required": false,
  "native_rebuild_byte_identity_required": false,
  "old_Attempt12_cache_used": false,
  "original_tolerance_unchanged": true,
  "no_auto_retry": true,
  "all_historical_files_byte_identical": true,
  "cleanup_ledger_bookkeeping_remaining": 1,
  "actual_task_processes": [],
  "actual_GPU6_processes": "",
  "L9_TRAINING_EXECUTED": "NO",
  "NEW_ROTATION_TRAINING_EXECUTED": "NO",
  "AIME_EXECUTED": "NO",
  "FREE_GENERATION_EXECUTED": "NO",
  "FP_REFERENCES_MIXED_ACROSS_HARDWARE": "NO"
}
```
