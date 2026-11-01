# Complete premeasurement freeze with authorized four-GPU scheduling

One document capture on GPU0, shared sealed drive; Native/H/C6R/C7 on GPUs0/1/2/3 in independent processes. Join all four successful workers and verify every result before exact shard release, then advance to the next frozen document. No production mathematics, dtype, drive schema, metric, threshold, horizon, input, rotation or release rule changed.

{
  "EXPERIMENT": "QWEN_GDN_FROZEN_DRIVE_ANCHOR_MECHANISM_V1",
  "scientific_type": "PROSPECTIVE_CONTROLLED_MECHANISM_DISCOVERY",
  "documents": {
    "CANDIDATE_CORPUS": "long_bench/repobench-p",
    "CANDIDATE_POOL_REVISION": "7dcaafcb4974ad5893d9f22a7f36fd70fa811bfa8c5fc5448a931ddc0917eb9a",
    "CONFIRMATION_EXECUTED": 0,
    "CONFIRMATION_RULE_FROZEN": "YES",
    "DISCOVERY_DOCUMENT_SELECTION_GATE": "PASS",
    "DISCOVERY_DOC_COUNT": 4,
    "confirmation_inputs_materialized": false,
    "documents": [
      {
        "canonical_doc_id": "long_bench/repobench-p/test/d318e101c2458d5e844d488bad7f6ce0e147f5075a880514",
        "container_file_size": 24157264,
        "content_SHA256": "21bd7bf7146cfddcf3506629876b49161b78c93f605722a5ead228663f9cc808",
        "duplicate": false,
        "eligible": true,
        "exclusion_status": "NOT_IN_KNOWN_PROJECT_OR_RESERVED_CORPUS",
        "file_size": 145097,
        "frozen_token_count": 32768,
        "readable_nonempty": true,
        "record_index": 289,
        "relative_path": "long_bench-test.arrow#row=289",
        "selection_key": "13117f06748a20b0be10599e7f223fe93864c9223d3e1468c6199bea1f2ca4e2",
        "selection_rank": 1,
        "source_path": "/data/ydai/qwen_model_cache/datasets/long_bench/repobench-p/1.0.0/0384ffc1cabf36a2322f408bfdc21ea54d6a2704f2b1e17ca49d89035e666f1e/long_bench-test.arrow",
        "text_path": "inputs/discovery_01.txt",
        "text_sha256": "21bd7bf7146cfddcf3506629876b49161b78c93f605722a5ead228663f9cc808",
        "token_count": 41821,
        "token_path": "inputs/discovery_01_tokens.json",
        "token_sha256": "1f1666ec6108d398a0538714a2745dfa349b5e3b80e6552f7ecd66047a611f5b"
      },
      {
        "canonical_doc_id": "long_bench/repobench-p/test/bd788a58347b1655aec6622c6ba1d8e6687199abb90a510d",
        "container_file_size": 24157264,
        "content_SHA256": "786b3c4faa8695be6c568368171629bd02594d9cc7b84cb432fbc931a00027f3",
        "duplicate": false,
        "eligible": true,
        "exclusion_status": "NOT_IN_KNOWN_PROJECT_OR_RESERVED_CORPUS",
        "file_size": 145013,
        "frozen_token_count": 32768,
        "readable_nonempty": true,
        "record_index": 474,
        "relative_path": "long_bench-test.arrow#row=474",
        "selection_key": "21e9e3b84925ef9378c660a03dd007bde68530288fdde6ec1aa2172533b06d52",
        "selection_rank": 2,
        "source_path": "/data/ydai/qwen_model_cache/datasets/long_bench/repobench-p/1.0.0/0384ffc1cabf36a2322f408bfdc21ea54d6a2704f2b1e17ca49d89035e666f1e/long_bench-test.arrow",
        "text_path": "inputs/discovery_02.txt",
        "text_sha256": "786b3c4faa8695be6c568368171629bd02594d9cc7b84cb432fbc931a00027f3",
        "token_count": 35282,
        "token_path": "inputs/discovery_02_tokens.json",
        "token_sha256": "e17cb43e66e512eba02779821e0f283b254d24f87074a0d392427f5a7fa9d95d"
      },
      {
        "canonical_doc_id": "long_bench/repobench-p/test/3a4de092b20b1c9dd64199a1d382ddd2a906a37ed2dcda96",
        "container_file_size": 24157264,
        "content_SHA256": "27eb1065f7d47c28a60dc480d178822efe41b1b29d96835d95077ce1e93596b3",
        "duplicate": false,
        "eligible": true,
        "exclusion_status": "NOT_IN_KNOWN_PROJECT_OR_RESERVED_CORPUS",
        "file_size": 133592,
        "frozen_token_count": 32768,
        "readable_nonempty": true,
        "record_index": 154,
        "relative_path": "long_bench-test.arrow#row=154",
        "selection_key": "2b800e9637a9941bd02b9f6b30f4276aef0d89f0ab7ddb39848f9d61f1331c54",
        "selection_rank": 3,
        "source_path": "/data/ydai/qwen_model_cache/datasets/long_bench/repobench-p/1.0.0/0384ffc1cabf36a2322f408bfdc21ea54d6a2704f2b1e17ca49d89035e666f1e/long_bench-test.arrow",
        "text_path": "inputs/discovery_03.txt",
        "text_sha256": "27eb1065f7d47c28a60dc480d178822efe41b1b29d96835d95077ce1e93596b3",
        "token_count": 34640,
        "token_path": "inputs/discovery_03_tokens.json",
        "token_sha256": "af67c991879e8a3c26a3b0d6dce9df7bc5b1fd4d239ce882ed8e7fd30d6347d2"
      },
      {
        "canonical_doc_id": "long_bench/repobench-p/test/5eab3c8573bc773090ab7ea27211875c61890ca4e2f92711",
        "container_file_size": 24157264,
        "content_SHA256": "f39fcb5bdb4f5e413339d0d4dfeeaa204c37939b11c422fe2fe70a19a6583f68",
        "duplicate": false,
        "eligible": true,
        "exclusion_status": "NOT_IN_KNOWN_PROJECT_OR_RESERVED_CORPUS",
        "file_size": 127861,
        "frozen_token_count": 32768,
        "readable_nonempty": true,
        "record_index": 374,
        "relative_path": "long_bench-test.arrow#row=374",
        "selection_key": "51f689907e8890df737e8d672981b1557081bdc8341de72d0bf1d84deb5056f2",
        "selection_rank": 4,
        "source_path": "/data/ydai/qwen_model_cache/datasets/long_bench/repobench-p/1.0.0/0384ffc1cabf36a2322f408bfdc21ea54d6a2704f2b1e17ca49d89035e666f1e/long_bench-test.arrow",
        "text_path": "inputs/discovery_04.txt",
        "text_sha256": "f39fcb5bdb4f5e413339d0d4dfeeaa204c37939b11c422fe2fe70a19a6583f68",
        "token_count": 35637,
        "token_path": "inputs/discovery_04_tokens.json",
        "token_sha256": "e976679d76762bd4769e2375715b3542446a592e66a480ccb13953e1c1ca1485"
      }
    ],
    "status": "SELECTED_AND_FROZEN"
  },
  "anchors": {
    "ANCHOR_CONDITION_COUNT": 4,
    "ANCHOR_IDENTITY_GATE": "PASS",
    "anchors": [
      {
        "anchor_id": "A0",
        "condition": "Native INT8-C128",
        "identity_rule": "No rotation; exact identity map",
        "matrix_dtype": "torch.float32",
        "matrix_sha256": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
        "matrix_shape": [
          128,
          128
        ],
        "native_basis": "S_native = S_stored",
        "orthogonality_max_abs": 0.0,
        "per_layer_matrix_sha256": {
          "0": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
          "1": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
          "10": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
          "12": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
          "13": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
          "14": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
          "16": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
          "17": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
          "18": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
          "2": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
          "20": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
          "21": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
          "22": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
          "24": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
          "25": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
          "26": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
          "28": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
          "29": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
          "30": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
          "4": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
          "5": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
          "6": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
          "8": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa",
          "9": "66957d69fd7a0016b2ee4a95af26a8f7b93a932819cba34db82effad451f08fa"
        },
        "qk_rule": "Original normalize_qk then no rotation"
      },
      {
        "anchor_id": "A1",
        "condition": "Hadamard",
        "identity_rule": "Unchanged formal hadamard(128) function; FP32 normalized Sylvester H128",
        "matrix_dtype": "torch.float32",
        "matrix_sha256": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
        "matrix_shape": [
          128,
          128
        ],
        "native_basis": "S_native = H128 @ S_stored",
        "orthogonality_max_abs": 5.960464477539063e-08,
        "per_layer_matrix_sha256": {
          "0": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
          "1": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
          "10": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
          "12": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
          "13": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
          "14": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
          "16": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
          "17": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
          "18": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
          "2": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
          "20": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
          "21": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
          "22": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
          "24": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
          "25": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
          "26": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
          "28": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
          "29": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
          "30": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
          "4": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
          "5": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
          "6": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
          "8": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68",
          "9": "6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68"
        },
        "qk_rule": "Original normalize_qk then q/k.float @ H128"
      },
      {
        "anchor_id": "A2",
        "condition": "C6R",
        "direct_frozen_export_load": true,
        "export_file_sha256": "a886588b3613f66e6635e5874dbfaeaee26b850e05b4a1524da0bef952d22da6",
        "export_path": "/data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1/experiments/QWEN_GDN_PROPAGATION_SURROGATE_VALIDATION_V1/provenance/recovered_anchors/matrix/C6R.pt",
        "matrix_dtype": "torch.float32",
        "matrix_shape": [
          128,
          128
        ],
        "native_basis": "S_native = H128 @ (Delta @ S_stored); separate products, never fused",
        "original_provenance": {
          "condition": "C6R",
          "exposure": 752,
          "matrix": "/data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1/experiments/QWEN_GDN_PROPAGATION_SURROGATE_VALIDATION_V1/provenance/recovered_anchors/matrix/C6R.pt",
          "matrix_sha256": "a886588b3613f66e6635e5874dbfaeaee26b850e05b4a1524da0bef952d22da6",
          "original_matrix": "/root/autodl-tmp/GDN-A800-cleanroom/experiments/QWEN_GDN_A800_CLEANROOM_C5R_C6R_C7_MATCHED_V1/training/C6R/frozen_rotations/update_094_exposure_0752.pt",
          "original_theta": "/root/autodl-tmp/GDN-A800-cleanroom/experiments/QWEN_GDN_A800_CLEANROOM_C5R_C6R_C7_MATCHED_V1/training/C6R/checkpoints/update_094_exposure_0752.pt",
          "selection_sha256": "b310c0cfe8c4908ef95cb8bb9e04964086e65359873d47386ee515b253bdb011",
          "theta": "/data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1/experiments/QWEN_GDN_PROPAGATION_SURROGATE_VALIDATION_V1/provenance/recovered_anchors/theta/update_094_exposure_0752.pt",
          "theta_sha256": "620a35101b95cadd16f50cd1b869aeef54da9db0b3c76076fe82a374a6ee17cf",
          "update": 94
        },
        "orthogonality_max_abs": 2.384185791015625e-06,
        "per_layer_matrices": [
          {
            "layer": 0,
            "matrix_sha256": "2c1bb2bd4e9f3ee4157ceea809f3178a6ebb1d5d423cbd5abaf23551e6d494f9",
            "orthogonality_max_abs": 1.6689300537109375e-06
          },
          {
            "layer": 1,
            "matrix_sha256": "3d6e2448e14eb69ea0d5c44bb550bbe75b72da33655b334098ad03129af065d4",
            "orthogonality_max_abs": 2.1457672119140625e-06
          },
          {
            "layer": 2,
            "matrix_sha256": "4e2e976149265cbe41cf664e381a96aef9775c5c05ec471c7e00cbeb6e10bc06",
            "orthogonality_max_abs": 1.7881393432617188e-06
          },
          {
            "layer": 4,
            "matrix_sha256": "1d86b21a0a8a3a41ef9bc106a7a5330f6b776ef1721a44377ccbbecbeada1980",
            "orthogonality_max_abs": 1.3113021850585938e-06
          },
          {
            "layer": 5,
            "matrix_sha256": "e376b6fc4b4f81b9a88c002ae8348a9aebb5720240bb007e6c4de80f68d6dc16",
            "orthogonality_max_abs": 2.1457672119140625e-06
          },
          {
            "layer": 6,
            "matrix_sha256": "a5643980eb8ed8ed37aed364f0b9bbda74614e25286a7bdc6007bdc1df042898",
            "orthogonality_max_abs": 2.384185791015625e-06
          },
          {
            "layer": 8,
            "matrix_sha256": "0e3121439ff233228e2f6c546721fc442e0987d99515681a56636237e6c63d2c",
            "orthogonality_max_abs": 1.5497207641601562e-06
          },
          {
            "layer": 9,
            "matrix_sha256": "370c4d120d4305dacb0a2178d28ec45799be62d6d702919773fc94935d29b500",
            "orthogonality_max_abs": 1.3113021850585938e-06
          },
          {
            "layer": 10,
            "matrix_sha256": "5b75a1e959e4c87bba918d7774481504556c813428b93b931e7d3a4152554c93",
            "orthogonality_max_abs": 1.4901161193847656e-06
          },
          {
            "layer": 12,
            "matrix_sha256": "9f8cf3fd0722c00b80adf6ff9f0501314b08a43cd9b73943ff7b361b34e74dd3",
            "orthogonality_max_abs": 1.430511474609375e-06
          },
          {
            "layer": 13,
            "matrix_sha256": "22418e9a9f517aa3ff44b6ecb8a36d4245338d7392e4ef8e5d11d74073a4f66f",
            "orthogonality_max_abs": 1.7881393432617188e-06
          },
          {
            "layer": 14,
            "matrix_sha256": "1764fb8654b840415c01f1e78f45355b388062ce4dd6142ef825e34470b21742",
            "orthogonality_max_abs": 1.6689300537109375e-06
          },
          {
            "layer": 16,
            "matrix_sha256": "ab86987027626fedb0b5b5dce89180b9eb29e683bca1450007e41be085d73457",
            "orthogonality_max_abs": 1.7285346984863281e-06
          },
          {
            "layer": 17,
            "matrix_sha256": "6b5c14e9d4bc995780e2ee1ade9027172068f59e90111e66bbc0780528ce2135",
            "orthogonality_max_abs": 1.6689300537109375e-06
          },
          {
            "layer": 18,
            "matrix_sha256": "bae52d45da5f27ca49ba57d86584d08f48c7fc631a1a4c652d431bd6afa29fdc",
            "orthogonality_max_abs": 1.430511474609375e-06
          },
          {
            "layer": 20,
            "matrix_sha256": "a1e529c33281a79ad350f694e19c60adba51e6b34fdbc13ff08fd0af9e0d4bad",
            "orthogonality_max_abs": 1.4901161193847656e-06
          },
          {
            "layer": 21,
            "matrix_sha256": "e1b5097abaac8fda398fdb122f845deefc2dc410f846663134e4e387f9850e71",
            "orthogonality_max_abs": 1.5497207641601562e-06
          },
          {
            "layer": 22,
            "matrix_sha256": "d5a3311d94e9dacf7e12fc405e690044c78298b4409abab6bf32a4df42dea2e0",
            "orthogonality_max_abs": 1.8477439880371094e-06
          },
          {
            "layer": 24,
            "matrix_sha256": "da2e2b8cedf0d48896f8771fea6137f55ad99251f54ac324c48a81423bb9b37b",
            "orthogonality_max_abs": 1.3113021850585938e-06
          },
          {
            "layer": 25,
            "matrix_sha256": "537e17c3f13ce0b61ada253a7b9dc53e169091cfebdf76bd5795f804ff8c1eec",
            "orthogonality_max_abs": 1.430511474609375e-06
          },
          {
            "layer": 26,
            "matrix_sha256": "5558ff52fea94d858141636db3103fc16c19440298a41e2ace3cf2fa00b21063",
            "orthogonality_max_abs": 1.7881393432617188e-06
          },
          {
            "layer": 28,
            "matrix_sha256": "07923665ea0b349017cf2922dae9a098a650869f4c8ac609216170867e1fa179",
            "orthogonality_max_abs": 1.1920928955078125e-06
          },
          {
            "layer": 29,
            "matrix_sha256": "f7bc8fa2cbedbf0b8f605ed60feeed2b4160b89963ba5e6421deae711a2ff305",
            "orthogonality_max_abs": 1.4901161193847656e-06
          },
          {
            "layer": 30,
            "matrix_sha256": "50b455232f24a1b4c196672ed37aa3e2a680b99777b3b3955ea2de9b968db673",
            "orthogonality_max_abs": 1.6689300537109375e-06
          }
        ],
        "qk_rule": "Original normalize_qk then q/k.float @ H128 @ Delta; separate products",
        "stored_from_native": "Delta.T @ (H128 @ S_native); separate products",
        "theta_reconstruction": false
      },
      {
        "anchor_id": "A3",
        "condition": "C7",
        "direct_frozen_export_load": true,
        "export_file_sha256": "57e7a11714405806385064f07438e5e2ac6b5952dd421606332b894cef3ee6e8",
        "export_path": "/data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1/experiments/QWEN_GDN_PROPAGATION_SURROGATE_VALIDATION_V1/provenance/recovered_anchors/matrix/C7.pt",
        "matrix_dtype": "torch.float32",
        "matrix_shape": [
          128,
          128
        ],
        "native_basis": "S_native = H128 @ (Delta @ S_stored); separate products, never fused",
        "original_provenance": {
          "condition": "C7",
          "exposure": 152,
          "matrix": "/data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1/experiments/QWEN_GDN_PROPAGATION_SURROGATE_VALIDATION_V1/provenance/recovered_anchors/matrix/C7.pt",
          "matrix_sha256": "57e7a11714405806385064f07438e5e2ac6b5952dd421606332b894cef3ee6e8",
          "original_matrix": "/root/autodl-tmp/GDN-A800-cleanroom/experiments/QWEN_GDN_A800_CLEANROOM_C5R_C6R_C7_MATCHED_V1/training/C7/frozen_rotations/update_019_exposure_0152.pt",
          "original_theta": "/root/autodl-tmp/GDN-A800-cleanroom/experiments/QWEN_GDN_A800_CLEANROOM_C5R_C6R_C7_MATCHED_V1/training/C7/checkpoints/update_019_exposure_0152.pt",
          "selection_sha256": "22fc746f988392cbe937c05f94bfc70426a2e76fe420955887be288ea8bb862f",
          "theta": "/data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1/experiments/QWEN_GDN_PROPAGATION_SURROGATE_VALIDATION_V1/provenance/recovered_anchors/theta/update_019_exposure_0152.pt",
          "theta_sha256": "b08f66fa64f3453821a4d9dd9586c5263a747c461202c1dbc31b33193bde4958",
          "update": 19
        },
        "orthogonality_max_abs": 2.384185791015625e-06,
        "per_layer_matrices": [
          {
            "layer": 0,
            "matrix_sha256": "05f5e29d8443530a00b31a92aa5f3779efd6e7e84184edb004e1d1e82e9aac56",
            "orthogonality_max_abs": 1.6689300537109375e-06
          },
          {
            "layer": 1,
            "matrix_sha256": "4bf75ffb646ee56d02096b2bf53acc8012afb3604cc242929c1faef2c8338131",
            "orthogonality_max_abs": 1.7881393432617188e-06
          },
          {
            "layer": 2,
            "matrix_sha256": "e2c4c559abf4e62acafd53d6fe24d33e067bc89971db63c4d080ff39103426cc",
            "orthogonality_max_abs": 2.384185791015625e-06
          },
          {
            "layer": 4,
            "matrix_sha256": "19b0f896411281d4196e0519fcd0cfc3a125988e84d643928efda4d8d0bc084a",
            "orthogonality_max_abs": 1.7881393432617188e-06
          },
          {
            "layer": 5,
            "matrix_sha256": "742517a8bcc4d0fd645d90f0b726f733bcfd9da5e4376fc917a7b4184f6d29b7",
            "orthogonality_max_abs": 2.0265579223632812e-06
          },
          {
            "layer": 6,
            "matrix_sha256": "84f1515206cc10a40c06e3f2ba63a0f7cb3ff0081742b73bebc18ff9ae03a385",
            "orthogonality_max_abs": 1.9073486328125e-06
          },
          {
            "layer": 8,
            "matrix_sha256": "4a8672d459732b8c1aca0dfca85857d8e5fb154988b80eedf006a540bc5aa345",
            "orthogonality_max_abs": 1.6689300537109375e-06
          },
          {
            "layer": 9,
            "matrix_sha256": "88221f6368c477f20e143c3b8f291ecb2ecdc1eb3f73b41accd75332de4a6b6c",
            "orthogonality_max_abs": 2.086162567138672e-06
          },
          {
            "layer": 10,
            "matrix_sha256": "f53a8819cf6e5b476379702ba7c08ed1e19b4cd0af85c7aa054f8801129d8f41",
            "orthogonality_max_abs": 1.7285346984863281e-06
          },
          {
            "layer": 12,
            "matrix_sha256": "bca349016557733f698f52addf94398ad0a72fb8c34d9b45fb3ef8f1aae926e9",
            "orthogonality_max_abs": 1.9073486328125e-06
          },
          {
            "layer": 13,
            "matrix_sha256": "fade2d513f19131d792aada9a2be8d799f376ab6eafb37746ecac9bbf310e459",
            "orthogonality_max_abs": 2.384185791015625e-06
          },
          {
            "layer": 14,
            "matrix_sha256": "1cfbb89d4354e0408405ddb8c71525c3412164db0a340bc5259b172be8a4edca",
            "orthogonality_max_abs": 1.6093254089355469e-06
          },
          {
            "layer": 16,
            "matrix_sha256": "6cfef09b59316bd684bd66848950644421d56990534260644460e09e210f8259",
            "orthogonality_max_abs": 1.9669532775878906e-06
          },
          {
            "layer": 17,
            "matrix_sha256": "f5e5a9928a432424ef5d191bde7c41457fa8177cd9ca3b8baf8708bfd7c23c95",
            "orthogonality_max_abs": 1.6689300537109375e-06
          },
          {
            "layer": 18,
            "matrix_sha256": "374acb177ef146b597cf159ac07764712938e7f27be39b46e75af57a959e419f",
            "orthogonality_max_abs": 1.7285346984863281e-06
          },
          {
            "layer": 20,
            "matrix_sha256": "c792b300cd6c705c2efaf7a0bc9bfa19d8f326e29d77b71a728f6b201500837f",
            "orthogonality_max_abs": 1.9073486328125e-06
          },
          {
            "layer": 21,
            "matrix_sha256": "89c8c184f42a302e25af48b6d57800876eb3341c2b4a4d2a9e3e124384b9bd8c",
            "orthogonality_max_abs": 2.0265579223632812e-06
          },
          {
            "layer": 22,
            "matrix_sha256": "ff3a346aec504fe0c8727db6484ac4d46cc7493ee5d333a726a2d3e038825bd3",
            "orthogonality_max_abs": 1.7285346984863281e-06
          },
          {
            "layer": 24,
            "matrix_sha256": "264566252bda1f277fc1860be943f16875f2c56a648b60a2884327e260775ea0",
            "orthogonality_max_abs": 1.9073486328125e-06
          },
          {
            "layer": 25,
            "matrix_sha256": "6b2427172ea3aa6aa639ada23f1b17fd6ae29c3f9156c19096d580624a52bfeb",
            "orthogonality_max_abs": 1.6689300537109375e-06
          },
          {
            "layer": 26,
            "matrix_sha256": "0a1a2fc4bbedf60af01230cf493d5a469fad84c114e5374761627619059b9c13",
            "orthogonality_max_abs": 1.6689300537109375e-06
          },
          {
            "layer": 28,
            "matrix_sha256": "797774cfecbc0358d051b39875e2011f772df9cb21ed81a5007ac15767920eaa",
            "orthogonality_max_abs": 2.0265579223632812e-06
          },
          {
            "layer": 29,
            "matrix_sha256": "d446bc8da7ccc466479ecb2ec84412785fa46fcc830803c7d4346c7fd32a3cff",
            "orthogonality_max_abs": 1.430511474609375e-06
          },
          {
            "layer": 30,
            "matrix_sha256": "8f8985b9326b95f9dc680562089e2e101a989dcdaf9ed18cb3063dfb76fa13ce",
            "orthogonality_max_abs": 1.5497207641601562e-06
          }
        ],
        "qk_rule": "Original normalize_qk then q/k.float @ H128 @ Delta; separate products",
        "stored_from_native": "Delta.T @ (H128 @ S_native); separate products",
        "theta_reconstruction": false
      }
    ],
    "canonical_sources": [
      {
        "path": "/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/scripts/run_qwen_c5_c6_formal.py",
        "sha256": "77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69"
      },
      {
        "path": "/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py",
        "sha256": "90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a"
      },
      {
        "path": "/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2/experiments/QWEN_GDN_C5_C6_EVIDENCE_GUIDED_DIAGNOSIS_V1/scripts/sparse_probe.py",
        "sha256": "c214b3e7d2dacabb445378ef85cda5ed4ffe8fc2679d56d7c78217a8a95780b9"
      }
    ],
    "layer_ids": [
      0,
      1,
      2,
      4,
      5,
      6,
      8,
      9,
      10,
      12,
      13,
      14,
      16,
      17,
      18,
      20,
      21,
      22,
      24,
      25,
      26,
      28,
      29,
      30
    ],
    "model_loaded": false,
    "no_interpolation": true,
    "no_new_learned_rotation": true,
    "numerical_replay_invariance": "NOT_EVALUATED",
    "orthogonality_bound_inherited": 0.0001,
    "source_mapping": {
      "ideal_invariance": "only in exact arithmetic with orthogonal maps",
      "native_state": "separate Delta @ S_stored then H128 @ result",
      "numerical_invariance": "NOT_EVALUATED",
      "physical_source": {
        "commit": "a979a9f678f4ba0d6b9510a56760d36728929b8c",
        "file_sha256": "c214b3e7d2dacabb445378ef85cda5ed4ffe8fc2679d56d7c78217a8a95780b9",
        "function": "physical_state",
        "function_sha256": "aac23c68c4161374c7a12dbcbc8224ad6e1235d5c785dc0a7eee61593c4f9158",
        "path": "/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2/experiments/QWEN_GDN_C5_C6_EVIDENCE_GUIDED_DIAGNOSIS_V1/scripts/sparse_probe.py"
      },
      "qk_forward": "separate q/k.float @ H128 @ Delta",
      "status": "SOURCE_RECOVERED_NUMERICAL_PENDING",
      "stored_from_native": "Delta.T @ (H128 @ S_native)"
    }
  },
  "document_manifest_sha256": "8a95cd1e6082355315010be0d64a74e89b516d56b176ce92e3c917bb99f7d050",
  "anchor_manifest_sha256": "d772c8070e0f89d6bd9834cf32bfe297d519ce66157bf890074d3a48ca1d898b",
  "horizons": [
    2048,
    8192,
    32768
  ],
  "metric_rules": {
    "COUNTERFACTUAL_DECOMPOSITION_GATE": {
      "common_denominator": "FP native state squared Frobenius norm accumulated in FP64 +1e-12 per layer/token",
      "conditions": [
        "Native",
        "H",
        "C6R",
        "C7"
      ],
      "energy_accumulator": "FP64; signed alignment retained",
      "energy_relative_bound": "16*gamma_N_FP64*max(1,ACC,HISTORY,PATH_INJECTION,abs(ALIGNMENT)); N=32*128*128",
      "normalized_window": "inclusive1024..T for science; inclusive1..T for short engineering checks only",
      "steps": 256,
      "vector": "FP64 E=SQ-FP; H=F(SQ)-F(FP); I_PATH=SQ-F(SQ); I_REF=Q(FP)-F(FP); INTERACTION=I_PATH-I_REF",
      "vector_absolute_bound": "8*2^-53*max(1,max_abs(original FP32 states converted to FP64))"
    },
    "FP_REPLAY_PARITY_GATE": {
      "criterion": "Exact tensor SHA across all24 layers at every replay token, inherited parent FP exact check",
      "lengths": [
        32,
        256
      ],
      "no_posthoc_fallback": true,
      "numerical_policy_for_reporting_only": {
        "comparison": "max_abs_residual <= 8*gamma128*max(1,max_abs_of_compared_inputs_and_outputs)",
        "exact_checks": [
          "FP production helper vs model captured FP final states",
          "repeat deterministic operator",
          "rotation file and per-layer tensor SHA"
        ],
        "fp32_unit_roundoff": 5.960464477539063e-08,
        "frozen_before_full_parity_panel": true,
        "gamma128": 7.629452739355006e-06,
        "invariance": "exact tensor equality; FAIL allowed, then condition-specific production transition must be used",
        "rationale": "FP32 128-coordinate reductions plus separate H/Delta and inverse mapping reductions; conservative dtype-derived absolute forward-error bound. Difference cancellation is scaled to original operands, not tiny residual.",
        "reduction_width": 128,
        "roundoff_multiplier": 8
      }
    },
    "NO_QDQ_ROTATION_INVARIANCE_GATE": {
      "comparison_operands": "condition-specific unquantized native state and native FP state; max(1,abs operands)",
      "conditions": [
        "Native",
        "H",
        "C6R",
        "C7"
      ],
      "criterion": "max_abs_residual <= 8*gamma128*max(1,max_abs_of_compared_inputs_and_outputs)",
      "exact_arithmetic_invariance_not_claimed": true,
      "gamma128": 7.629452739355006e-06,
      "layers": 24,
      "multiplier": 8,
      "steps": 256
    },
    "REPLAY_DETERMINISM_GATE": {
      "all_conditions": true,
      "criterion": "All key state hashes and metric hashes byte-exact",
      "fixture2048": "First2048 tokens of already frozen Doc1 captured once only after short gates PASS; engineering only",
      "length2048_fresh_processes": 2,
      "length256_fresh_processes": 3,
      "strict_algorithms": true,
      "unsupported_operator": "FAIL_AND_STOP; do not patch"
    },
    "REPLAY_ISOLATION_GATE": {
      "blocked_cumsum_call_count": 0,
      "full_model_forward_count": 0,
      "instrument": "Fail-closed torch.cumsum/Tensor.cumsum and torch.nn.Module.__call__ guards; only original helper and formal transform/QDQ allowed"
    },
    "STREAMING_REDUCER_PARITY_GATE": {
      "criterion": "All six normalized terms and six raw energy means match within16*gamma_256_FP64*max(1,abs values)",
      "full_materialization": "All per-token per-layer FP64 energy/normalized observations; not full hidden/state trajectory",
      "no_changed_reduction_or_metric_definition": true,
      "short_length": 256
    },
    "any_new_failure": "FAIL_AND_STOP; preserve input/results; no retry, no threshold change, no failed drive deletion",
    "clean_reference_Q": "Separate condition-specific inverse-map of FP state then original helper, formal quantizer/writeback and physical map",
    "epsilon": 1e-12,
    "frozen_before_GPU_capture_results": true,
    "gate_order": [
      "CAPTURE_EMPIRICAL_REPRODUCIBILITY_GATE",
      "DRIVE_FACTORIZATION_GATE",
      "CAPTURE_SCHEMA_GATE",
      "FP_REPLAY_PARITY_GATE",
      "NO_QDQ_ROTATION_INVARIANCE_GATE",
      "COUNTERFACTUAL_DECOMPOSITION_GATE",
      "REPLAY_ISOLATION_GATE",
      "REPLAY_DETERMINISM_GATE",
      "STREAMING_REDUCER_PARITY_GATE"
    ],
    "horizons": [
      2048,
      8192,
      32768
    ],
    "no_inverse_remapping_of_persistent_path": true,
    "parent_J_INJ_denominator_not_relabelled_as_FD": true,
    "parent_metric_source_sha256": "6499c64b08a6637b54ec42be0b28dffad525063bc28fa904cbdaee2aeaf06ad3",
    "persistent_state": "Production stored basis across all tokens; physical state is a derived read-only separate H/Delta mapping",
    "scientific_first_token": 1024,
    "scientific_metric_names": [
      "J_ACC_FD",
      "J_HISTORY_FD",
      "J_INJ_PATH_FD",
      "J_INJ_REF_FD",
      "J_ALIGN_FD",
      "J_INTERACTION_FD"
    ],
    "source_transition": "Unchanged original torch_recurrent_gated_delta_rule reused, no copied recurrence formula",
    "weighting": "Equal layer weight; per-layer per-token Frobenius norm includes all32 heads and128x128 coordinates; equal token weight"
  },
  "metric_window": "inclusive1024..T",
  "epsilon": 1e-12,
  "weighting": "Equal layer/token weighting; whole32-head state Frobenius energies",
  "persistent_trajectory": true,
  "reset512": false,
  "quantizer": "Unmodified formal INT8-C128 quantize_c128; persistent writeback",
  "capture_policy_sha256": "59539fc9a1cccd709c5d9cf5c0ca77b2108248bcb0f91a34c454e394f89c4c3d",
  "canonical_capture": "FIRST_SUCCESSFUL_VALID_CAPTURE",
  "capture_strict_determinism": false,
  "replay_strict_determinism": true,
  "release_policy": "Unchanged exact new drive_shards release after capture+all4anchors+all3horizons+all seals verified; no failed drive deletion",
  "keep_forever": "documents/tokens/tokenizer/source/model/drive manifests/per-shard hashes/runtime/anchors/metrics/results/release records/sparse FP references",
  "storage_strategy": "SEQUENTIAL_PER_DOCUMENT_CAPTURE_PARALLEL_ANCHOR_REPLAY_SEAL_RELEASE",
  "maximum_concurrent_full_document_drives": 1,
  "maximum_concurrent_replay_workers": 4,
  "schedule": {
    "AIME_generation": false,
    "authorization": "Human: \u5728\u4fdd\u8bc1\u78c1\u76d8\u7a7a\u95f4\u7684\u524d\u63d0\u4e0b\uff0c\u6539\u4e3a\u56db\u5361\u5e76\u884c",
    "capture_gpu": 0,
    "condition_gpu": {
      "C6R": 2,
      "C7": 3,
      "H": 1,
      "Native": 0
    },
    "documents": [
      1,
      2,
      3,
      4
    ],
    "gate_order": [
      "existing_G11_2048_run01_natural_completion",
      "same_frozen_G11_run02",
      "G11_compare",
      "G12",
      "four_GPU_256_fixture_PARITY",
      "full_premeasurement_commit",
      "per_document_capture_parallel_replay_seal_release"
    ],
    "max_concurrent_full_document_drives": 1,
    "max_concurrent_replays": 4,
    "new_failure": "FAIL_AND_STOP; no retry, no threshold relaxation; no incomplete drive deletion",
    "next_document_requires_all_four_results_sealed_and_current_drive_released": true,
    "physical_devices_per_process": "CUDA_VISIBLE_DEVICES restricts each worker to exactly one physical GPU; unchanged local cuda:0",
    "runtime_gpu_inventory": "0, GPU-a78176ed-f97f-c54f-6201-5a01b570f422, NVIDIA GeForce RTX 3090, 24576 MiB\n1, GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3, NVIDIA GeForce RTX 3090, 24576 MiB\n2, GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31, NVIDIA GeForce RTX 3090, 24576 MiB\n3, GPU-30b07aba-a728-7649-4899-e09d33f8bc75, NVIDIA GeForce RTX 3090, 24576 MiB\n",
    "scheduling_only": true,
    "scientific_inputs_and_computations_changed": false,
    "shared_drive_read_only": true,
    "topology_parity_conditions": [
      "Native",
      "H",
      "C6R",
      "C7"
    ],
    "topology_parity_criterion": "Exact identity.outputs (every-token FP/Q hashes, metric-byte hashes, endpoint/horizon values and closure maxima) and runtime call counts vs existing GPU0 reference",
    "topology_parity_fixture_tokens": 256,
    "training": false
  },
  "storage_formula": "BASE+max(5GiB,20%BASE)<=AVAILABLE; margin exactly once",
  "lifecycle_peak_required_bytes": 27895529472,
  "available_at_final_freeze": 28679835648,
  "primary": "C6R-minus-H",
  "secondary": [
    "C7-minus-C6R",
    "Native-minus-H"
  ],
  "interpretation": "Descriptive mechanism discovery, n_docs4; no universal surrogate rank or end-to-end reasoning causality",
  "statistics": [
    "per_doc",
    "median_paired_difference",
    "sign_count",
    "range",
    "horizon_consistency"
  ],
  "confirmation_rule": "Original frozen hash ordering after four Discovery docs; NOT_EXECUTED",
  "prohibited": {
    "training": true,
    "optimizer_step": true,
    "AIME": true,
    "free_generation": true,
    "confirmation": true,
    "intervention": true,
    "rotation_search": true
  },
  "all_engineering_gates": "PASS",
  "cross_gpu_parity_gate": "PASS",
  "any_new_failure": "FAIL_AND_STOP; no retry or relaxation; retain incomplete drive",
  "historical_integrity": {
    "OLD_EVIDENCE_INTEGRITY_GATE": "PASS",
    "SCIENTIFIC_MEASUREMENT_COUNT": 0,
    "SCIENTIFIC_MEASUREMENT_COUNT_BEFORE_NEW_EXPERIMENT": 0,
    "seals": {
      "parent": {
        "root": "/data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1/experiments/QWEN_GDN_PROPAGATION_SURROGATE_VALIDATION_V1",
        "manifest": "/data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1/experiments/QWEN_GDN_PROPAGATION_SURROGATE_VALIDATION_V1/FINAL_SHA256SUMS",
        "manifest_sha256": "332ecdaacef77263dfaca51a5cca22c8e6dc6e2bd62ff22aec0b3a7b4c176dc5",
        "sealed_artifact_count": 47,
        "all_exact": true,
        "rows": [
          {
            "relative_path": "FINAL_REPORT.md",
            "expected_sha256": "146f30bc2c5f3f65d153e2934373f3d90bfde939278aec3a0f6d4951d3877281",
            "actual_sha256": "146f30bc2c5f3f65d153e2934373f3d90bfde939278aec3a0f6d4951d3877281",
            "PASS": true
          },
          {
            "relative_path": "analysis/confirmation_summary.json",
            "expected_sha256": "8cc4539b6d2fdbe138eed0a77a9b484aef102dbccf0b72c741db37fd232b4951",
            "actual_sha256": "8cc4539b6d2fdbe138eed0a77a9b484aef102dbccf0b72c741db37fd232b4951",
            "PASS": true
          },
          {
            "relative_path": "analysis/discovery_summary.json",
            "expected_sha256": "8cc4539b6d2fdbe138eed0a77a9b484aef102dbccf0b72c741db37fd232b4951",
            "actual_sha256": "8cc4539b6d2fdbe138eed0a77a9b484aef102dbccf0b72c741db37fd232b4951",
            "PASS": true
          },
          {
            "relative_path": "analysis/final_verdict.json",
            "expected_sha256": "8bf45663cc3067e6028a9822c4273c4549517aa4b6cf1373e0316cb61f3f06f2",
            "actual_sha256": "8bf45663cc3067e6028a9822c4273c4549517aa4b6cf1373e0316cb61f3f06f2",
            "PASS": true
          },
          {
            "relative_path": "analysis/phase_a.error.json",
            "expected_sha256": "f84dbdb4ac09011a1133e146d1df698c3fdb661aee79d359dcd1e9686c3a1ad4",
            "actual_sha256": "f84dbdb4ac09011a1133e146d1df698c3fdb661aee79d359dcd1e9686c3a1ad4",
            "PASS": true
          },
          {
            "relative_path": "analysis/phase_a_repaired_001.error.json",
            "expected_sha256": "1e2deb5a0a2c53f0afc1c9c61fe73d6c0ff2be5f9a8d65eea515e4a94868201b",
            "actual_sha256": "1e2deb5a0a2c53f0afc1c9c61fe73d6c0ff2be5f9a8d65eea515e4a94868201b",
            "PASS": true
          },
          {
            "relative_path": "logs/phase_a.exitcode",
            "expected_sha256": "4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865",
            "actual_sha256": "4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865",
            "PASS": true
          },
          {
            "relative_path": "logs/phase_a.log",
            "expected_sha256": "abe9155d5b86ca218c6368c6b806cd0bd7c1ae0dd523ee7faa37602c6fc53f1b",
            "actual_sha256": "abe9155d5b86ca218c6368c6b806cd0bd7c1ae0dd523ee7faa37602c6fc53f1b",
            "PASS": true
          },
          {
            "relative_path": "logs/phase_a_repaired_001.exitcode",
            "expected_sha256": "4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865",
            "actual_sha256": "4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865",
            "PASS": true
          },
          {
            "relative_path": "logs/phase_a_repaired_001.log",
            "expected_sha256": "ab060230caf4461fb6f04af06f4e71026b80273028219c3fb9fded93a6b9855f",
            "actual_sha256": "ab060230caf4461fb6f04af06f4e71026b80273028219c3fb9fded93a6b9855f",
            "PASS": true
          },
          {
            "relative_path": "protocol/frozen_fresh_documents.json",
            "expected_sha256": "eac9cc1afa95d2d49d6600d3c6b8a02b24764f46f8d10f8239035640595f8534",
            "actual_sha256": "eac9cc1afa95d2d49d6600d3c6b8a02b24764f46f8d10f8239035640595f8534",
            "PASS": true
          },
          {
            "relative_path": "protocol/frozen_rotation_bank.json",
            "expected_sha256": "ee54e8f2b70d43a3af8c8286a6fc3ca0b604b034d12b0b599bb117d7bf7fd237",
            "actual_sha256": "ee54e8f2b70d43a3af8c8286a6fc3ca0b604b034d12b0b599bb117d7bf7fd237",
            "PASS": true
          },
          {
            "relative_path": "protocol/metric_definitions.json",
            "expected_sha256": "6499c64b08a6637b54ec42be0b28dffad525063bc28fa904cbdaee2aeaf06ad3",
            "actual_sha256": "6499c64b08a6637b54ec42be0b28dffad525063bc28fa904cbdaee2aeaf06ad3",
            "PASS": true
          },
          {
            "relative_path": "protocol/phase_a_parameterization_policy.json",
            "expected_sha256": "62c2793997a939423c999f14a5dad702c16c4bcd8b5dce0dcb5ded05bfe22440",
            "actual_sha256": "62c2793997a939423c999f14a5dad702c16c4bcd8b5dce0dcb5ded05bfe22440",
            "PASS": true
          },
          {
            "relative_path": "protocol/preregistration.json",
            "expected_sha256": "9c83cfe576dff152f151c1d027a603ba3a9553e0018b54b16c024037753edf15",
            "actual_sha256": "9c83cfe576dff152f151c1d027a603ba3a9553e0018b54b16c024037753edf15",
            "PASS": true
          },
          {
            "relative_path": "protocol/preregistration_repaired_001.json",
            "expected_sha256": "de68a9113be1e2613ec15c0892ef9aa557e3187f9d3e2dc3b6635c5db67590d9",
            "actual_sha256": "de68a9113be1e2613ec15c0892ef9aa557e3187f9d3e2dc3b6635c5db67590d9",
            "PASS": true
          },
          {
            "relative_path": "provenance/GDN_ERROR_RECURRENCE_PARITY.json",
            "expected_sha256": "7120ad7b82d3a83db5d1da91ddbcfe89789f9c7f535238f6f27c22c7417dc415",
            "actual_sha256": "7120ad7b82d3a83db5d1da91ddbcfe89789f9c7f535238f6f27c22c7417dc415",
            "PASS": true
          },
          {
            "relative_path": "provenance/GDN_RECURRENCE_SOURCE_AUDIT.json",
            "expected_sha256": "183012790a8eb5f9c147a81f7d2280f6533c82b057008a08c6d4b10f235efb07",
            "actual_sha256": "183012790a8eb5f9c147a81f7d2280f6533c82b057008a08c6d4b10f235efb07",
            "PASS": true
          },
          {
            "relative_path": "provenance/GDN_ZERO_WRITE_PARITY.json",
            "expected_sha256": "e5ed7dec9c826a840f757401f62303484aa76b01ca65f12aa25f19b31e957147",
            "actual_sha256": "e5ed7dec9c826a840f757401f62303484aa76b01ca65f12aa25f19b31e957147",
            "PASS": true
          },
          {
            "relative_path": "provenance/PHASE_A_PARITY_FREEZE.json",
            "expected_sha256": "7dcfd5ad43fbeaa905ccdc8544e7675430f9d2804abe6f0a96de9571e60eebb5",
            "actual_sha256": "7dcfd5ad43fbeaa905ccdc8544e7675430f9d2804abe6f0a96de9571e60eebb5",
            "PASS": true
          },
          {
            "relative_path": "provenance/PHASE_A_PARITY_FREEZE_REPAIRED_001.json",
            "expected_sha256": "f6573e436c26ba070877e759b052d462d815f80c2f0c006835347b800047c2c4",
            "actual_sha256": "f6573e436c26ba070877e759b052d462d815f80c2f0c006835347b800047c2c4",
            "PASS": true
          },
          {
            "relative_path": "provenance/RECOVERY_SHA256SUMS",
            "expected_sha256": "b0514d59a3952f347448837df527a9b5b941aa596f8d2b54c859c67724d54bc9",
            "actual_sha256": "b0514d59a3952f347448837df527a9b5b941aa596f8d2b54c859c67724d54bc9",
            "PASS": true
          },
          {
            "relative_path": "provenance/anchor_recovery.json",
            "expected_sha256": "395303bd0f69942034703eb06d73fafde434c85d8ec4524ea9037e7de35875d2",
            "actual_sha256": "395303bd0f69942034703eb06d73fafde434c85d8ec4524ea9037e7de35875d2",
            "PASS": true
          },
          {
            "relative_path": "provenance/final_parent_revalidation.json",
            "expected_sha256": "4ee3d49beb9d8db596bc4de3d476a3cba5a941c8b9af360970021c09f87bd9d4",
            "actual_sha256": "4ee3d49beb9d8db596bc4de3d476a3cba5a941c8b9af360970021c09f87bd9d4",
            "PASS": true
          },
          {
            "relative_path": "provenance/hardware_identity.json",
            "expected_sha256": "b097503d2fc8a0c0399558253fcb2f95c7a96e3499fc4d385e57effbe20a4dd4",
            "actual_sha256": "b097503d2fc8a0c0399558253fcb2f95c7a96e3499fc4d385e57effbe20a4dd4",
            "PASS": true
          },
          {
            "relative_path": "provenance/native_basis_mapping.json",
            "expected_sha256": "ae7b5eaf9c6c8d0932b3dcfdd9fc62ca4cd4868ae53ea01e0ee444c4bd98bba5",
            "actual_sha256": "ae7b5eaf9c6c8d0932b3dcfdd9fc62ca4cd4868ae53ea01e0ee444c4bd98bba5",
            "PASS": true
          },
          {
            "relative_path": "provenance/native_basis_source_mapping.json",
            "expected_sha256": "d4c249d147820b8cc4b4cab3b95da60a89870de1e6b668db30edffa92ecb216c",
            "actual_sha256": "d4c249d147820b8cc4b4cab3b95da60a89870de1e6b668db30edffa92ecb216c",
            "PASS": true
          },
          {
            "relative_path": "provenance/original_a800_export_audit_source.py",
            "expected_sha256": "231a7565880311cd5cb452fc67eaa311bdb10c795332806c5decd09654382e48",
            "actual_sha256": "231a7565880311cd5cb452fc67eaa311bdb10c795332806c5decd09654382e48",
            "PASS": true
          },
          {
            "relative_path": "provenance/parent_artifact_identity.json",
            "expected_sha256": "13b809febd529949502f37c8367ed7b93ce9d5c2b5f38431b44a04f9f8aebb1d",
            "actual_sha256": "13b809febd529949502f37c8367ed7b93ce9d5c2b5f38431b44a04f9f8aebb1d",
            "PASS": true
          },
          {
            "relative_path": "provenance/phase_a_metadata_repair_001.json",
            "expected_sha256": "3794f7c73ac8d36b46f9c8372d7bc02692483561e901be29e317ddbd88107918",
            "actual_sha256": "3794f7c73ac8d36b46f9c8372d7bc02692483561e901be29e317ddbd88107918",
            "PASS": true
          },
          {
            "relative_path": "provenance/quantizer_identity.json",
            "expected_sha256": "dbb21e413d752f9fb794839b55321a97562e72c7059e6a0cdb11c861019e704e",
            "actual_sha256": "dbb21e413d752f9fb794839b55321a97562e72c7059e6a0cdb11c861019e704e",
            "PASS": true
          },
          {
            "relative_path": "provenance/recovered_anchors/matrix/C5R.pt",
            "expected_sha256": "2916169ea163de9c7f179d73936f38d6effd8c4407488298d5e29a3d8de01076",
            "actual_sha256": "2916169ea163de9c7f179d73936f38d6effd8c4407488298d5e29a3d8de01076",
            "PASS": true
          },
          {
            "relative_path": "provenance/recovered_anchors/matrix/C6R.pt",
            "expected_sha256": "a886588b3613f66e6635e5874dbfaeaee26b850e05b4a1524da0bef952d22da6",
            "actual_sha256": "a886588b3613f66e6635e5874dbfaeaee26b850e05b4a1524da0bef952d22da6",
            "PASS": true
          },
          {
            "relative_path": "provenance/recovered_anchors/matrix/C7.pt",
            "expected_sha256": "57e7a11714405806385064f07438e5e2ac6b5952dd421606332b894cef3ee6e8",
            "actual_sha256": "57e7a11714405806385064f07438e5e2ac6b5952dd421606332b894cef3ee6e8",
            "PASS": true
          },
          {
            "relative_path": "provenance/recovered_anchors/theta/update_019_exposure_0152.pt",
            "expected_sha256": "b08f66fa64f3453821a4d9dd9586c5263a747c461202c1dbc31b33193bde4958",
            "actual_sha256": "b08f66fa64f3453821a4d9dd9586c5263a747c461202c1dbc31b33193bde4958",
            "PASS": true
          },
          {
            "relative_path": "provenance/recovered_anchors/theta/update_094_exposure_0752.pt",
            "expected_sha256": "620a35101b95cadd16f50cd1b869aeef54da9db0b3c76076fe82a374a6ee17cf",
            "actual_sha256": "620a35101b95cadd16f50cd1b869aeef54da9db0b3c76076fe82a374a6ee17cf",
            "PASS": true
          },
          {
            "relative_path": "provenance/recovered_anchors/theta/update_113_exposure_0904.pt",
            "expected_sha256": "a11cf60f6ff11c2bb97f31b09129b8bfaadbf553f2005abc9276343ddca10fb5",
            "actual_sha256": "a11cf60f6ff11c2bb97f31b09129b8bfaadbf553f2005abc9276343ddca10fb5",
            "PASS": true
          },
          {
            "relative_path": "provenance/rotation_parameterization_identity.json",
            "expected_sha256": "197c521e210cd573538a38301927444f4a248052df46524c09370e3741d18b1d",
            "actual_sha256": "197c521e210cd573538a38301927444f4a248052df46524c09370e3741d18b1d",
            "PASS": true
          },
          {
            "relative_path": "provenance/runtime_identity.json",
            "expected_sha256": "3fda1dc7a27432ae1100adfed1a3a519050d055c23f0d13ae9d45a93499cded9",
            "actual_sha256": "3fda1dc7a27432ae1100adfed1a3a519050d055c23f0d13ae9d45a93499cded9",
            "PASS": true
          },
          {
            "relative_path": "provenance/transition_rotation_invariance.json",
            "expected_sha256": "b3b628edd37f7dd074a453a2d9638e2aa2c294da3224b03165fbc780aefebb6f",
            "actual_sha256": "b3b628edd37f7dd074a453a2d9638e2aa2c294da3224b03165fbc780aefebb6f",
            "PASS": true
          },
          {
            "relative_path": "scripts/propagation_phase_a.py",
            "expected_sha256": "b21b5445215eb86f9dd999ff574bba9dfc9f61360819cccbc2dee5c7c4f44aa3",
            "actual_sha256": "b21b5445215eb86f9dd999ff574bba9dfc9f61360819cccbc2dee5c7c4f44aa3",
            "PASS": true
          },
          {
            "relative_path": "scripts/propagation_phase_a_repaired_001.py",
            "expected_sha256": "260a8c4fab0a9127a4ad74cbc04ad4604b33af395c90993a54b441b9e512790e",
            "actual_sha256": "260a8c4fab0a9127a4ad74cbc04ad4604b33af395c90993a54b441b9e512790e",
            "PASS": true
          },
          {
            "relative_path": "scripts/propagation_recovery_closure.py",
            "expected_sha256": "1a4e57f43a37643f4f2fbdd068f9dafb614e43918c9f8eeb233d0fa6b70ace89",
            "actual_sha256": "1a4e57f43a37643f4f2fbdd068f9dafb614e43918c9f8eeb233d0fa6b70ace89",
            "PASS": true
          },
          {
            "relative_path": "scripts/propagation_stop_finalize.py",
            "expected_sha256": "e0b67d458aed5241271b702a1e4d627edde192b3399693417021207ab771a399",
            "actual_sha256": "e0b67d458aed5241271b702a1e4d627edde192b3399693417021207ab771a399",
            "PASS": true
          },
          {
            "relative_path": "scripts/run_propagation_phase_a.sh",
            "expected_sha256": "d0a4655aad007bd58ce2fb3bef84056781a8b577af81504b7dbff4a11db18b8f",
            "actual_sha256": "d0a4655aad007bd58ce2fb3bef84056781a8b577af81504b7dbff4a11db18b8f",
            "PASS": true
          },
          {
            "relative_path": "scripts/run_propagation_phase_a_repaired_001.sh",
            "expected_sha256": "6fb438035608934b5bd928b0fa6663598b99119c401a21601161eb3b0276769a",
            "actual_sha256": "6fb438035608934b5bd928b0fa6663598b99119c401a21601161eb3b0276769a",
            "PASS": true
          },
          {
            "relative_path": "traces/trace_status.json",
            "expected_sha256": "db536bc847ea31ee880c6f5a03ffeb71f33f13b8e2e28fb0f9e17290b1e7f695",
            "actual_sha256": "db536bc847ea31ee880c6f5a03ffeb71f33f13b8e2e28fb0f9e17290b1e7f695",
            "PASS": true
          }
        ]
      },
      "cumsum": {
        "root": "/data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1/experiments/QWEN_GDN_DETERMINISTIC_CUMSUM_CLOSURE_V1",
        "manifest": "/data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1/experiments/QWEN_GDN_DETERMINISTIC_CUMSUM_CLOSURE_V1/FINAL_SHA256SUMS",
        "manifest_sha256": "6586abcdd591de6242ddc7af6abc20bbe8fb0cd8afd2e038f890439dce28ae9b",
        "sealed_artifact_count": 80,
        "all_exact": true,
        "rows": [
          {
            "relative_path": "FINAL_REPORT.md",
            "expected_sha256": "3008c595635ee0e75e023111e2a5d093f0a23c3fd06886e7a0c64d7be886fd72",
            "actual_sha256": "3008c595635ee0e75e023111e2a5d093f0a23c3fd06886e7a0c64d7be886fd72",
            "PASS": true
          },
          {
            "relative_path": "chunk/reference.json",
            "expected_sha256": "662ed673cc51b73b4fbedde28fbff707f00c07cc2bb8f558bea1a167ea3bc132",
            "actual_sha256": "662ed673cc51b73b4fbedde28fbff707f00c07cc2bb8f558bea1a167ea3bc132",
            "PASS": true
          },
          {
            "relative_path": "chunk/reference.pt",
            "expected_sha256": "c42f0e71039a2da1fcd2d795fd5388e604c27efe7b522fb838d813de35bcff9b",
            "actual_sha256": "c42f0e71039a2da1fcd2d795fd5388e604c27efe7b522fb838d813de35bcff9b",
            "PASS": true
          },
          {
            "relative_path": "chunk/strict.json",
            "expected_sha256": "e2782de8472022930c36585592c8d1b38e34af5b6f9ce3bf94f29f37192155ee",
            "actual_sha256": "e2782de8472022930c36585592c8d1b38e34af5b6f9ce3bf94f29f37192155ee",
            "PASS": true
          },
          {
            "relative_path": "chunk/strict.pt",
            "expected_sha256": "dfeb1957383b031e445cdb1becda088f75fae5f425b74801011e6eb3c8fd032a",
            "actual_sha256": "dfeb1957383b031e445cdb1becda088f75fae5f425b74801011e6eb3c8fd032a",
            "PASS": true
          },
          {
            "relative_path": "fixtures/chunk128.pt",
            "expected_sha256": "06d32f2735d698c870ba19d8af2e1a5122523a631850756afe3484fda039cae1",
            "actual_sha256": "06d32f2735d698c870ba19d8af2e1a5122523a631850756afe3484fda039cae1",
            "PASS": true
          },
          {
            "relative_path": "fixtures/chunk64.pt",
            "expected_sha256": "c22590be4eebb3dd43e468bc3312c62e907f861caebe6a46bbb73dfbd3be393a",
            "actual_sha256": "c22590be4eebb3dd43e468bc3312c62e907f861caebe6a46bbb73dfbd3be393a",
            "PASS": true
          },
          {
            "relative_path": "fixtures/dev256.pt",
            "expected_sha256": "1d9a7631fadbe9a187eb3c075ac67dc0753001f5629cdc0f724aaf5333b0711f",
            "actual_sha256": "1d9a7631fadbe9a187eb3c075ac67dc0753001f5629cdc0f724aaf5333b0711f",
            "PASS": true
          },
          {
            "relative_path": "fixtures/historical32.pt",
            "expected_sha256": "acd2f5bfd73467f7af8941783cb15547329906d85daf4571bb5834bfd0b0ffdd",
            "actual_sha256": "acd2f5bfd73467f7af8941783cb15547329906d85daf4571bb5834bfd0b0ffdd",
            "PASS": true
          },
          {
            "relative_path": "fixtures/historical_prefill1.pt",
            "expected_sha256": "a813da7ba84e7a1c0ff148fc8b9133c9ffd2901b97f12230dc3ddc2ef7fa4382",
            "actual_sha256": "a813da7ba84e7a1c0ff148fc8b9133c9ffd2901b97f12230dc3ddc2ef7fa4382",
            "PASS": true
          },
          {
            "relative_path": "fixtures/long_prefill.pt",
            "expected_sha256": "792462c5025b3c5553d5815ae9e9af5f5382ec890c0fae6968a05c934a0da53a",
            "actual_sha256": "792462c5025b3c5553d5815ae9e9af5f5382ec890c0fae6968a05c934a0da53a",
            "PASS": true
          },
          {
            "relative_path": "full_model/reference_1.json",
            "expected_sha256": "6cc4a2bd8ccfc842b744f8938f3738d571ee0091daab2b9ca522dcd5b744cdfb",
            "actual_sha256": "6cc4a2bd8ccfc842b744f8938f3738d571ee0091daab2b9ca522dcd5b744cdfb",
            "PASS": true
          },
          {
            "relative_path": "full_model/reference_1_dev256.pt",
            "expected_sha256": "5c8f3ac23263eefad0aab666eea3d9dd3517f632ee54340ff9456e0875e437b1",
            "actual_sha256": "5c8f3ac23263eefad0aab666eea3d9dd3517f632ee54340ff9456e0875e437b1",
            "PASS": true
          },
          {
            "relative_path": "full_model/reference_1_dev256_trace.json",
            "expected_sha256": "96f0823c56572e0a87c19c70d5f9291fd4b7001f78af79938ba8430a5f1fcf2b",
            "actual_sha256": "96f0823c56572e0a87c19c70d5f9291fd4b7001f78af79938ba8430a5f1fcf2b",
            "PASS": true
          },
          {
            "relative_path": "full_model/reference_1_historical32.pt",
            "expected_sha256": "192902616473761416bb0e9f1514a4958bc68e5538958704a5a38cefb70a8f61",
            "actual_sha256": "192902616473761416bb0e9f1514a4958bc68e5538958704a5a38cefb70a8f61",
            "PASS": true
          },
          {
            "relative_path": "full_model/reference_1_historical32_trace.json",
            "expected_sha256": "71eb3509a943fd7d737b375ecb8e36ed6a55294e5bebcc0a839ca8f5ad37743c",
            "actual_sha256": "71eb3509a943fd7d737b375ecb8e36ed6a55294e5bebcc0a839ca8f5ad37743c",
            "PASS": true
          },
          {
            "relative_path": "full_model/strict_1.json",
            "expected_sha256": "96850367d5fe4b453435069814008a812e6a304874d33451ce39ac7c0b67f86c",
            "actual_sha256": "96850367d5fe4b453435069814008a812e6a304874d33451ce39ac7c0b67f86c",
            "PASS": true
          },
          {
            "relative_path": "full_model/strict_1_dev256.pt",
            "expected_sha256": "06ebb8967347731d6b022822e5ceeee71108d16854c59438c762ec41f8aec8d7",
            "actual_sha256": "06ebb8967347731d6b022822e5ceeee71108d16854c59438c762ec41f8aec8d7",
            "PASS": true
          },
          {
            "relative_path": "full_model/strict_1_dev256_trace.json",
            "expected_sha256": "2d4046b371de6e718e78606bdc6a17be0797bb59a5f4e536e79beaf2242ccd74",
            "actual_sha256": "2d4046b371de6e718e78606bdc6a17be0797bb59a5f4e536e79beaf2242ccd74",
            "PASS": true
          },
          {
            "relative_path": "full_model/strict_1_historical32.pt",
            "expected_sha256": "192902616473761416bb0e9f1514a4958bc68e5538958704a5a38cefb70a8f61",
            "actual_sha256": "192902616473761416bb0e9f1514a4958bc68e5538958704a5a38cefb70a8f61",
            "PASS": true
          },
          {
            "relative_path": "full_model/strict_1_historical32_trace.json",
            "expected_sha256": "71eb3509a943fd7d737b375ecb8e36ed6a55294e5bebcc0a839ca8f5ad37743c",
            "actual_sha256": "71eb3509a943fd7d737b375ecb8e36ed6a55294e5bebcc0a839ca8f5ad37743c",
            "PASS": true
          },
          {
            "relative_path": "logs/capture.exitcode",
            "expected_sha256": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
            "actual_sha256": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
            "PASS": true
          },
          {
            "relative_path": "logs/capture.log",
            "expected_sha256": "2f91097dc01d28103bc7b9957515e31b29a38c7592d6741df04698b836ab4813",
            "actual_sha256": "2f91097dc01d28103bc7b9957515e31b29a38c7592d6741df04698b836ab4813",
            "PASS": true
          },
          {
            "relative_path": "logs/chunk.log",
            "expected_sha256": "c6323081aa157bd0eb062ae29c72fb6ffbb83aed3dda5f9201a610200d0abea8",
            "actual_sha256": "c6323081aa157bd0eb062ae29c72fb6ffbb83aed3dda5f9201a610200d0abea8",
            "PASS": true
          },
          {
            "relative_path": "logs/full_qualification.exitcode",
            "expected_sha256": "4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865",
            "actual_sha256": "4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865",
            "PASS": true
          },
          {
            "relative_path": "logs/full_qualification.log",
            "expected_sha256": "fd18506e8b0773a2cea86b0d3ae467e40dffcf64d80bd9dba0dbb803bd5f830b",
            "actual_sha256": "fd18506e8b0773a2cea86b0d3ae467e40dffcf64d80bd9dba0dbb803bd5f830b",
            "PASS": true
          },
          {
            "relative_path": "logs/operator.log",
            "expected_sha256": "6e2e763cf8a9c7ac2662385d8f64e4ecbde4bf309f5708c5d6d90c0304c34f36",
            "actual_sha256": "6e2e763cf8a9c7ac2662385d8f64e4ecbde4bf309f5708c5d6d90c0304c34f36",
            "PASS": true
          },
          {
            "relative_path": "operator/reference_1.json",
            "expected_sha256": "f066e4a3c6d04466f7ae11ea6910c7af02a9fe004c3af708aa4bb876b57c8cde",
            "actual_sha256": "f066e4a3c6d04466f7ae11ea6910c7af02a9fe004c3af708aa4bb876b57c8cde",
            "PASS": true
          },
          {
            "relative_path": "operator/reference_1.pt",
            "expected_sha256": "3c8ed3ecedf3659cb36c374c453e6628751f5d0afe135c20f77d04e6d0cf43da",
            "actual_sha256": "3c8ed3ecedf3659cb36c374c453e6628751f5d0afe135c20f77d04e6d0cf43da",
            "PASS": true
          },
          {
            "relative_path": "operator/strict_1.json",
            "expected_sha256": "34fb7be2cb2775278317593e6b7d2e4f1f7c5684780809baa0e3dfe5fdee819b",
            "actual_sha256": "34fb7be2cb2775278317593e6b7d2e4f1f7c5684780809baa0e3dfe5fdee819b",
            "PASS": true
          },
          {
            "relative_path": "operator/strict_1.pt",
            "expected_sha256": "7d80249ed52e6418ca392ffc79808e555394f4d207d5a08b387df80c9fe9ddac",
            "actual_sha256": "7d80249ed52e6418ca392ffc79808e555394f4d207d5a08b387df80c9fe9ddac",
            "PASS": true
          },
          {
            "relative_path": "operator/strict_2.json",
            "expected_sha256": "f33234f81b29d83c518edb92599d05a4c5756720142972f946c84689477d94fc",
            "actual_sha256": "f33234f81b29d83c518edb92599d05a4c5756720142972f946c84689477d94fc",
            "PASS": true
          },
          {
            "relative_path": "operator/strict_2.pt",
            "expected_sha256": "7d80249ed52e6418ca392ffc79808e555394f4d207d5a08b387df80c9fe9ddac",
            "actual_sha256": "7d80249ed52e6418ca392ffc79808e555394f4d207d5a08b387df80c9fe9ddac",
            "PASS": true
          },
          {
            "relative_path": "operator/strict_3.json",
            "expected_sha256": "dda7400c4f2d6ffd220a765e05ef599bd80bc0e416c01336b04b2f0a6443d367",
            "actual_sha256": "dda7400c4f2d6ffd220a765e05ef599bd80bc0e416c01336b04b2f0a6443d367",
            "PASS": true
          },
          {
            "relative_path": "operator/strict_3.pt",
            "expected_sha256": "7d80249ed52e6418ca392ffc79808e555394f4d207d5a08b387df80c9fe9ddac",
            "actual_sha256": "7d80249ed52e6418ca392ffc79808e555394f4d207d5a08b387df80c9fe9ddac",
            "PASS": true
          },
          {
            "relative_path": "operator/strict_4.json",
            "expected_sha256": "c37dc89cda16288b099c0ee591831f877973da206684337e92a45b08a731bd45",
            "actual_sha256": "c37dc89cda16288b099c0ee591831f877973da206684337e92a45b08a731bd45",
            "PASS": true
          },
          {
            "relative_path": "operator/strict_4.pt",
            "expected_sha256": "7d80249ed52e6418ca392ffc79808e555394f4d207d5a08b387df80c9fe9ddac",
            "actual_sha256": "7d80249ed52e6418ca392ffc79808e555394f4d207d5a08b387df80c9fe9ddac",
            "PASS": true
          },
          {
            "relative_path": "operator/strict_5.json",
            "expected_sha256": "7429a4e6d45f795e5be20c45411012bffc1aa35937af7b76788fba6bd2c0ab63",
            "actual_sha256": "7429a4e6d45f795e5be20c45411012bffc1aa35937af7b76788fba6bd2c0ab63",
            "PASS": true
          },
          {
            "relative_path": "operator/strict_5.pt",
            "expected_sha256": "7d80249ed52e6418ca392ffc79808e555394f4d207d5a08b387df80c9fe9ddac",
            "actual_sha256": "7d80249ed52e6418ca392ffc79808e555394f4d207d5a08b387df80c9fe9ddac",
            "PASS": true
          },
          {
            "relative_path": "protocol/NUMERICAL_PARITY_POLICY.json",
            "expected_sha256": "5a871c3d1ef97d0f70bf9164afd3d81aa5cecfa48763eff8c5ce8ec7addaf073",
            "actual_sha256": "5a871c3d1ef97d0f70bf9164afd3d81aa5cecfa48763eff8c5ce8ec7addaf073",
            "PASS": true
          },
          {
            "relative_path": "protocol/closure_policy.json",
            "expected_sha256": "d0eb4cae5820c58fa7038902117c8dd49c3e6e0fa90b5bb8578b849926f4cfe0",
            "actual_sha256": "d0eb4cae5820c58fa7038902117c8dd49c3e6e0fa90b5bb8578b849926f4cfe0",
            "PASS": true
          },
          {
            "relative_path": "protocol/frozen_probe_tokens.json",
            "expected_sha256": "415b7fcadf3505b94e6254a9da74ac29be762d4fed68bd665e4adf8db277b651",
            "actual_sha256": "415b7fcadf3505b94e6254a9da74ac29be762d4fed68bd665e4adf8db277b651",
            "PASS": true
          },
          {
            "relative_path": "protocol/full_probe_execution_policy.json",
            "expected_sha256": "568abb9e26b12314a74ff46dd18cc8f9283a64c8fc435f0f446b91b4bd97b18c",
            "actual_sha256": "568abb9e26b12314a74ff46dd18cc8f9283a64c8fc435f0f446b91b4bd97b18c",
            "PASS": true
          },
          {
            "relative_path": "protocol/numerical_parity_policy.json",
            "expected_sha256": "5a871c3d1ef97d0f70bf9164afd3d81aa5cecfa48763eff8c5ce8ec7addaf073",
            "actual_sha256": "5a871c3d1ef97d0f70bf9164afd3d81aa5cecfa48763eff8c5ce8ec7addaf073",
            "PASS": true
          },
          {
            "relative_path": "provenance/CUMSUM_RUNTIME_SHAPE_AUDIT.json",
            "expected_sha256": "9391b9d968da95abafda53660e5a0ed691e1fcae5c5483643d45644a1bd10274",
            "actual_sha256": "9391b9d968da95abafda53660e5a0ed691e1fcae5c5483643d45644a1bd10274",
            "PASS": true
          },
          {
            "relative_path": "provenance/additional_deterministic_blocker_audit.json",
            "expected_sha256": "0f494bdb74e825db759e1b2a9283bf9795fc2c6bda1f50037eae18b855e58b4d",
            "actual_sha256": "0f494bdb74e825db759e1b2a9283bf9795fc2c6bda1f50037eae18b855e58b4d",
            "PASS": true
          },
          {
            "relative_path": "provenance/chunk_dtype_revalidation.json",
            "expected_sha256": "5cad28d894a956b8a05773071ff3a1db4e356e02b224a043a5787a09d27f50e5",
            "actual_sha256": "5cad28d894a956b8a05773071ff3a1db4e356e02b224a043a5787a09d27f50e5",
            "PASS": true
          },
          {
            "relative_path": "provenance/chunk_gdn_parity.json",
            "expected_sha256": "16a3908a4e07873d6b89dd6019bf20fa88a6e6f820316bc6b64be71d461a4343",
            "actual_sha256": "16a3908a4e07873d6b89dd6019bf20fa88a6e6f820316bc6b64be71d461a4343",
            "PASS": true
          },
          {
            "relative_path": "provenance/cumsum_runtime_shape_audit.json",
            "expected_sha256": "9391b9d968da95abafda53660e5a0ed691e1fcae5c5483643d45644a1bd10274",
            "actual_sha256": "9391b9d968da95abafda53660e5a0ed691e1fcae5c5483643d45644a1bd10274",
            "PASS": true
          },
          {
            "relative_path": "provenance/final_parent_stop_revalidation.json",
            "expected_sha256": "0cd06a626698d8a11d36b62f8738789f47efbe54f68a10095de0e01e06607b30",
            "actual_sha256": "0cd06a626698d8a11d36b62f8738789f47efbe54f68a10095de0e01e06607b30",
            "PASS": true
          },
          {
            "relative_path": "provenance/final_verdict.json",
            "expected_sha256": "a26234be05bbad633313da9b4161fed7dbbd6fb5ff9b212150c3a0c95b7d080b",
            "actual_sha256": "a26234be05bbad633313da9b4161fed7dbbd6fb5ff9b212150c3a0c95b7d080b",
            "PASS": true
          },
          {
            "relative_path": "provenance/fresh_process_determinism.json",
            "expected_sha256": "188b4f82f65892f173bc06fc2e0631d50363c056276bfddc1ba5cc6b0e5f1640",
            "actual_sha256": "188b4f82f65892f173bc06fc2e0631d50363c056276bfddc1ba5cc6b0e5f1640",
            "PASS": true
          },
          {
            "relative_path": "provenance/frozen_cumsum_inputs.json",
            "expected_sha256": "708836c24d0b31b3e2b487cf549aeaa1394aa0e5bc6141f6135443025839f7d6",
            "actual_sha256": "708836c24d0b31b3e2b487cf549aeaa1394aa0e5bc6141f6135443025839f7d6",
            "PASS": true
          },
          {
            "relative_path": "provenance/full_model_parity_1.error.json",
            "expected_sha256": "6baa967511dd3b82e73c7898081df8aa9587cba1af90d70a7a894ab5bee77676",
            "actual_sha256": "6baa967511dd3b82e73c7898081df8aa9587cba1af90d70a7a894ab5bee77676",
            "PASS": true
          },
          {
            "relative_path": "provenance/full_model_probe_parity.json",
            "expected_sha256": "7d6a663349bbf5459bd50e1b6940f326ca86012732f5a752ed2b657123922fe4",
            "actual_sha256": "7d6a663349bbf5459bd50e1b6940f326ca86012732f5a752ed2b657123922fe4",
            "PASS": true
          },
          {
            "relative_path": "provenance/operator_determinism.json",
            "expected_sha256": "9b91a38ccc41ecf605d3594d3d867d586a20e23ef42f581f93e94fb51f2656d8",
            "actual_sha256": "9b91a38ccc41ecf605d3594d3d867d586a20e23ef42f581f93e94fb51f2656d8",
            "PASS": true
          },
          {
            "relative_path": "provenance/operator_parity.json",
            "expected_sha256": "c47c8b1a1f3afa40c6f7de0e4d6e8a9dc98990248430b0f043eb8d6bdb45b85a",
            "actual_sha256": "c47c8b1a1f3afa40c6f7de0e4d6e8a9dc98990248430b0f043eb8d6bdb45b85a",
            "PASS": true
          },
          {
            "relative_path": "provenance/original_parent_stop_identity.json",
            "expected_sha256": "0cd06a626698d8a11d36b62f8738789f47efbe54f68a10095de0e01e06607b30",
            "actual_sha256": "0cd06a626698d8a11d36b62f8738789f47efbe54f68a10095de0e01e06607b30",
            "PASS": true
          },
          {
            "relative_path": "provenance/original_source_identity.json",
            "expected_sha256": "5abb3938769a69d2950dbde19371b47baf347b97a1407f24022f456d2938cae1",
            "actual_sha256": "5abb3938769a69d2950dbde19371b47baf347b97a1407f24022f456d2938cae1",
            "PASS": true
          },
          {
            "relative_path": "provenance/patch_diff.txt",
            "expected_sha256": "6170cf2b44bc50cb9ad72018c94c51cc9ddb2bc97c47d884f78d456313a43fa1",
            "actual_sha256": "6170cf2b44bc50cb9ad72018c94c51cc9ddb2bc97c47d884f78d456313a43fa1",
            "PASS": true
          },
          {
            "relative_path": "provenance/patched_source_identity.json",
            "expected_sha256": "1bd647a027a616962e2b1d86e3f99709d8f0d005116c136cd5ee0547ffce4625",
            "actual_sha256": "1bd647a027a616962e2b1d86e3f99709d8f0d005116c136cd5ee0547ffce4625",
            "PASS": true
          },
          {
            "relative_path": "provenance/performance_benchmark.json",
            "expected_sha256": "8becd0d9762c82dd62173887fd52bc0a7c833a1749c317e3f11e28bcb05853f9",
            "actual_sha256": "8becd0d9762c82dd62173887fd52bc0a7c833a1749c317e3f11e28bcb05853f9",
            "PASS": true
          },
          {
            "relative_path": "provenance/report_answers.json",
            "expected_sha256": "86be426d22828ad083c224c6a6838a480ac0b3231db2694c7d75815ee419fd1a",
            "actual_sha256": "86be426d22828ad083c224c6a6838a480ac0b3231db2694c7d75815ee419fd1a",
            "PASS": true
          },
          {
            "relative_path": "provenance/report_packaging_incident_001.json",
            "expected_sha256": "04a8f2d02b2528a797f95a71b3b0e716ee245bdf397177f28784a10fcf36eaf6",
            "actual_sha256": "04a8f2d02b2528a797f95a71b3b0e716ee245bdf397177f28784a10fcf36eaf6",
            "PASS": true
          },
          {
            "relative_path": "runtime_patch/deterministic_cumsum.py",
            "expected_sha256": "f1767d60da692119be4865e63424c0e4ac3a2193acbeec76cdd1ceaa755be31e",
            "actual_sha256": "f1767d60da692119be4865e63424c0e4ac3a2193acbeec76cdd1ceaa755be31e",
            "PASS": true
          },
          {
            "relative_path": "runtime_patch/original_production_chunk.py",
            "expected_sha256": "1066875a52cdd601582d9b3b8ebb31aa6a3d563de5dc34eb500ae860febbc7d0",
            "actual_sha256": "1066875a52cdd601582d9b3b8ebb31aa6a3d563de5dc34eb500ae860febbc7d0",
            "PASS": true
          },
          {
            "relative_path": "runtime_patch/patched_gdn.py",
            "expected_sha256": "d018009e6d605181358addab314f70401516f623505c5b2abf6d435df4be3bf0",
            "actual_sha256": "d018009e6d605181358addab314f70401516f623505c5b2abf6d435df4be3bf0",
            "PASS": true
          },
          {
            "relative_path": "runtime_patch/patched_production_chunk.py",
            "expected_sha256": "9b0ca6085436a5c5a35f06b292f1e610ac1c7285c0bd3b383798c2d2d1828577",
            "actual_sha256": "9b0ca6085436a5c5a35f06b292f1e610ac1c7285c0bd3b383798c2d2d1828577",
            "PASS": true
          },
          {
            "relative_path": "scripts/benchmark_patch.py",
            "expected_sha256": "057b350321eb15646a0c35e6e787d478084aab38b677465b927b7f92fe82902b",
            "actual_sha256": "057b350321eb15646a0c35e6e787d478084aab38b677465b927b7f92fe82902b",
            "PASS": true
          },
          {
            "relative_path": "scripts/capture_cumsum_fixture.py",
            "expected_sha256": "1ddbd7918ab7ff8a7cf1b548aa94f67dcc45f590b579928a12da1f920c199a7d",
            "actual_sha256": "1ddbd7918ab7ff8a7cf1b548aa94f67dcc45f590b579928a12da1f920c199a7d",
            "PASS": true
          },
          {
            "relative_path": "scripts/chunk_dtype_revalidation.py",
            "expected_sha256": "f78d5802bc1307da7ffd1ed500fc86611dfdb3ba6ca4034752381eef73494649",
            "actual_sha256": "f78d5802bc1307da7ffd1ed500fc86611dfdb3ba6ca4034752381eef73494649",
            "PASS": true
          },
          {
            "relative_path": "scripts/closure_common.py",
            "expected_sha256": "afdef1a4c3227918f7e2fd938dafa24ecce7f27ceee88e2af50be4393b7d9e65",
            "actual_sha256": "afdef1a4c3227918f7e2fd938dafa24ecce7f27ceee88e2af50be4393b7d9e65",
            "PASS": true
          },
          {
            "relative_path": "scripts/finalize_closure.failed_001.py",
            "expected_sha256": "45d7e4427a83b33231252417d5779c63aa9db36a077e33d481517af638a46f83",
            "actual_sha256": "45d7e4427a83b33231252417d5779c63aa9db36a077e33d481517af638a46f83",
            "PASS": true
          },
          {
            "relative_path": "scripts/finalize_closure.py",
            "expected_sha256": "2e698b6f82ab4abeaecef680ef8979bdc03f62bcaf0324dfa870a6a50df6e671",
            "actual_sha256": "2e698b6f82ab4abeaecef680ef8979bdc03f62bcaf0324dfa870a6a50df6e671",
            "PASS": true
          },
          {
            "relative_path": "scripts/qualify_operator.py",
            "expected_sha256": "9d738ba33f9d3b88b2aadfead0365e933ca2e314f8ea2a50297f6be2b2643502",
            "actual_sha256": "9d738ba33f9d3b88b2aadfead0365e933ca2e314f8ea2a50297f6be2b2643502",
            "PASS": true
          },
          {
            "relative_path": "scripts/run_chunk_parity.py",
            "expected_sha256": "5de4f78dfa408e7652ef1ace59c4b0819a979e9a04cd37c7ceba618821e153ef",
            "actual_sha256": "5de4f78dfa408e7652ef1ace59c4b0819a979e9a04cd37c7ceba618821e153ef",
            "PASS": true
          },
          {
            "relative_path": "scripts/run_full_probe_parity.py",
            "expected_sha256": "7de14bb24d82e18af1655b7691ae87bc08a61ba8440b9f1a58afad7fc57dd70f",
            "actual_sha256": "7de14bb24d82e18af1655b7691ae87bc08a61ba8440b9f1a58afad7fc57dd70f",
            "PASS": true
          },
          {
            "relative_path": "scripts/run_full_qualification.sh",
            "expected_sha256": "ea1e7ad9f702c0f529f51cee9bcffa82d470d4e58e5b996ac1fe6e68d5af8ab4",
            "actual_sha256": "ea1e7ad9f702c0f529f51cee9bcffa82d470d4e58e5b996ac1fe6e68d5af8ab4",
            "PASS": true
          },
          {
            "relative_path": "scripts/run_operator_qualification.sh",
            "expected_sha256": "878ec9656e570a9fe05f98828686f47fe5a9660cdae8ef2162e4fdfa582df696",
            "actual_sha256": "878ec9656e570a9fe05f98828686f47fe5a9660cdae8ef2162e4fdfa582df696",
            "PASS": true
          },
          {
            "relative_path": "scripts/test_deterministic_cumsum.py",
            "expected_sha256": "c5b73bf6492734a5130da666feec3b7fee2c7c3f62bbe1a34b1811f619c29cec",
            "actual_sha256": "c5b73bf6492734a5130da666feec3b7fee2c7c3f62bbe1a34b1811f619c29cec",
            "PASS": true
          }
        ]
      },
      "preflight": {
        "root": "/data/zypan/worktrees/qwen-gdn-frozen-drive-counterfactual-replay-v1/experiments/QWEN_GDN_FROZEN_DRIVE_COUNTERFACTUAL_REPLAY_V1",
        "manifest": "/data/zypan/worktrees/qwen-gdn-frozen-drive-counterfactual-replay-v1/experiments/QWEN_GDN_FROZEN_DRIVE_COUNTERFACTUAL_REPLAY_V1/SHA256SUMS",
        "manifest_sha256": "aef142d3f970f1b12434268f2d2c3d5b3633dba1e9cb6fefd3262817f1ef9f17",
        "sealed_artifact_count": 8,
        "all_exact": true,
        "rows": [
          {
            "relative_path": "FINAL_REPORT.md",
            "expected_sha256": "f46f2325d77d94db9921ad87160c6a636828f4765a8f57aec2c978567f0e4397",
            "actual_sha256": "f46f2325d77d94db9921ad87160c6a636828f4765a8f57aec2c978567f0e4397",
            "PASS": true
          },
          {
            "relative_path": "execution_status.json",
            "expected_sha256": "8f67eab228bfdf9b42bde84dcbaaa45852a8608febbccfc674e591016cd404de",
            "actual_sha256": "8f67eab228bfdf9b42bde84dcbaaa45852a8608febbccfc674e591016cd404de",
            "PASS": true
          },
          {
            "relative_path": "final_parent_seal_revalidation.json",
            "expected_sha256": "611b1a9edec7155959262067847a7441267a1e2c9c9b6f6f7c23bdbb15c8612e",
            "actual_sha256": "611b1a9edec7155959262067847a7441267a1e2c9c9b6f6f7c23bdbb15c8612e",
            "PASS": true
          },
          {
            "relative_path": "final_verdict.json",
            "expected_sha256": "8beeffb3cf3da82513bc2c76f264279d62d99a2057f5a812b71629ddc9827658",
            "actual_sha256": "8beeffb3cf3da82513bc2c76f264279d62d99a2057f5a812b71629ddc9827658",
            "PASS": true
          },
          {
            "relative_path": "frozen_input_identity.json",
            "expected_sha256": "63efc539c26f2b82921b885a81e4c304ad846a824342217be4284d3619c6355f",
            "actual_sha256": "63efc539c26f2b82921b885a81e4c304ad846a824342217be4284d3619c6355f",
            "PASS": true
          },
          {
            "relative_path": "parent_seal_check.json",
            "expected_sha256": "ce5ead164438c904b433c53fd1e858fca0431879f1d65ffb7e2e35c2233e6d32",
            "actual_sha256": "ce5ead164438c904b433c53fd1e858fca0431879f1d65ffb7e2e35c2233e6d32",
            "PASS": true
          },
          {
            "relative_path": "preflight_parent_inventory.json",
            "expected_sha256": "9e3317db58bebcc3a0fef5fddee3b6a28567d1231b2c14925d336dc7e599db82",
            "actual_sha256": "9e3317db58bebcc3a0fef5fddee3b6a28567d1231b2c14925d336dc7e599db82",
            "PASS": true
          },
          {
            "relative_path": "scripts/preflight_and_stop.py",
            "expected_sha256": "dfb1645865c173eaa8b591067a759531f3abef9e807fd403316731d722432622",
            "actual_sha256": "dfb1645865c173eaa8b591067a759531f3abef9e807fd403316731d722432622",
            "PASS": true
          }
        ]
      },
      "amendment": {
        "root": "/data/zypan/worktrees/qwen-gdn-frozen-drive-counterfactual-replay-v1/experiments/QWEN_GDN_FROZEN_DRIVE_COUNTERFACTUAL_REPLAY_V1/amendments/PREMEASUREMENT_INPUT_FREEZE_AMENDMENT_V1",
        "manifest": "/data/zypan/worktrees/qwen-gdn-frozen-drive-counterfactual-replay-v1/experiments/QWEN_GDN_FROZEN_DRIVE_COUNTERFACTUAL_REPLAY_V1/amendments/PREMEASUREMENT_INPUT_FREEZE_AMENDMENT_V1/SHA256SUMS",
        "manifest_sha256": "d35012f2c843775b95be9979caa0fdfd4cc937dbf3cabbebb70340f56816d9e0",
        "sealed_artifact_count": 13,
        "all_exact": true,
        "rows": [
          {
            "relative_path": "FINAL_AMENDMENT_REPORT.md",
            "expected_sha256": "4e726d807fa0580332080615c0ec14fd02824755ab8cace152f3de94902f9292",
            "actual_sha256": "4e726d807fa0580332080615c0ec14fd02824755ab8cace152f3de94902f9292",
            "PASS": true
          },
          {
            "relative_path": "discovery_selection_rule_recovery.json",
            "expected_sha256": "ff9819d26e1901a1c376710dfa299dc896253f14293346f5c350eae72fd2ff15",
            "actual_sha256": "ff9819d26e1901a1c376710dfa299dc896253f14293346f5c350eae72fd2ff15",
            "PASS": true
          },
          {
            "relative_path": "discovery_selection_rule_recovery.md",
            "expected_sha256": "e36d8fee84441e1f8ad5fd57495c2f2fd5e6d5c371073306c92f329c49d7a4d1",
            "actual_sha256": "e36d8fee84441e1f8ad5fd57495c2f2fd5e6d5c371073306c92f329c49d7a4d1",
            "PASS": true
          },
          {
            "relative_path": "execution_status.json",
            "expected_sha256": "c312d8f25583e8fed62cbd950bd83625187ebb00a2640c53cd33f48fe7856aa7",
            "actual_sha256": "c312d8f25583e8fed62cbd950bd83625187ebb00a2640c53cd33f48fe7856aa7",
            "PASS": true
          },
          {
            "relative_path": "final_old_seal_revalidation.json",
            "expected_sha256": "b19a5ae93af1cd1daceae93738cf22b34ce10047521110106e223b38e2c36734",
            "actual_sha256": "b19a5ae93af1cd1daceae93738cf22b34ce10047521110106e223b38e2c36734",
            "PASS": true
          },
          {
            "relative_path": "final_verdict.json",
            "expected_sha256": "1b0a70068b576ba983d62fc9e1497c8693bf09d50b7a4bbe7248013f6cf7c170",
            "actual_sha256": "1b0a70068b576ba983d62fc9e1497c8693bf09d50b7a4bbe7248013f6cf7c170",
            "PASS": true
          },
          {
            "relative_path": "pre_amendment_evidence_inventory.json",
            "expected_sha256": "89f5535bcede1e61355142e6b528add80e2780a79264cd3c5a79d02faf771c75",
            "actual_sha256": "89f5535bcede1e61355142e6b528add80e2780a79264cd3c5a79d02faf771c75",
            "PASS": true
          },
          {
            "relative_path": "rotation_bank_rule_recovery.json",
            "expected_sha256": "b28dd1a9490397667fc44688a441b0acb79396a8a37f2398607039c118dcdb9e",
            "actual_sha256": "b28dd1a9490397667fc44688a441b0acb79396a8a37f2398607039c118dcdb9e",
            "PASS": true
          },
          {
            "relative_path": "rotation_bank_rule_recovery.md",
            "expected_sha256": "75666ac1b78bdca0f9d18fe6ed3295a8be07dde21bfa72bf494b8f3da0591b88",
            "actual_sha256": "75666ac1b78bdca0f9d18fe6ed3295a8be07dde21bfa72bf494b8f3da0591b88",
            "PASS": true
          },
          {
            "relative_path": "scientific_measurement_audit.json",
            "expected_sha256": "f0b7086e6a60c77a43dfe829cb0249ccfffafdeced35a65957c8eb0d4338cc9c",
            "actual_sha256": "f0b7086e6a60c77a43dfe829cb0249ccfffafdeced35a65957c8eb0d4338cc9c",
            "PASS": true
          },
          {
            "relative_path": "source/amendment_task_original.txt",
            "expected_sha256": "10a307a665dda489399188f2f6c7ae291ae0a836649ddb1620454280fea1ec37",
            "actual_sha256": "10a307a665dda489399188f2f6c7ae291ae0a836649ddb1620454280fea1ec37",
            "PASS": true
          },
          {
            "relative_path": "source/audit_and_stop.py",
            "expected_sha256": "ef022fba7f188dd3c614a99e0dd3d39e2789d3983a89ba773dbc92a51d2d871f",
            "actual_sha256": "ef022fba7f188dd3c614a99e0dd3d39e2789d3983a89ba773dbc92a51d2d871f",
            "PASS": true
          },
          {
            "relative_path": "source/parent_task_original.txt",
            "expected_sha256": "8b9fdc7facb109c41704774b7f2f5b315b62c578554665e39767ef1d5af664e4",
            "actual_sha256": "8b9fdc7facb109c41704774b7f2f5b315b62c578554665e39767ef1d5af664e4",
            "PASS": true
          }
        ]
      }
    },
    "unchanged_verdicts": {
      "parent": {
        "C128_QDQ_IDENTITY_GATE": "PASS",
        "CONFIRMATION_32K_MEDIAN_CONCORDANCE": "NA",
        "CONFIRMATION_32K_MEDIAN_RHO": "NA",
        "CONFIRMATION_32K_POSITIVE_RHO": "NA",
        "CONFIRMATION_COMPLETE": "0/48",
        "CONFIRMATION_SELECTED_BEATS_JINJ_COUNT": "NA",
        "CROSS_TERM_RESIDUAL": "DESCRIPTIVE_ONLY_NOT_MEASURED",
        "DISCOVERY_COMPLETE": "0/48",
        "DISCOVERY_JDECAY_MEDIAN_RHO_32K": "NA",
        "DISCOVERY_JINJ_MEDIAN_RHO_2K": "NA",
        "DISCOVERY_JINJ_MEDIAN_RHO_32K": "NA",
        "DISCOVERY_JINJ_MEDIAN_RHO_8K": "NA",
        "DISCOVERY_JPROP_MEDIAN_RHO_32K": "NA",
        "DISCOVERY_SELECTED_BEATS_JINJ_COUNT": "NA",
        "DISCOVERY_SURROGATE_GATE": "NOT_EVALUATED",
        "DOCUMENT_OVERLAP_AUDIT_GATE": "NOT_EVALUATED",
        "EVIDENCE_TYPE": "PROSPECTIVE_CROSS_ROTATION_LONG_HORIZON_MECHANISM_VALIDATION",
        "EXPERIMENT": "QWEN_GDN_PROPAGATION_SURROGATE_VALIDATION_V1",
        "FINAL_ACTION": "STOP",
        "FRESH_DOCUMENT_SELECTION_GATE": "NOT_EVALUATED",
        "GDN_DRIVEN_ERROR_RECURRENCE_GATE": "NOT_EVALUATED",
        "GDN_RECURRENCE_SOURCE_IDENTITY_GATE": "PASS",
        "GDN_ZERO_WRITE_TRANSITION_PARITY_GATE": "NOT_EVALUATED",
        "J_DECAY_AVAILABLE": "YES",
        "NATIVE_BASIS_MAPPING_GATE": "NOT_EVALUATED",
        "NEW_AIME_GENERATION": 0,
        "NEW_C8_TRAINING_AUTHORIZED": "NO",
        "NEW_FREE_GENERATION": 0,
        "NEW_TRAINING": "NO",
        "PHYSICAL_TRANSITION_ROTATION_INVARIANCE": "NOT_EVALUATED",
        "PRIMARY_NEXT_STEP_CATEGORY": "STOP",
        "PROPAGATION_MC_PROBE_COUNT": "NOT_FROZEN",
        "QWEN_GDN_PROPAGATION_SURROGATE_CONFIRMATION": "NOT_EVALUATED",
        "QWEN_RUNTIME_IDENTITY_GATE": "FAIL",
        "REASONING_IMPROVEMENT": "NOT_EVALUATED",
        "ROTATION_BANK_COUNT": 0,
        "ROTATION_BANK_FREEZE_GATE": "NOT_EVALUATED",
        "ROTATION_BANK_ORTHOGONALITY_GATE": "NOT_EVALUATED",
        "ROTATION_BANK_UNIQUENESS_GATE": "NOT_EVALUATED",
        "ROTATION_PARAMETERIZATION_IDENTITY_GATE": "PASS",
        "SELECTED_SURROGATE": "NONE",
        "TASK_STATUS": "STOP_STRICT_DETERMINISTIC_RUNTIME_UNAVAILABLE"
      },
      "cumsum": {
        "errors": [
          {
            "evidence": {
              "additional_unsupported_deterministic_op": false,
              "message": "full model numerical parity failed; STOP",
              "stop": true,
              "traceback": "Traceback (most recent call last):\n  File \"/data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1/experiments/QWEN_GDN_DETERMINISTIC_CUMSUM_CLOSURE_V1/scripts/run_full_probe_parity.py\", line 105, in <module>\n    else:globals()[a.phase]()\n  File \"/data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1/experiments/QWEN_GDN_DETERMINISTIC_CUMSUM_CLOSURE_V1/scripts/run_full_probe_parity.py\", line 86, in parity\n    C.check(gate=='PASS','full model numerical parity failed; STOP')\n  File \"/data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1/experiments/QWEN_GDN_DETERMINISTIC_CUMSUM_CLOSURE_V1/scripts/closure_common.py\", line 22, in check\n    if not ok: raise RuntimeError(msg)\nRuntimeError: full model numerical parity failed; STOP\n"
            },
            "path": "provenance/full_model_parity_1.error.json",
            "sha256": "6baa967511dd3b82e73c7898081df8aa9587cba1af90d70a7a894ab5bee77676"
          }
        ],
        "failed_gates": [
          "FULL_MODEL_SHORT_PROBE_PARITY_GATE",
          "PATCHED_FULL_MODEL_FRESH_PROCESS_DETERMINISM_GATE"
        ],
        "parent_confirmation_started": 0,
        "parent_discovery_started": 0,
        "parent_stop_unchanged": true,
        "verdict": {
          "ADDITIONAL_UNSUPPORTED_DETERMINISTIC_OP_COUNT": 0,
          "BLOCKED_OPERATION": "torch.cumsum",
          "BLOCKED_SOURCE_LOCATION": "/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py:291",
          "CUDA_RUNTIME": "12.1",
          "CUMSUM_NUMERICAL_PARITY_GATE": "PASS",
          "CUMSUM_RUNTIME_DIM_LENGTH": 64,
          "DETERMINISTIC_CUMSUM_COMPATIBILITY_GATE": "FAIL",
          "EVIDENCE_TYPE": "ENGINEERING_DETERMINISTIC_RUNTIME_COMPATIBILITY_CLOSURE",
          "EXPERIMENT": "QWEN_GDN_DETERMINISTIC_CUMSUM_CLOSURE_V1",
          "FINAL_ACTION": "FAIL_AND_STOP",
          "FIXED_ORDER_CUMSUM_DETERMINISM_GATE": "PASS",
          "FULL_MODEL_SHORT_PROBE_PARITY_GATE": "FAIL",
          "NEW_AIME_GENERATION": 0,
          "NEW_C8_TRAINING_AUTHORIZED": "NO",
          "NEW_FREE_GENERATION": 0,
          "NEW_TRAINING": "NO",
          "PARENT_EXPERIMENT": "QWEN_GDN_PROPAGATION_SURROGATE_VALIDATION_V1",
          "PARENT_PROPAGATION_EXPERIMENT_AUTHORIZED_TO_RESUME": "NO",
          "PARENT_STOP_PRESERVED": "YES",
          "PATCHED_CHUNK_GDN_PARITY_GATE": "PASS",
          "PATCHED_FULL_MODEL_FRESH_PROCESS_DETERMINISM_GATE": "NOT_EVALUATED",
          "PATCHED_PREFILL_SLOWDOWN": null,
          "RESUME_FROM_GATE": "NONE",
          "SURGICAL_PATCH_ROUTE": "NOT_EVALUATED",
          "TORCH_VERSION": "2.5.1+cu121"
        }
      },
      "preflight": {
        "gpu_computation_started": false,
        "old_experiment_files_modified": 0,
        "parent_scientific_resume_started": false,
        "scientific_measurements": 0,
        "stop_phase": "Phase0 / G1",
        "stop_reason": "Parent sealed evidence does not contain selected/hash-frozen Discovery documents or an instantiated/hash-frozen12-condition rotation bank. New task sections4, G1 and31 require immediate stop and forbid selecting/building replacements.",
        "verdict": {
          "CAPTURE_INTEGRITY_GATE": "NOT_EVALUATED",
          "CONFIRMATION_AUTHORIZED_AUTOMATICALLY": "NO",
          "CONFIRMATION_DOCS_EXECUTED": 0,
          "CONFIRMATION_DOCS_READ": "NO",
          "COUNTERFACTUAL_DECOMPOSITION_GATE": "NOT_EVALUATED",
          "DISCOVERY_COMPLETE": "0/48",
          "DISCOVERY_ONLY": "YES",
          "DRIVE_FACTORIZATION_GATE": "NOT_EVALUATED",
          "ENGINEERING_PREDECESSOR": "QWEN_GDN_DETERMINISTIC_CUMSUM_CLOSURE_V1",
          "EVIDENCE_TYPE": "PROSPECTIVE_CONTROLLED_RECURRENT_MECHANISM_DISCOVERY",
          "EXPERIMENT": "QWEN_GDN_FROZEN_DRIVE_COUNTERFACTUAL_REPLAY_V1",
          "FINAL_ACTION": "FAIL_AND_STOP",
          "FP_REPLAY_PARITY_GATE": "NOT_EVALUATED",
          "FROZEN_INPUT_IDENTITY_GATE": "FAIL",
          "FULL_MODEL_CLOSED_LOOP_PROPAGATION": "NOT_EVALUATED",
          "HORIZON_2048_COMPLETE": "0/48",
          "HORIZON_32768_COMPLETE": "0/48",
          "HORIZON_8192_COMPLETE": "0/48",
          "NEW_AIME_GENERATION": 0,
          "NEW_C8_TRAINING_AUTHORIZED": "NO",
          "NEW_FREE_GENERATION": 0,
          "NEW_TRAINING": "NO",
          "OLD_CUMSUM_CLOSURE_FAIL_PRESERVED": "YES",
          "OPTIMIZER_STEP": 0,
          "PARENT_EXPERIMENT": "QWEN_GDN_PROPAGATION_SURROGATE_VALIDATION_V1",
          "PARENT_PROPAGATION_EXPERIMENT_AUTHORIZED_TO_RESUME": "NO",
          "PARENT_SEAL_INTEGRITY_GATE": "PASS",
          "PARENT_STOP_PRESERVED": "YES",
          "PHYSICAL_TRANSITION_ROTATION_INVARIANCE_GATE": "NOT_EVALUATED",
          "REPLAY_DETERMINISM_GATE": "NOT_EVALUATED",
          "SCIENTIFIC_SCOPE": "INTRINSIC_RECURRENT_ERROR_DYNAMICS_UNDER_FROZEN_EXTERNAL_DRIVE",
          "STREAMING_REDUCER_PARITY_GATE": "NOT_EVALUATED"
        }
      },
      "amendment": {
        "audit_commit_is_not_premeasurement_freeze_commit": true,
        "no_freeze_manifest_created": true,
        "old_artifacts_modified": 0,
        "stop_reason": [
          "authoritative candidate-pool manifest/path",
          "pinned dataset revision/record identity",
          "applicable pre-existing ordering/comparator",
          "applicable tie policy if ordering keys collide"
        ],
        "verdict": {
          "AMENDMENT": "QWEN_GDN_FROZEN_DRIVE_COUNTERFACTUAL_REPLAY_V1_PREMEASUREMENT_INPUT_FREEZE_AMENDMENT_V1",
          "CONFIRMATION_DOCS_EXECUTED": 0,
          "CONFIRMATION_DOCS_READ": "NO",
          "CUMSUM_CLOSURE_SEAL_INTEGRITY_GATE": "PASS",
          "DISCOVERY_COMPLETE": "0/48",
          "DISCOVERY_DOCUMENT_FREEZE_GATE": "NOT_EVALUATED",
          "DISCOVERY_DOCUMENT_RULE_RECOVERY_GATE": "FAIL",
          "DISCOVERY_DOC_COUNT": 0,
          "FINAL_ACTION": "FAIL_AND_STOP",
          "FROZEN_INPUT_IDENTITY_GATE": "FAIL",
          "HORIZON_POLICY_FREEZE_GATE": "NOT_EVALUATED",
          "METRIC_DEFINITION_FREEZE_GATE": "NOT_EVALUATED",
          "NEW_AIME_GENERATION": 0,
          "NEW_C8_TRAINING_AUTHORIZED": "NO",
          "NEW_FREE_GENERATION": 0,
          "NEW_TRAINING": "NO",
          "NEXT_AUTHORIZED_PHASE": "NONE",
          "NUMERICAL_POLICY_FREEZE_GATE": "NOT_EVALUATED",
          "OLD_CUMSUM_CLOSURE_FAIL_PRESERVED": "YES",
          "OPTIMIZER_STEP": 0,
          "PARENT_PROPAGATION_EXPERIMENT_AUTHORIZED_TO_RESUME": "NO",
          "PARENT_SEAL_INTEGRITY_GATE": "PASS",
          "PARENT_STOP_PRESERVED": "YES",
          "PREMEASUREMENT_COMMIT_SHA": null,
          "PREMEASUREMENT_FREEZE_COMPLETE": "NO",
          "PREMEASUREMENT_FREEZE_MANIFEST_GATE": "NOT_EVALUATED",
          "PREVIOUS_PREFLIGHT_FAIL_PRESERVED": "YES",
          "PREVIOUS_PREFLIGHT_SEAL_INTEGRITY_GATE": "PASS",
          "PROSPECTIVE_INPUT_FREEZE_ELIGIBLE": "YES",
          "PUSH_PERFORMED": "NO",
          "ROTATION_BANK_COUNT": 0,
          "ROTATION_BANK_MATERIALIZATION_GATE": "NOT_EVALUATED",
          "ROTATION_BANK_ORTHOGONALITY_GATE": "NOT_EVALUATED",
          "ROTATION_BANK_RULE_RECOVERY_GATE": "NOT_EVALUATED",
          "SCIENTIFIC_MEASUREMENT_COUNT_BEFORE_FREEZE": 0,
          "SCIENTIFIC_METRIC_EXECUTION_COUNT": 0
        }
      }
    },
    "failed_amendment_audit_commit": "196e1dac33d365dbf48730862465e30a41f9b9ae",
    "new_repo": "/data/zypan/worktrees/qwen-gdn-frozen-drive-anchor-mechanism-v1",
    "new_branch": "exp/qwen-gdn-frozen-drive-anchor-mechanism-v1",
    "new_base_commit": "c5795f2d47ca9b4e189ec3f425b0b86e7ee95564",
    "authoritative_original_paths_used": true,
    "CONFIRMATION_EXECUTED": 0
  },
  "previous_surgical_fix_reused": true
}