# Code provenance

This inventory contains only project-owned code whose use is supported by frozen artifacts, reports, manifests, or commit history. It does not claim that third-party packages are project code.

| Role | File | Git tracked? | SHA256 | Used for |
|---|---|---:|---|---|
| trajectory_metric_source | `experiments/LING_L6_L7_PAIRED_TRAJECTORY_CLOSURE_V1/provenance/code_snapshots/analyze_long_generations.py` | YES (archive branch) | `8adc34093e59158a7bd2287f91a3ee167f04c3953ed82a86f18629eee82d14dc` | Established n-gram/repetition metric definitions reused by baseline extraction |
| baseline_extraction | `experiments/LING_L6_L7_PAIRED_TRAJECTORY_CLOSURE_V1/scripts/extract_baselines.py` | YES (archive branch) | `75c5d2bee8111662443ef36c65f014fd815f2abdcfe218bb1034b8c91b73f0c4` | Read-only extraction of frozen FP/H trajectory diagnostics |
| trajectory_analysis | `experiments/LING_L6_L7_PAIRED_TRAJECTORY_CLOSURE_V1/scripts/run_analysis.py` | YES (archive branch) | `9adefadd731fd78e208c5a1f25d9c10a1b080a6b5ceb10bb12505b96c3a94365` | Outcome reconstruction, paired statistics, trajectory/switch analysis, rotation sanity, integrity inventory |
| scoring | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/aime26_scorer_v4.py` | YES (archive branch) | `fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b` | Frozen V4 answer extraction and correctness logic |
| parity | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/analyze_cross_host_parity.py` | YES (archive branch) | `e63cf2232528e1749d79b337a7dec63a5455cd08fcf9ffce872f38b97cc53a60` | FP 4090/3090 cross-host parity analysis |
| parity | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/analyze_unified_h_parity.py` | YES (archive branch) | `ad0aa24bd1ea083c20d9820917a43bf9508827c926b0669629d30ce9cd085af8` | Unified Hadamard runtime parity analysis |
| training | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/collect_validation_after_train.sh` | YES (archive branch) | `5e9ad189b4565aea38868a885be92d40a69343144bf2054fe54900a630a4534e` | Post-training validation collection |
| parity | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/compare_condition_int8_parity.py` | YES (archive branch) | `88377bf06b266824d1f8b9a46921b7519afa46f04fb164d172c9c638e9ca86e9` | Condition-level INT8 bitwise comparison |
| finalization | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/finalize_amended_hardware.py` | YES (archive branch) | `d1940b02a4e027a4025851c9d4de8149a2efc9e375128b02764a18a7bd4de9f5` | Canonical amended hardware metadata finalization |
| finalization | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/finalize_amendment_overlay.py` | YES (archive branch) | `8a2d6b0a87b95b11556827a5d854b143f043910f4a335a987bf9e159ac5fd2f0` | Protocol-amendment overlay finalization |
| finalization | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/finalize_experiment.py` | YES (archive branch) | `2a84500ddce88578dd320345cf660413721931c28af6cf3ca82b70e9f3944740` | Final experiment aggregation and report generation |
| training | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/finalize_l3_early_stop.py` | YES (archive branch) | `656ca6c88ebde28cb0fc42fa6af5a45babc931b531a45c099673b8e47d3f2f9f` | Frozen L3 early-stop control used by training workflow |
| training | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/finalize_l3_when_complete.sh` | YES (archive branch) | `3cc15c2f6f087aab1b1b7f91fb12615ce2ffaea6d5d1ffc6b2835b1308f3aab4` | L3 completion wrapper used by training workflow |
| training | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/finalize_trace_manifests.py` | YES (archive branch) | `6f29a8c11cb4e0ffbf24be0724c17c1272c60bdfb755d67112c3010990bd2932` | Training trace manifest finalization |
| training | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/finalize_training_metadata.py` | YES (archive branch) | `a6707dc55283e170f280fe3a56c7576bdf7ae1412e5785ae164b0876e32f8eb6` | Training metadata finalization |
| evaluation | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/launch_formal_server_3090.sh` | YES (archive branch) | `6ba4db7a38d8c28e448eb39ba6318e01875eb4b6e3ae6003d40cd9676a7dd0c8` | 3090 formal runtime launcher |
| runtime | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/launch_ling_sglang_from_patch.py` | YES (archive branch) | `7e1c0bc87f3cb707e3eb76b1dc9ad4387cfdb0f97c4d8b10ad7143dc93c7701c` | Project-owned SGLang patched-runtime launcher |
| runtime | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/launch_parity_server.sh` | YES (archive branch) | `13fb54f47f2e894b1a7b26d82813127bd8ecfb1badc9edd83ef0da24679aa07b` | Deterministic parity server launcher |
| evaluation | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/make_baseline_reuse_manifest.py` | YES (archive branch) | `b7c60c0891391e4799ae7ecdaff5a885a4d5c3bfac3a7c733e44810dea7a1b64` | Frozen baseline reuse manifest construction |
| training | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/recurrent_dense_train.py` | YES (archive branch) | `d1c433598875ea84c80c6f6d4fb5fe48088d5ca0bf44ccff9d950a77560fd9ae` | Recurrent writeback, TBPTT, L6/L7 objectives, rotation export |
| parity | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/run_condition_int8_parity_3090.sh` | YES (archive branch) | `48cd7e468b31a26f0a0abfdcc263702ebf64cd35aa27d3d584741807989c634f` | 3090 condition-level INT8 gate |
| parity | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/run_condition_int8_parity_4090.sh` | YES (archive branch) | `da5e3b0c24d411477bfa040ddd020e08d8d56346e4c1a64783c02e674e4a9a1c` | 4090 condition-level INT8 gate |
| parity | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/run_cross_host_parity_3090.sh` | YES (archive branch) | `02d37cfd24b45039f4f3f929c46acfe35976f48f77ef01e219fb9830b97a5128` | 3090 cross-host parity capture |
| parity | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/run_cross_host_parity_when_free.sh` | YES (archive branch) | `6a1792d943a9646f1ee4e580af66028fced05707e8fd985616d6562fd7af6bf5` | 4090 cross-host parity capture wrapper |
| evaluation | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/run_formal_condition_3090.sh` | YES (archive branch) | `f6f00201bff187e1df950dc11e0cbee96b99b9e6b3c8660d97b034cbad06d1c7` | 3090 formal shard launcher |
| evaluation | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/run_formal_condition_3090_amendment_v2.sh` | YES (archive branch) | `d864ba2a292c0e389d7364dd619d837091d3c9ed133d2bd2b8554d215380fa8a` | Amended 3090 formal shard launcher |
| evaluation | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/run_formal_condition_4090.sh` | YES (archive branch) | `096c758a9fc3144af80f1a616c219156525ca3031ca24cd4f0aefb9d8d4e05c1` | 4090 formal shard launcher |
| evaluation | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/run_formal_condition_4090_amendment_v2.sh` | YES (archive branch) | `7d95a7792ca828b097ed685e3835603b866b12c3b1dd8c10469ab89ec4da9ccd` | Amended 4090 formal shard launcher |
| evaluation | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/run_formal_l6_l7.sh` | YES (archive branch) | `b55a3815b73913e4e86caefcfff4299fdb66c30fab1136610c7d9187b275718b` | Formal L6/L7 evaluation orchestrator |
| evaluation | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/run_formal_recurrent.py` | YES (archive branch) | `efce71e62e8e996d3958429cc60e3512393c65938dcf4ef97190ed39273b93b4` | Canonical recurrent generation client and metadata writer |
| evaluation | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/run_l7_idle_4090_gpu1_helper_v3.sh` | YES (archive branch) | `7d6e201ecde578df682f5a9dafc0889cd768358d56b4fc3d05e4124e94d13872` | Frozen L7 idle-4090 helper V3 |
| evaluation | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/run_l7_idle_4090_gpu1_helper_v4.sh` | YES (archive branch) | `34c88cd89f6d023ea090cc572d928c9a8b87c44db209f500e44bf33d587e867f` | Frozen L7 helper ordering V4 |
| evaluation | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/run_q29_4090_helper.sh` | YES (archive branch) | `c299a7112fd970c5275d11cc6c85f9d46c7ea270abaf349aea0bd022d547e3e1` | Frozen q29 4090 helper amendment |
| runtime | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/run_runtime_gate_client.py` | YES (archive branch) | `aadfe6c9fa8cc2815597b19db7d3c6116d9a14b757aadf2cd8ca0fdb40b806ec` | Runtime semantic and numerical gate client |
| training | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/run_training_pipeline.sh` | YES (archive branch) | `02717382c6711133f8b3d045fb17db62ded246f6a3618440212961aca2d28056` | Frozen recurrent training orchestrator |
| parity | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/run_unified_h_parity.sh` | YES (archive branch) | `060d369c06de93a0edfac4f93ba99199af0b347a3138b4e391391e4795d86345` | Unified Hadamard parity capture |
| training | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/sanity_compare.py` | YES (archive branch) | `1084a363c2782cd73435ef4bdd5b1cb1bac35c06dce4391e2c71451f9c2b3ccc` | Rotation/numerical sanity comparison |
| scoring | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/score_recurrent_formal.py` | YES (archive branch) | `ebf3a05ece99be29407b9124a3c915dd6819a0563c9f11693ed4c0e52662b208` | Frozen formal-output scorer wrapper |
| runtime | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/sglang_kda_unified_final_r_patch.py` | YES (archive branch) | `e5e17b09a40ae9bf3702cdf417538f759d213e2b8ecdbd5d1cda6ca964667aef` | Project-owned unified final-R recurrent-state patch |
| parity | `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/tensor_bitwise_manifest.py` | YES (archive branch) | `24bac54221f0a9fbf8e4b94657c4fe2285bd700c43b3cc0a841acb755124a567` | Canonical tensor hashing for bitwise gates |

## Evidence rule

Each upstream L6/L7 script is named by frozen experiment metadata and has an introduction commit in the experiment lineage. The two newly archived closure sources are byte-exact snapshots whose hashes are independently recorded in the frozen provenance artifacts. No environment package or third-party repository was copied wholesale.

## Third-party runtime identity

The frozen provenance records SGLang `0.5.19` at source commit `0bcd822377da7b5718e674eaf9c870d349424dd1`; PyTorch `2.9.1+cu128`; Triton `3.5.1`; and FLA `0.5.2`. Project changes are represented by the archived patch/launcher files above.
