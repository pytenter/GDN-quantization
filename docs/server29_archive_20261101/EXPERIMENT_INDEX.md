# .29 既有 GDN/KDA 实验数据归档索引

本次仅收集既有文件，没有训练、生成、评分重跑或GPU实验。原实验目录、旧工作树、远端历史均未覆盖。

“来源包”是不同位置/版本的一组证据，不等于一次独立科学实验；同一实验可能在 worktree、归档和 .55 迁移备份重复出现。状态是所链接原文件的记载，无法从顶层证据确认则明确为 UNKNOWN。不能用子门禁PASS推断正式实验完成。

全量机器清单：[artifact_index.jsonl.gz](../../results/server29/20261101/artifact_index.jsonl.gz)。每个来源包另有 SUMMARY 和逐文件清单。按 SHA256 存储原样去重对象；链接到 main 的已存在文件不重复复制。完整 Git 分支/commit 与不同内容版本逐文件保留。

[全分支blob来源索引](../../results/server29/20261101/ALL_BRANCH_BLOB_INDEX.json.gz)保留完整引用列表；逐文件github_sources只显示至多12处匹配，完整匹配可按existing_public_git_blob_sha1查询此表。

单独的[未上传文件清单](../../results/server29/20261101/excluded_artifacts.jsonl.gz)列出源路径、大小、可得SHA和原因；[扫描边界](SKIPPED_SCAN_BOUNDARIES.json)记录没有进入的目录或非regular文件，凭据不会被读取来补算哈希。

大型数值表中符合限制者另以gzip无损压缩：[表格清单](../../results/server29/20261101/lossless_compressed_tables/TABLES.json)。其source SHA和压缩文件SHA分列，解压后逐字节SHA已复核。对象去重不保留原目录布局，原报告的相对链接需按清单重建；入口索引直接链接实际归档位置。

[实验名称目录](../../results/server29/20261101/EXPERIMENT_NAME_CATALOG.json) 将实际路径中的研究标识关联到全部来源包，仅是检索分组，不认证同名来源的协议等价，也不合并成绩。

## 必须区分的协议

Ling 81,920-token 历史 FP44/60 不是 256K Canonical60 的匹配 FP 对照。已从固定分支恢复原报告与逐样本CSV：H35/60、L6 36/60、L7 33/60；FP仅14/20可用子集。60个唯一sample、seed1/2各30，已只读重算计数并与原汇总一致。没有运行模型或重新解码评分。

[恢复的原始报告](../../results/server29/20261101/historical_branch/LING_L6_L7_CANONICAL_60_CONFIRMATORY_V1/reports/FINAL_REPORT.md) · [逐样本原表](../../results/server29/20261101/historical_branch/LING_L6_L7_CANONICAL_60_CONFIRMATORY_V1/analysis/canonical60_outcomes.csv) · [独立数量核对](CANONICAL60_COUNT_RECONCILIATION.json)

PACE 原已评分布尔字段只读计数：原 PACE R2=21/30、R3=0/30；另一次 LQ Direct dP8=20/30、dP16=24/30。这是两个不同实验，不混合成绩。[精简逐样本投影与来源SHA](../../results/server29/20261101/recomputed_projections/COUNT_RECONCILIATION.json)。投影省去长extraction审计字段，明确不是原score文件的完整替代，原文件是否上传以机器清单为准。

Qwen/GDN与Ling/KDA、INT8 R128/C128和G32/G64/G128、sym/asym、不同seed/context、PACE和Direct Read不合并成绩。量化反量化数值语义不自动等于物理packed INT8常驻。Smoke、CPU门禁、失败、停止、仅计划和未完成材料均保留原scope。

## 来源包索引

| 来源包/研究目录 | 模型原文件声明 | 原状态范围 | 文件数 | 本次归档映射 |
|---|---|---|---:|---|
| `experiment_runs/LING_KDA_L7_OBJECTIVE_REDUNDANCY_AND_PERSISTENT_VALIDATION_V1` | UNKNOWN | UNKNOWN | 335 | [来源及关键文件](../../results/server29/20261101/sources/0b16fb2e2299/README.md) |
| `experiment_runs/LING_KDA_LONG_HORIZON_DIRECTION_CAUSAL_V1` | UNKNOWN | COMPLETED_REPORTED | 280 | [来源及关键文件](../../results/server29/20261101/sources/31d9ede8a6ec/README.md) |
| `experiment_runs/LING_KDA_P1_L7_3090_24GB_RESOURCE_CLOSURE_V5` | UNKNOWN | UNKNOWN | 0 | [来源及关键文件](../../results/server29/20261101/sources/06fac05183eb/README.md) |
| `experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1` | UNKNOWN | UNKNOWN | 72 | [来源及关键文件](../../results/server29/20261101/sources/1da5d65b6879/README.md) |
| `experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V1` | UNKNOWN | UNKNOWN | 87 | [来源及关键文件](../../results/server29/20261101/sources/01aac07e51d8/README.md) |
| `experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2` | UNKNOWN | UNKNOWN | 156 | [来源及关键文件](../../results/server29/20261101/sources/be766b055921/README.md) |
| `experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1` | UNKNOWN | UNKNOWN | 569 | [来源及关键文件](../../results/server29/20261101/sources/a0d601191438/README.md) |
| `experiment_runs/LING_KDA_P1_L7_EFFICIENT_RECOVERY_V4` | UNKNOWN | UNKNOWN | 336 | [来源及关键文件](../../results/server29/20261101/sources/df9f146be8cf/README.md) |
| `experiment_runs/LING_KDA_P1_L7_FAST_RECOVERY_AND_FORMAL_32K_V3` | UNKNOWN | UNKNOWN | 179 | [来源及关键文件](../../results/server29/20261101/sources/6ff09a7eb37b/README.md) |
| `experiment_runs/LING_KDA_P1_L7_FULL_PERSISTENT_PROSPECTIVE_V2` | UNKNOWN | UNKNOWN | 71 | [来源及关键文件](../../results/server29/20261101/sources/88d52f4024ed/README.md) |
| `experiment_runs/LING_KDA_P1_L7_FUNCTIONAL_VALIDATION_V1` | UNKNOWN | UNKNOWN | 899 | [来源及关键文件](../../results/server29/20261101/sources/6d85a9f81566/README.md) |
| `experiment_runs/LING_KDA_P1_L7_MIXED_OUTCOME_RESOLUTION_V1` | UNKNOWN | UNKNOWN | 297 | [来源及关键文件](../../results/server29/20261101/sources/f8369210b8e6/README.md) |
| `experiment_runs/LING_KDA_P1_L7_PHASEB_STOP_FORENSIC_AUDIT_V1` | UNKNOWN | UNKNOWN | 12 | [来源及关键文件](../../results/server29/20261101/sources/4d2cb1a31ec9/README.md) |
| `experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1` | UNKNOWN | UNKNOWN | 1229 | [来源及关键文件](../../results/server29/20261101/sources/fa9c6cd7bcab/README.md) |
| `experiment_runs/LING_KDA_PERSISTENT_SURROGATE_PROSPECTIVE_VALIDATION_V1` | UNKNOWN | UNKNOWN | 110 | [来源及关键文件](../../results/server29/20261101/sources/c660759fa58b/README.md) |
| `experiment_runs/LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1` | UNKNOWN | UNKNOWN | 213 | [来源及关键文件](../../results/server29/20261101/sources/c1a541146256/README.md) |
| `experiment_runs/LING_KDA_ROTATION_DUAL_AXIS_STABLE_RECOVERY_V1` | UNKNOWN | UNKNOWN | 6 | [来源及关键文件](../../results/server29/20261101/sources/f84ca18a53d3/README.md) |
| `experiment_runs/LING_KDA_SHORT_HORIZON_SURROGATE_VALIDATION_V1` | UNKNOWN | UNKNOWN | 13 | [来源及关键文件](../../results/server29/20261101/sources/c90cfd7f1012/README.md) |
| `experiment_runs/LING_KDA_SHORT_HORIZON_SURROGATE_VALIDATION_V1__resume20261010_v2` | UNKNOWN | UNKNOWN | 106 | [来源及关键文件](../../results/server29/20261101/sources/d0fde041571b/README.md) |
| `experiment_runs/LING_OBSERVABILITY_ATTEMPT05_CROSS_RUNTIME_ELIGIBILITY_AUDIT_V1` | UNKNOWN | UNKNOWN | 7 | [来源及关键文件](../../results/server29/20261101/sources/7cf9a76e8dbd/README.md) |
| `experiment_runs/LING_OBSERVABILITY_MECHANISM_TRIAGE_V1` | UNKNOWN | UNKNOWN | 17 | [来源及关键文件](../../results/server29/20261101/sources/257dc12fb929/README.md) |
| `experiment_runs/ling-kda-dual-axis-resume-01` | UNKNOWN | UNKNOWN | 160 | [来源及关键文件](../../results/server29/20261101/sources/8018e985ba09/README.md) |
| `experiment_runs/ling_attempt05_qualification_logs` | UNKNOWN | UNKNOWN | 1 | [来源及关键文件](../../results/server29/20261101/sources/b5a78d783a05/README.md) |
| `experiment_runs/ling_historical_runtime_rebase_v1` | UNKNOWN | UNKNOWN | 92 | [来源及关键文件](../../results/server29/20261101/sources/6e5588295e08/README.md) |
| `experiment_runs/ling_historical_runtime_replay_v1` | UNKNOWN | UNKNOWN | 667 | [来源及关键文件](../../results/server29/20261101/sources/10a0f0052c07/README.md) |
| `experiment_runs/ling_observability_gpu3_references_v1` | UNKNOWN | UNKNOWN | 130 | [来源及关键文件](../../results/server29/20261101/sources/dbe74a975a05/README.md) |
| `experiment_runs/ling_observability_scientific_repro_v1` | UNKNOWN | UNKNOWN | 7 | [来源及关键文件](../../results/server29/20261101/sources/9b980c98c4e8/README.md) |
| `experiment_runs/ling_observability_scientific_repro_v2` | UNKNOWN | UNKNOWN | 13 | [来源及关键文件](../../results/server29/20261101/sources/afd4eed722fc/README.md) |
| `experiment_runs/ling_observability_scientific_repro_v3` | UNKNOWN | UNKNOWN | 14 | [来源及关键文件](../../results/server29/20261101/sources/406d4f8d1514/README.md) |
| `experiments/LING_KDA_GROUP32_FOUR_CODEC_AIME26_256K_SEED1_V1` | UNKNOWN | UNKNOWN | 72 | [来源及关键文件](../../results/server29/20261101/sources/bfdc50c10b64/README.md) |
| `experiments/LING_L6_L7_CANONICAL_60_CONFIRMATORY_V1` | Ling-3.0-tiny | UNKNOWN | 322 | [来源及关键文件](../../results/server29/20261101/sources/6311a091757c/README.md) |
| `experiments/LING_L6_L7_PAIRED_TRAJECTORY_CLOSURE_V1` | UNKNOWN | UNKNOWN | 4 | [来源及关键文件](../../results/server29/20261101/sources/dbd9fefeba21/README.md) |
| `experiments/LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V1` | UNKNOWN | UNKNOWN | 1 | [来源及关键文件](../../results/server29/20261101/sources/59fe9449f725/README.md) |
| `experiments/LING_RECURRENT_DENSE_L6_L7_V1` | Ling-3.0-tiny | UNKNOWN | 1333 | [来源及关键文件](../../results/server29/20261101/sources/51bd196c9df5/README.md) |
| `experiments/LING_RECURRENT_MEMORY_LIFETIME_AUDIT_V2` | UNKNOWN | UNKNOWN | 18 | [来源及关键文件](../../results/server29/20261101/sources/e4be55ad291f/README.md) |
| `experiments/LING_SGLANG_DENSE_AIME26_LAST20_256K_SINGLESEED_V1` | Ling-3.0-tiny | UNKNOWN | 7 | [来源及关键文件](../../results/server29/20261101/sources/723e1b2337b9/README.md) |
| `experiments/ling-four-codec-aime26-cpu-abi-repair-01` | UNKNOWN | UNKNOWN | 8 | [来源及关键文件](../../results/server29/20261101/sources/fe7dd31940f6/README.md) |
| `experiments/ling-four-codec-aime26-qualification-control-recovery-01` | UNKNOWN | UNKNOWN | 156 | [来源及关键文件](../../results/server29/20261101/sources/60b9d5c3220d/README.md) |
| `experiments/ling-four-codec-aime26-qualification-environment-recovery-02` | UNKNOWN | UNKNOWN | 313 | [来源及关键文件](../../results/server29/20261101/sources/57c9c39fe7e5/README.md) |
| `experiments/ling-four-codec-aime26-real-model-qualification-01` | UNKNOWN | UNKNOWN | 83 | [来源及关键文件](../../results/server29/20261101/sources/ae9d99479762/README.md) |
| `experiments/ling-four-codec-aime26-real-model-qualification-02` | UNKNOWN | UNKNOWN | 82 | [来源及关键文件](../../results/server29/20261101/sources/3ae996ee4940/README.md) |
| `experiments/ling-four-codec-aime26-real-model-qualification-04` | UNKNOWN | UNKNOWN | 53 | [来源及关键文件](../../results/server29/20261101/sources/701431dec9e3/README.md) |
| `experiments/qwen35_gdn_quant` | /data/zypan/modelscope_models/Qwen3.5-9B | UNKNOWN | 1 | [来源及关键文件](../../results/server29/20261101/sources/a8280a808e4a/README.md) |
| `analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1` | /data/zypan/modelscope_models/Qwen3.5-9B; Qwen3.5-9B | UNKNOWN | 59945 | [来源及关键文件](../../results/server29/20261101/sources/46d46cb479d2/README.md) |
| `worktrees/hadamard-init-dense-oracle-v1-ling-3090-v2` | Ling-3.0-tiny/KDA | UNKNOWN | 1090 | [来源及关键文件](../../results/server29/20261101/sources/0265298145de/README.md) |
| `worktrees/ling-256k-length-sensitivity-v1` | /data/zypan/modelscope_models/Qwen3.5-9B | UNKNOWN | 1090 | [来源及关键文件](../../results/server29/20261101/sources/4dc30f19487c/README.md) |
| `worktrees/ling-error-coherence-diagnosis-v1` | UNKNOWN | UNKNOWN | 502 | [来源及关键文件](../../results/server29/20261101/sources/aa00e2b30b0e/README.md) |
| `worktrees/ling-kda-4doc-cleanroom-final-v1` | UNKNOWN | UNKNOWN | 139 | [来源及关键文件](../../results/server29/20261101/sources/d7bffd39695b/README.md) |
| `worktrees/ling-kda-4doc-cleanroom-gpu3-single-v1` | UNKNOWN | UNKNOWN | 140 | [来源及关键文件](../../results/server29/20261101/sources/09dd1e65e87f/README.md) |
| `worktrees/ling-kda-4doc-cleanroom-gpu3-single-v2` | UNKNOWN | UNKNOWN | 141 | [来源及关键文件](../../results/server29/20261101/sources/a61704de97ee/README.md) |
| `worktrees/ling-kda-4doc-cleanroom-gpu6-single-v1` | UNKNOWN | UNKNOWN | 143 | [来源及关键文件](../../results/server29/20261101/sources/6dded2920526/README.md) |
| `worktrees/ling-kda-p1-l7-3090-resource-closure-v5` | UNKNOWN | UNKNOWN | 98 | [来源及关键文件](../../results/server29/20261101/sources/305bf001fd31/README.md) |
| `worktrees/ling-kda-p1-l7-efficient-recovery-v4` | UNKNOWN | UNKNOWN | 99 | [来源及关键文件](../../results/server29/20261101/sources/0ed93fe19354/README.md) |
| `worktrees/ling-kda-p1-l7-fast-recovery-and-formal-32k-v3` | UNKNOWN | UNKNOWN | 79 | [来源及关键文件](../../results/server29/20261101/sources/a0d38803bc50/README.md) |
| `worktrees/ling-kda-p1-l7-full-persistent-prospective-v2` | UNKNOWN | UNKNOWN | 67 | [来源及关键文件](../../results/server29/20261101/sources/48ce00de7695/README.md) |
| `worktrees/ling-kda-p1-l7-mixed-outcome-resolution-v1` | UNKNOWN | UNKNOWN | 105 | [来源及关键文件](../../results/server29/20261101/sources/49c12f777652/README.md) |
| `worktrees/ling-kda-p1-l7-prospective-extension-4doc-v1` | UNKNOWN | UNKNOWN | 109 | [来源及关键文件](../../results/server29/20261101/sources/2b0a2721bbdb/README.md) |
| `worktrees/ling-kda-prospective-engineering-recovery` | UNKNOWN | UNKNOWN | 117 | [来源及关键文件](../../results/server29/20261101/sources/9f0c73ac0cb3/README.md) |
| `worktrees/ling-kda-prospective-execution04` | UNKNOWN | UNKNOWN | 122 | [来源及关键文件](../../results/server29/20261101/sources/36d6929eeedb/README.md) |
| `worktrees/ling-kda-prospective-execution05` | UNKNOWN | UNKNOWN | 132 | [来源及关键文件](../../results/server29/20261101/sources/35474a5a185e/README.md) |
| `worktrees/ling-kda-prospective-qualified-completion` | UNKNOWN | UNKNOWN | 119 | [来源及关键文件](../../results/server29/20261101/sources/1828edbb78db/README.md) |
| `worktrees/ling-kda-prospective-resume-repair-05` | UNKNOWN | UNKNOWN | 127 | [来源及关键文件](../../results/server29/20261101/sources/5977ea94a175/README.md) |
| `worktrees/ling-kda-rotation-dual-axis-functional-resume-01` | UNKNOWN | UNKNOWN | 164 | [来源及关键文件](../../results/server29/20261101/sources/10d7021d22e7/README.md) |
| `worktrees/ling-kda-rotation-dual-axis-functional-v1` | UNKNOWN | UNKNOWN | 163 | [来源及关键文件](../../results/server29/20261101/sources/52ca1270d506/README.md) |
| `worktrees/ling-kda-rotation-dual-axis-stable-recovery-v1` | UNKNOWN | UNKNOWN | 164 | [来源及关键文件](../../results/server29/20261101/sources/6b145d02b737/README.md) |
| `worktrees/ling-long-horizon-observability-bridge-v1` | UNKNOWN | UNKNOWN | 689 | [来源及关键文件](../../results/server29/20261101/sources/f73309d4f2b5/README.md) |
| `worktrees/ling-long-horizon-observability-bridge-v3` | UNKNOWN | UNKNOWN | 34703 | [来源及关键文件](../../results/server29/20261101/sources/b33dca2daa82/README.md) |
| `worktrees/ling-long-horizon-observability-bridge-v3-authority` | UNKNOWN | UNKNOWN | 20492 | [来源及关键文件](../../results/server29/20261101/sources/3e4274f7fb36/README.md) |
| `worktrees/ling-long-horizon-propagation-confirm-v1` | UNKNOWN | UNKNOWN | 581 | [来源及关键文件](../../results/server29/20261101/sources/02874510a195/README.md) |
| `worktrees/ling-observability-execution-capture-engineering-tests-v1` | UNKNOWN | UNKNOWN | 180 | [来源及关键文件](../../results/server29/20261101/sources/7da85016cac2/README.md) |
| `worktrees/ling-observability-gpu3-expansion-v1` | UNKNOWN | UNKNOWN | 370 | [来源及关键文件](../../results/server29/20261101/sources/2bd4252d5424/README.md) |
| `worktrees/ling-observability-handoff-controller-v2` | UNKNOWN | UNKNOWN | 99 | [来源及关键文件](../../results/server29/20261101/sources/70dbe32858ee/README.md) |
| `worktrees/ling-observability-provenance-amendment-v1` | UNKNOWN | UNKNOWN | 2079 | [来源及关键文件](../../results/server29/20261101/sources/539f0d380d17/README.md) |
| `repos/GDN-quantization` | /data/zypan/models/Ling-3.0-tiny | UNKNOWN | 1118 | [来源及关键文件](../../results/server29/20261101/sources/85cd9df3f0ea/README.md) |
| `ling_handoff_attempt03_upload` | UNKNOWN | UNKNOWN | 2 | [来源及关键文件](../../results/server29/20261101/sources/3e8c54721efc/README.md) |
| `ling_monitoring_v1` | UNKNOWN | UNKNOWN | 7 | [来源及关键文件](../../results/server29/20261101/sources/221a08eca279/README.md) |
| `qwen_asym_int4_group32_aime26_archive` | /data/zypan/modelscope_models/Qwen3.5-9B; Qwen3.5-9B | UNKNOWN | 1975 | [来源及关键文件](../../results/server29/20261101/sources/065773c87578/README.md) |
| `qwen_c6r_dual_m_interaction_archive` | UNKNOWN | UNKNOWN | 561 | [来源及关键文件](../../results/server29/20261101/sources/6834606783aa/README.md) |
| `qwen_dual_axis_replication_archive` | Qwen3.5-9B | UNKNOWN | 487 | [来源及关键文件](../../results/server29/20261101/sources/a2daf85c25bc/README.md) |
| `qwen_fast_runtime_qualification_archive` | /data/zypan/modelscope_models/Qwen3.5-9B; Qwen3.5-9B | FAILED_OR_REJECTED | 7460 | [来源及关键文件](../../results/server29/20261101/sources/6b0b33710da6/README.md) |
| `qwen_functional_replication_archive` | UNKNOWN | UNKNOWN | 921 | [来源及关键文件](../../results/server29/20261101/sources/b90e8f56d9dd/README.md) |
| `qwen_gdn_pace48_archive` | Qwen3.5-9B | COMPLETED_REPORTED | 43159 | [来源及关键文件](../../results/server29/20261101/sources/696fc6ba62e7/README.md) |
| `qwen_gdn_v2_compute` | UNKNOWN | UNKNOWN | 291 | [来源及关键文件](../../results/server29/20261101/sources/b34a21b42892/README.md) |
| `qwen_gdn_wscale_rowbal_archive` | UNKNOWN | UNKNOWN | 39029 | [来源及关键文件](../../results/server29/20261101/sources/45216d7e0c7c/README.md) |
| `qwen_group32_four_codec_aime26_archive` | /data/zypan/modelscope_models/Qwen3.5-9B; Qwen3.5-9B | UNKNOWN | 2422 | [来源及关键文件](../../results/server29/20261101/sources/420ad4d05511/README.md) |
| `qwen_group64_four_codec_aime26_archive` | /data/zypan/modelscope_models/Qwen3.5-9B; Qwen3.5-9B | UNKNOWN | 54695 | [来源及关键文件](../../results/server29/20261101/sources/6c07583b9b85/README.md) |
| `qwen_historical_port_int8_group32_qualification_archive` | /data/zypan/modelscope_models/Qwen3.5-9B; NOT_RUN; Qwen3.5-9B | PARTIAL | 12917 | [来源及关键文件](../../results/server29/20261101/sources/7bf3635b1fb2/README.md) |
| `qwen_historical_runtime_gpu_recovery_archive` | /data/zypan/modelscope_models/Qwen3.5-9B; Qwen3.5-9B | UNKNOWN | 9447 | [来源及关键文件](../../results/server29/20261101/sources/b610b3a71d60/README.md) |
| `qwen_historical_runtime_restoration_archive` | /data/zypan/modelscope_models/Qwen3.5-9B; Qwen3.5-9B | UNKNOWN | 576 | [来源及关键文件](../../results/server29/20261101/sources/94398bea4a7d/README.md) |
| `qwen_oscar_p1_remainder_archive` | /data/zypan/modelscope_models/Qwen3.5-9B; Qwen3.5-9B | UNKNOWN | 756 | [来源及关键文件](../../results/server29/20261101/sources/6aeb169241ac/README.md) |
| `qwen_oscar_resident_rotation_efficient_qualification_archive` | UNKNOWN | FAILED_OR_REJECTED | 32335 | [来源及关键文件](../../results/server29/20261101/sources/778675df03bf/README.md) |
| `qwen_oscar_sym32_aime26_archive` | /data/zypan/modelscope_models/Qwen3.5-9B; Qwen3.5-9B | UNKNOWN | 582 | [来源及关键文件](../../results/server29/20261101/sources/3ea48225b634/README.md) |
| `qwen_oscar_sym32_c1_s3_v3_archive` | UNKNOWN | UNKNOWN | 142 | [来源及关键文件](../../results/server29/20261101/sources/973467cea98a/README.md) |
| `qwen_oscar_sym32_unified_v2_archive` | /data/zypan/modelscope_models/Qwen3.5-9B; Qwen3.5-9B | UNKNOWN | 1933 | [来源及关键文件](../../results/server29/20261101/sources/fe5dc7b6b005/README.md) |
| `research_archive` | UNKNOWN | UNKNOWN | 442 | [来源及关键文件](../../results/server29/20261101/sources/032319bb2e7e/README.md) |
| `consolidation` | UNKNOWN | UNKNOWN | 111 | [来源及关键文件](../../results/server29/20261101/sources/ebc9b532930c/README.md) |
| `results` | UNKNOWN | UNKNOWN | 12 | [来源及关键文件](../../results/server29/20261101/sources/c099142bc318/README.md) |
| `diagnostic_inputs` | Ling-3.0-tiny | UNKNOWN | 590 | [来源及关键文件](../../results/server29/20261101/sources/d8f9f0bde09c/README.md) |
| `cleanroom` | /data/zypan/modelscope_models/Qwen3.5-9B; Ling-3.0-tiny; Ling-3.0-tiny/KDA | UNKNOWN | 1003 | [来源及关键文件](../../results/server29/20261101/sources/d25d985650eb/README.md) |
| `remote_source` | UNKNOWN | UNKNOWN | 146 | [来源及关键文件](../../results/server29/20261101/sources/60170e702718/README.md) |
| `src` | UNKNOWN | UNKNOWN | 12431 | [来源及关键文件](../../results/server29/20261101/sources/25a6634263c1/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/GDN_KDA_ASYM_INT8_CPU_FEASIBILITY_V1` | UNKNOWN | UNKNOWN | 33 | [来源及关键文件](../../results/server29/20261101/sources/7ca826e52ad2/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/pace48_leap_unroll_dp8_dp16_v1` | UNKNOWN | UNKNOWN | 1073 | [来源及关键文件](../../results/server29/20261101/sources/b32c0487e0a7/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/pace48_lqunroll_dp32_throughput_v1` | UNKNOWN | STOPPED | 485 | [来源及关键文件](../../results/server29/20261101/sources/8a233269feb4/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/pace_manual_partial_score_20261026_current20_v1` | UNKNOWN | UNKNOWN | 10 | [来源及关键文件](../../results/server29/20261101/sources/d7471675a475/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/pace_manual_partial_score_20261027_lifecycle_v1` | UNKNOWN | UNKNOWN | 11 | [来源及关键文件](../../results/server29/20261101/sources/2957cdcf3644/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_aime26_stepquant6_vs_h_dualm_2seed_v1` | UNKNOWN | FAILED_OR_REJECTED | 962 | [来源及关键文件](../../results/server29/20261101/sources/b63b8901f5cd/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_asym_int4_group32_aime26_seed1_v1` | /data/zypan/modelscope_models/Qwen3.5-9B; Qwen3.5-9B | UNKNOWN | 1976 | [来源及关键文件](../../results/server29/20261101/sources/214c4cfa890d/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_c6r_dual_m_continuation_20261017` | UNKNOWN | UNKNOWN | 232 | [来源及关键文件](../../results/server29/20261101/sources/cd8629338408/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_c6r_dual_m_interaction_reuse_5doc_v1` | UNKNOWN | UNKNOWN | 311 | [来源及关键文件](../../results/server29/20261101/sources/d55ca4b4a84b/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_dual_axis_prospective_replication_5doc_v1` | Qwen3.5-9B | UNKNOWN | 401 | [来源及关键文件](../../results/server29/20261101/sources/e19fa5528a35/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_dual_m_equivalent_speedup_v1` | UNKNOWN | UNKNOWN | 129 | [来源及关键文件](../../results/server29/20261101/sources/f627dadd0355/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_dualm_aware_rotation_training_readiness_v1` | UNKNOWN | UNKNOWN | 377 | [来源及关键文件](../../results/server29/20261101/sources/558ecd595230/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_fast_runtime_qualification_v1` | /data/zypan/modelscope_models/Qwen3.5-9B; NOT_RUN; Qwen3.5-9B | FAILED_OR_REJECTED | 9844 | [来源及关键文件](../../results/server29/20261101/sources/ab63e4020f10/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_group32_four_codec_aime26_seed1_v1` | /data/zypan/modelscope_models/Qwen3.5-9B; Qwen3.5-9B | UNKNOWN | 2423 | [来源及关键文件](../../results/server29/20261101/sources/94742d2ab8e7/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1` | /data/zypan/modelscope_models/Qwen3.5-9B; Qwen3.5-9B | UNKNOWN | 22110 | [来源及关键文件](../../results/server29/20261101/sources/750ebc135973/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_h_dualm_aime26_throughput_20261017_v1` | UNKNOWN | UNKNOWN | 37 | [来源及关键文件](../../results/server29/20261101/sources/208eb1b8bdba/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_hadamard_dual_m_aime26_formal_60_v1` | UNKNOWN | PARTIAL | 425 | [来源及关键文件](../../results/server29/20261101/sources/0aa50b6af0a4/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_historical_port_int8_group32_qualification_v1` | /data/zypan/modelscope_models/Qwen3.5-9B; NOT_RUN; Qwen3.5-9B | PARTIAL | 21203 | [来源及关键文件](../../results/server29/20261101/sources/fa0fa63337e5/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_historical_runtime_gpu_recovery_v1` | /data/zypan/modelscope_models/Qwen3.5-9B; NOT_RUN; Qwen3.5-9B | COMPLETED_REPORTED | 9956 | [来源及关键文件](../../results/server29/20261101/sources/1a7af22b5ea7/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_historical_runtime_restoration_v1` | /data/zypan/modelscope_models/Qwen3.5-9B; NOT_RUN; Qwen3.5-9B | UNKNOWN | 979 | [来源及关键文件](../../results/server29/20261101/sources/fafb80223600/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_historical_throughput_reconciliation_v1` | UNKNOWN | UNKNOWN | 178 | [来源及关键文件](../../results/server29/20261101/sources/cfb0b549ec26/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_hmmt26_fp_lq8_hlq8_lq16_v1` | UNKNOWN | UNKNOWN | 394 | [来源及关键文件](../../results/server29/20261101/sources/d07f4e11807d/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_oscar_p1_aime26_remainder_gpu0_20261022` | /data/zypan/modelscope_models/Qwen3.5-9B; Qwen3.5-9B | UNKNOWN | 757 | [来源及关键文件](../../results/server29/20261101/sources/4d8bcf4d102c/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_oscar_resident_rotation_efficient_qualification_v1` | UNKNOWN | UNKNOWN | 110 | [来源及关键文件](../../results/server29/20261101/sources/a1c98ab928ba/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_oscar_resident_rotation_efficient_qualification_v1_directory_repair_rerun_20261022` | UNKNOWN | UNKNOWN | 11361 | [来源及关键文件](../../results/server29/20261101/sources/e238766aeee1/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_oscar_resident_rotation_efficient_qualification_v1_directory_repair_rerun_20261022_engineering_stop_recovery_v1` | UNKNOWN | UNKNOWN | 25 | [来源及关键文件](../../results/server29/20261101/sources/10e97b0bc040/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_oscar_resident_rotation_efficient_qualification_v1_engineering_repair_20261022` | UNKNOWN | UNKNOWN | 626 | [来源及关键文件](../../results/server29/20261101/sources/0f45553ceabe/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_oscar_resident_rotation_efficient_qualification_v1_exit_race_repair_20261023` | UNKNOWN | UNKNOWN | 967 | [来源及关键文件](../../results/server29/20261101/sources/e80e991943fc/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_oscar_resident_rotation_efficient_qualification_v1_formal_scope_repair_20261023` | UNKNOWN | UNKNOWN | 2765 | [来源及关键文件](../../results/server29/20261101/sources/57d6212fc673/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_oscar_resident_rotation_efficient_qualification_v1_gpu_continuation_20261022` | UNKNOWN | FAILED_OR_REJECTED | 13071 | [来源及关键文件](../../results/server29/20261101/sources/70a99fe3d4b0/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_oscar_resident_rotation_efficient_qualification_v1_profiler_exit_repair_20261023` | UNKNOWN | UNKNOWN | 1457 | [来源及关键文件](../../results/server29/20261101/sources/e0ddf62e2e84/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_oscar_resident_rotation_efficient_qualification_v1_qualification_continuation_20261023` | UNKNOWN | UNKNOWN | 1968 | [来源及关键文件](../../results/server29/20261101/sources/884a668b2129/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_oscar_sym32_c1_s3_canonical_recovery_aime26_v3` | UNKNOWN | UNKNOWN | 145 | [来源及关键文件](../../results/server29/20261101/sources/be8b9b78bb9d/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_oscar_sym32_p1_c1_s3_aime26_seed1_v1` | /data/zypan/modelscope_models/Qwen3.5-9B; Qwen3.5-9B | UNKNOWN | 583 | [来源及关键文件](../../results/server29/20261101/sources/ebf058ab8c34/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_oscar_sym32_p1_c1_s3_unified_recovery_aime26_v2` | /data/zypan/modelscope_models/Qwen3.5-9B; Qwen3.5-9B | UNKNOWN | 749 | [来源及关键文件](../../results/server29/20261101/sources/8652bd1312d2/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_oscar_sym32_p1_c1_s3_unified_recovery_aime26_v2_rerun_20261022` | /data/zypan/modelscope_models/Qwen3.5-9B; Qwen3.5-9B | UNKNOWN | 1119 | [来源及关键文件](../../results/server29/20261101/sources/ef969eb7cfa5/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_pace48_aime26_seed1_60_v1` | UNKNOWN | UNKNOWN | 324 | [来源及关键文件](../../results/server29/20261101/sources/dde775e3748c/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_4gpu_continue_20261026` | UNKNOWN | UNKNOWN | 3509 | [来源及关键文件](../../results/server29/20261101/sources/f224a39f65d7/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_4gpu_continue_control_v2_20261026` | UNKNOWN | UNKNOWN | 3711 | [来源及关键文件](../../results/server29/20261101/sources/53dbb33d1771/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_4gpu_exit_fix_20261026` | UNKNOWN | UNKNOWN | 3843 | [来源及关键文件](../../results/server29/20261101/sources/6f5cd8bb687a/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_4gpu_lifecycle_repair_20261026` | UNKNOWN | UNKNOWN | 4304 | [来源及关键文件](../../results/server29/20261101/sources/4104f7aed66d/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_4gpu_rebalance_20261027` | UNKNOWN | UNKNOWN | 4440 | [来源及关键文件](../../results/server29/20261101/sources/fe88566e2218/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_scale_exponent_repair_20261025` | UNKNOWN | UNKNOWN | 824 | [来源及关键文件](../../results/server29/20261101/sources/f389dc0bccbb/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_scale_repair_20261025` | UNKNOWN | UNKNOWN | 607 | [来源及关键文件](../../results/server29/20261101/sources/d58952e815dd/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_hadamard_dual_axis_direct_32k_pilot_v1` | UNKNOWN | UNKNOWN | 418 | [来源及关键文件](../../results/server29/20261101/sources/7047559a2d5a/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_hadamard_dual_axis_int8_feasibility_v1` | UNKNOWN | UNKNOWN | 110 | [来源及关键文件](../../results/server29/20261101/sources/1d0bf1e71227/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/analyses/qwen_hadamard_dual_axis_real_state_runtime_gate_v1` | UNKNOWN | UNKNOWN | 213 | [来源及关键文件](../../results/server29/20261101/sources/e69d74703f4d/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/experiment_runs/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3_runtime` | UNKNOWN | UNKNOWN | 16 | [来源及关键文件](../../results/server29/20261101/sources/33775112482d/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/experiment_runs/QWEN_GDN_SHORT_HORIZON_SCALE_INFLATION_SURROGATE_VALIDATION_V1` | UNKNOWN | UNKNOWN | 28 | [来源及关键文件](../../results/server29/20261101/sources/c080c5712bd4/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/experiments/qwen35_gdn_quant` | /data/zypan/modelscope_models/Qwen3.5-9B | UNKNOWN | 83 | [来源及关键文件](../../results/server29/20261101/sources/403d0f8ef2a3/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/aime26-sglang-rotation-v1` | /data/zypan/modelscope_models/Qwen3.5-9B | UNKNOWN | 795 | [来源及关键文件](../../results/server29/20261101/sources/3731d167df20/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/arbitrary-orthogonal-equivalence-gate-v1-qwen` | UNKNOWN | UNKNOWN | 240 | [来源及关键文件](../../results/server29/20261101/sources/1f2dd45eafea/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/hadamard-init-dense-oracle-v1-qwen` | /data/zypan/modelscope_models/Qwen3.5-9B; Qwen3.5-9B; Qwen3.5-9B/GDN | UNKNOWN | 2156 | [来源及关键文件](../../results/server29/20261101/sources/ddcc9e129e53/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/orthogonal-numerical-closure-v1-qwen` | Qwen3.5-9B/GDN | UNKNOWN | 258 | [来源及关键文件](../../results/server29/20261101/sources/5d6325c9bb93/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/qwen-gdn-c6r-c7-4x3090-deployment-v1` | UNKNOWN | UNKNOWN | 255 | [来源及关键文件](../../results/server29/20261101/sources/22ac44778101/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/qwen-gdn-c6r-c7-canonical60-v1` | UNKNOWN | UNKNOWN | 1020 | [来源及关键文件](../../results/server29/20261101/sources/5e0ba29200e5/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/qwen-gdn-c6r-dual-m-interaction-reuse-5doc-v1` | UNKNOWN | UNKNOWN | 91 | [来源及关键文件](../../results/server29/20261101/sources/db26635a5f84/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/qwen-gdn-determinism-causal-isolation-v1` | UNKNOWN | UNKNOWN | 122 | [来源及关键文件](../../results/server29/20261101/sources/ed2d92800f6f/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/qwen-gdn-deterministic-scale-discovery-v3` | UNKNOWN | UNKNOWN | 195 | [来源及关键文件](../../results/server29/20261101/sources/b4a49b5838a6/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/qwen-gdn-distributed-recurrent-training-v2` | UNKNOWN | UNKNOWN | 93 | [来源及关键文件](../../results/server29/20261101/sources/3aeda923e0c8/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/qwen-gdn-dual-m-equivalent-speedup-v1` | UNKNOWN | UNKNOWN | 93 | [来源及关键文件](../../results/server29/20261101/sources/3544b4dbe11c/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/qwen-gdn-dualm-aware-rotation-training-readiness-v1` | UNKNOWN | UNKNOWN | 91 | [来源及关键文件](../../results/server29/20261101/sources/1705c63395db/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/qwen-gdn-fd-path-qdq-interaction-v1` | Qwen3.5-9B/GDN | UNKNOWN | 1299 | [来源及关键文件](../../results/server29/20261101/sources/cd6c0b134597/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/qwen-gdn-fd-scale-feedback-direction-v1` | Qwen3.5-9B/GDN | UNKNOWN | 1358 | [来源及关键文件](../../results/server29/20261101/sources/17861cd79826/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/qwen-gdn-formal-equivalent-mechanism-v2` | Qwen3.5-9B/GDN | UNKNOWN | 945 | [来源及关键文件](../../results/server29/20261101/sources/c7e660ecde4e/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/qwen-gdn-frozen-drive-anchor-mechanism-v1` | Qwen3.5-9B/GDN | UNKNOWN | 1493 | [来源及关键文件](../../results/server29/20261101/sources/a03477b69f04/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/qwen-gdn-frozen-drive-counterfactual-replay-v1` | Qwen3.5-9B/GDN | UNKNOWN | 864 | [来源及关键文件](../../results/server29/20261101/sources/ed0b2b72c662/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/qwen-gdn-fsdp2-output-hook-v1` | Qwen3.5-9B BF16 frozen backbone; canonical Qwen3.5 source | UNKNOWN | 301 | [来源及关键文件](../../results/server29/20261101/sources/24bf1197611e/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/qwen-gdn-fsdp2-reentry-v1` | Qwen3.5-9B BF16 frozen backbone; canonical Qwen3.5 source | UNKNOWN | 296 | [来源及关键文件](../../results/server29/20261101/sources/910cce59099a/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/qwen-gdn-fsdp2-v1` | Qwen3.5-9B BF16 frozen backbone; canonical Qwen3.5 source | UNKNOWN | 190 | [来源及关键文件](../../results/server29/20261101/sources/b8d743ef5abd/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/qwen-gdn-functional-causal-replication-v1` | UNKNOWN | UNKNOWN | 245 | [来源及关键文件](../../results/server29/20261101/sources/865f5876d18a/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/qwen-gdn-propagation-surrogate-validation-v1` | Qwen3.5-9B/GDN | UNKNOWN | 865 | [来源及关键文件](../../results/server29/20261101/sources/b885958b589b/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/qwen-gdn-scale-feedback-fullmodel-functional-causal-pilot-v1` | UNKNOWN | UNKNOWN | 116 | [来源及关键文件](../../results/server29/20261101/sources/5a4d5ecc3988/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/qwen-gdn-scale-surrogate-input-recovery-v2` | UNKNOWN | UNKNOWN | 22 | [来源及关键文件](../../results/server29/20261101/sources/d2b87c42a26a/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/qwen-gdn-static-pp2-v1` | Qwen3.5-9B BF16 frozen backbone; canonical Qwen3.5 source | UNKNOWN | 373 | [来源及关键文件](../../results/server29/20261101/sources/73132306629b/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/worktrees/qwen-gdn-zero3-v1` | Qwen3.5-9B BF16 frozen backbone; canonical Qwen3.5 source | UNKNOWN | 352 | [来源及关键文件](../../results/server29/20261101/sources/0952509525d0/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/research_archive` | UNKNOWN | UNKNOWN | 11 | [来源及关键文件](../../results/server29/20261101/sources/5e41bb6e21c7/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/results` | UNKNOWN | UNKNOWN | 708 | [来源及关键文件](../../results/server29/20261101/sources/9db066e2f135/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/reports` | UNKNOWN | UNKNOWN | 55 | [来源及关键文件](../../results/server29/20261101/sources/dbbcbbe842f3/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/artifacts` | UNKNOWN | UNKNOWN | 1 | [来源及关键文件](../../results/server29/20261101/sources/7771393422bf/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/runs` | /data/zypan/modelscope_models/Qwen3.5-9B | UNKNOWN | 541 | [来源及关键文件](../../results/server29/20261101/sources/81a8d0c5c463/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/qualification` | UNKNOWN | UNKNOWN | 1 | [来源及关键文件](../../results/server29/20261101/sources/907fadd8fcef/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/consolidation` | UNKNOWN | UNKNOWN | 106 | [来源及关键文件](../../results/server29/20261101/sources/f7244dbb1ca6/README.md) |
| `server55_zypan_backup_20261029_87e45424/restored/GDN-quantization` | /data/zypan/modelscope_models/Qwen3.5-9B | UNKNOWN | 816 | [来源及关键文件](../../results/server29/20261101/sources/b08a56fda70a/README.md) |
