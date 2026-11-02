# GDN/KDA recurrent-state quantization

研究 Qwen3.5-9B / Gated DeltaNet（GDN）与 Ling-3.0-tiny / KDA 的 **recurrent state 量化、旋转和长期误差传播**。主要路线包括 INT8 分组量化、Key/Value-side Hadamard、可学习正交旋转，以及 PACE mixed INT4/INT8 与 LeapQuant-style Direct Read。

量化对象是 recurrent state；下列实验不表示对模型权重进行了量化。仓库保存研究代码、冻结协议、结果表、失败/停止记录和来源证据，不提供完整预训练模型、运行环境或全量原始生成备份。经核验的小型旋转参数以文件清单为准。

## 从哪里开始

| 目的 | 入口 |
|---|---|
| 浏览本次收集的实验与来源 | [中文实验/来源包索引](docs/server29_archive_20261101/EXPERIMENT_INDEX.md) |
| 按实验名称查找不同版本 | [256 个名称条目](results/server29/20261101/EXPERIMENT_NAME_CATALOG.json) |
| 了解扫描范围、去重、排除和口径差异 | [中文归档报告](docs/server29_archive_20261101/ARCHIVE_REPORT_ZH.md) |
| 找原路径、归档路径、SHA256 和分支来源 | [逐文件清单 JSONL.gz](results/server29/20261101/artifact_index.jsonl.gz) |
| 查看 Ling 256K 完整协议 | [Canonical 256K protocol](docs/LING_CANONICAL_256K_PROTOCOL.md) |
| 查看生成与评分规则 | [Protocol](docs/PROTOCOL.md) · [Scorer protocol](docs/SCORER_PROTOCOL.md) |
| 查看代码与回归测试 | [experiments/](experiments/) · [tests/](tests/) · [依赖清单](requirements.txt) |

入口文档更新于 **2026-11-02**。下面的结果按各自冻结协议列出；归档收集和本次 README 更新没有重新训练、生成或评分。

## 结果摘要：按协议分开阅读

### 1. 历史 AIME26：81,920-token 预算

这是历史基线，不是下方 Ling 256K Canonical60 的匹配对照。`Incorrect` 包含 `Abstain`，每行 `Correct + Incorrect = 60`。

| 模型 | 条件 | Correct | Incorrect | 其中 Abstain | Accuracy |
|---|---|---:|---:|---:|---:|
| Qwen | FP_STATE | 52 | 8 | 3 | 86.67% |
| Qwen | INT8_C128 | 18 | 42 | 39 | 30.00% |
| Qwen | INT8_C128 + Key-Hadamard | 41 | 19 | 6 | 68.33% |
| Ling | FP_STATE | 44 | 16 | 11 | 73.33% |
| Ling | INT8_R128 | 23 | 37 | 28 | 38.33% |
| Ling | INT8_R128 + Value-Hadamard | 28 | 32 | 28 | 46.67% |

历史 INT8 主线采用 Qwen C128 + Key-side Hadamard，以及 Ling R128 + Value-side Hadamard。这些是在对应条件下测试的配置，不代表跨模型、精度和预算的全局最优。

### 2. Ling/KDA：256K Canonical60

[2026-09-30 最终报告](results/server29/20261101/historical_branch/LING_L6_L7_CANONICAL_60_CONFIRMATORY_V1/reports/FINAL_REPORT.md) 已包含 H/L6/L7 的 60 样本结果（seed 1/2 各 30）。此前 README 中 L6 running/partial、L7 pending 的描述属于较早快照，不再作为这组评测的状态。

| 条件 | Correct / available | Accuracy | 范围 |
|---|---:|---:|---|
| Hadamard（H） | 35/60 | 58.33% | Canonical60 |
| L6 | 36/60 | 60.00% | Canonical60 |
| L7 | 33/60 | 55.00% | Canonical60 |
| FP | **14/20** | 70.00% | 仅可用参考子集，未完成匹配的 60 样本 FP 对照 |

**历史 FP44/60 不能代替此处的 FP 对照。** L6 相对 H 多 1 题、L7 少 2 题；原报告将 L6 解释为 supportive but underpowered、L7 为 negative or mixed，不能据此宣称学习旋转具有普遍显著优势。

证据：[原始逐样本 CSV](results/server29/20261101/historical_branch/LING_L6_L7_CANONICAL_60_CONFIRMATORY_V1/analysis/canonical60_outcomes.csv) · [数量复核与分支/commit 来源](docs/server29_archive_20261101/CANONICAL60_COUNT_RECONCILIATION.json)。这些文件从固定历史分支恢复，来源与当前服务器目录中的文件分别记录。

该协议的预算为：

```text
context_length = 262144
max_total_tokens = 262144
max_new_tokens = 262144 - prompt_tokens - 512
```

256K 指总上下文预算，生成预算随 prompt 长度变化。较早的 seed-1 256K precursor（FP 21/30、INT8_R128 9/30、加 Value-Hadamard 17/30）也不能与本组 Canonical60 合并计数；完整定义见 [冻结协议](docs/LING_CANONICAL_256K_PROTOCOL.md)。

### 3. Qwen/GDN：PACE 与 LQ Direct Read

以下是两次独立实验。归档时只读复核了已有评分文件中的 `correct` 字段，没有重新解码评分。

| 实验 | 条件 | Correct / total |
|---|---|---:|
| 原 PACE FP16 exponent | R2 | 21/30 |
| 原 PACE FP16 exponent | R3 | 0/30 |
| LQ Direct dP8/dP16 | dP8 | 20/30 |
| LQ Direct dP8/dP16 | dP16 | 24/30 |

来源：[计数与原文件 SHA256](docs/server29_archive_20261101/PACE_COUNT_RECONCILIATION.json) · [逐样本标量投影](results/server29/20261101/recomputed_projections/)。投影省略长 extraction/rationale 字段，不能冒充原评分文件的完整副本；原文件的实际归档状态见逐文件清单。

不要混合这两套成绩，也不要把它们当作匹配的吞吐对照。吞吐比较还必须核对硬件、输入、序列长度、warmup、计时区间、重复次数及 profiler 是否开启；固定工作量 TPS、真实生成 TPS 和多卡聚合吞吐是不同口径。

## 2026-11-01 来源归档（2026-11-02 补正）

归档快照已进入 `main`：[固定提交 `15114c7`](https://github.com/pytenter/GDN-quantization/commit/15114c7f126d02f96b8f1fb27959fccd17f6d570)。以下数量描述该快照的归档范围，**不是当前仓库总文件数或独立正式实验数**。

| 项目 | 已核验范围 |
|---|---|
| 服务器扫描 | 506,178 个 regular 文件；185 个来源包 |
| 名称检索 | 256 个名称条目；同名快照、worktree、seed 和协议不自动合并 |
| 归档文件 | 67,929 个文件，逻辑体积 4,524,489,152 bytes（约 4.52 GB） |
| 大型 CSV/TSV | 23 张表，无损 gzip；3,947,996 行，解压后 SHA256 与源文件一致 |
| 紧凑 Parquet 指标 | 197 张原始表；7,617,889 行，完整解码检查后原样保存 |
| 历史分支恢复 | Canonical60 的 34 个紧凑原文件；保留原分支和 commit |

Parquet 的初版按扩展名分类有误，现已按真实 schema 和内容补正：这些是项目指标表，不应一概视为第三方数据集。较大的逐 token/layer 指标表仍按明确的紧凑归档策略保留原地。[逐表核验与排除原因](docs/server29_archive_20261101/PARQUET_SCHEMA_AND_CONTENT_REVIEW.json) · [无损压缩表清单](results/server29/20261101/lossless_compressed_tables/TABLES.json)。

来源包的 `UNKNOWN` 表示未确认其包级整体状态；一个包可能含多个阶段或运行版本，不等于实验失败。门禁 PASS、smoke、计划、CPU 验证和正式 endpoint 在索引中保留各自范围。

本次扫描范围内没有找到 `FUSED_DECODE_OPTIMIZATION` 或 `FINAL_FOUR_GPU_PAIRED5_ALL_LENGTHS_V1` 路径标识，因此不把本机记录当成 .29 已有工件。此结论不证明服务器其他位置一定不存在相应文件。

## 如何定位原始数据与校验

1. 从[实验索引](docs/server29_archive_20261101/EXPERIMENT_INDEX.md)或[名称目录](results/server29/20261101/EXPERIMENT_NAME_CATALOG.json)进入对应来源包。包内 `README.md` 提供关键文件链接，`SUMMARY.json` 保留原状态声明和来源信息。
2. 在 `artifacts.jsonl.gz` 或全局 `artifact_index.jsonl.gz` 查找原路径。重点查看 `original_relative_path`、`archive_path`、`upload_status`、`sha256` 和 `github_sources`；以记录实际字段为准。
3. `objects/` 按内容 SHA256 去重存储，原样对象与源文件字节一致。原目录布局没有完整复制，原报告内部的相对链接可能需要按清单还原。
4. 对 gzip 数值表，区分源文件 SHA256 与 `archive_sha256`；解压后再与源 SHA256 比较。对于标量投影等派生文件，按其明确标注的范围使用。

下载机器清单后，可用 Python 标准库流式检索，无需运行模型：

```python
import gzip
import json

needle = "LING_L6_L7_CANONICAL_60"
with gzip.open("artifact_index.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        origin = row.get("original_relative_path", row.get("original_path", ""))
        if needle in origin or needle in row.get("experiment_id", ""):
            print(origin, row.get("archive_path"), row.get("upload_status"), row.get("sha256"))
```

校验记录：[发布摘要](docs/server29_archive_20261101/PUBLICATION_SUMMARY.json) · [固定快照文件哈希清单](https://github.com/pytenter/GDN-quantization/blob/15114c7f126d02f96b8f1fb27959fccd17f6d570/docs/server29_archive_20261101/UPLOAD_MANIFEST.sha256)。校验应使用清单所属的固定 commit；清单包含当时的 README，后续导航文档更新不会改写该历史快照或其科学工件。

## 研究解释与归档边界

- 不跨模型、seed、样本集合、上下文预算、量化配置或旋转实现直接合并成绩。paired comparison 必须使用对应原始样本与协议。
- 量化—反量化的数值语义不自动证明物理 packed INT8 常驻；应核对对应代码、buffer 和运行证据。Parquet 指标列的 dtype 也不代表模型 state 的实际存储 dtype。
- 早期 Dense-State / Dense-Functional 的 `FP_STATE_RESET_EACH_TOKEN` 每步回到 FP teacher state；它与真实多步低比特 state 写回不同，不能用作后者训练效果的直接证明。
- Ling Value-Hadamard 的 corrected prefill-endpoint V2 将 kernel 返回的已旋转 Value-basis recurrent state 直接写入 cache，不再额外旋转 endpoint；不同实现版本需保留身份边界。
- 失败、停止、负结果和原始解析警告均保留并标注。归档收集不是对每个实验结论的独立再认证；训练完成或局部门禁通过也不等于正式评测完成。
- 密码、token、私钥、完整模型、环境缓存和无必要的第三方原始数据集不进入此归档。[未上传文件清单](results/server29/20261101/excluded_artifacts.jsonl.gz)记录可得的路径、大小、SHA256 与原因；[扫描边界](docs/server29_archive_20261101/SKIPPED_SCAN_BOUNDARIES.json)说明未进入的范围。**清单不等于原始大文件备份。**

## 目录与历史文档

```text
experiments/                      研究代码与实验协议
results/                          冻结结果和离线分析
results/server29/20261101/
  objects/                        原样、按 SHA256 去重的研究文件
  sources/                        各来源包的入口、摘要和逐文件清单
  historical_branch/              固定历史分支恢复的证据
  recomputed_projections/          已有评分字段的标量投影及数量复核
  lossless_compressed_tables/      无损压缩的原始数值表
reports/                          科学报告与整理审计
docs/                             协议、交接与来源文档
docs/server29_archive_20261101/    本次归档报告和核验记录
tests/                            评分器与旋转相关回归测试
```

[2026-09-26 交接](docs/HANDOFF_2026-09-26.md)、[当时的研究状态](docs/RESEARCH_STATUS_2026-09-26.md)、[原实验索引](docs/EXPERIMENT_INDEX.md)、[工件矩阵](docs/HANDOFF_ARTIFACT_MATRIX.md)及[服务器来源](docs/SERVER_PROVENANCE.md)保留历史上下文。其中 running、pending 或 NOT EXECUTED 仅适用于文档对应时间和范围，后续结果应查看各实验最终报告。

各快照可能依赖不同环境、模型和冻结资产，不能将集中归档视为一套可直接统一启动的评测工程。复现前请阅读对应实验的 protocol、源码身份和环境记录。

## 使用与引用

请先阅读 [LICENSE](LICENSE)。引用信息见 [CITATION.cff](CITATION.cff)；引用结果时同时注明实验 ID、配置、样本集合、协议及固定 commit。
