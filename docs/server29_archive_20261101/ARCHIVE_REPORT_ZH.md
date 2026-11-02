# .29 实验资产整理与归档报告

## 实际范围和状态

扫描185个明确的研究来源包，共 506,178 个regular文件，元数据体积 1,158,509,384,487 bytes。范围包括 experiment_runs、experiments、worktrees、repos/GDN-quantization、Qwen归档、研究迁移归档，以及 .29 内实际存在的 .55 restored 研究子目录。跳过模型/环境/缓存/凭据目录、不跟随软链接、不扫描其他用户。完整范围见source_packages.json与各来源包索引。

185是来源包数，不宣称185次独立正式实验；来源包状态计数：`{'UNKNOWN': 173, 'COMPLETED_REPORTED': 3, 'FAILED_OR_REJECTED': 5, 'PARTIAL': 3, 'STOPPED': 1}`。无法从原文件确认总体状态的保留UNKNOWN。顶层/data/zypan/GDN-quantization未找到，实际旧clone位于repos/GDN-quantization；其未提交文件和分支均未改变。

扫描范围内存在HMMT26、LQUnroll dP32吞吐等路径标识，但没有找到FUSED_DECODE_OPTIMIZATION或FINAL_FOUR_GPU_PAIRED5_ALL_LENGTHS_V1路径标识。该检查不证明服务器其他位置一定不存在；也不把本机对话或.55上较晚的执行数据当成.29已有数据。明细见EXPECTED_FAMILY_PATH_CHECK.json。

按内容去重后的新server对象 67,291 个，4,079,442,809 bytes；另恢复Canonical60历史分支34个紧凑原文件。具体最终Git暂存文件数/字节在提交前独立验证记录中核对。仓库已存在main/其他分支的关系计数：`{'NONE': 489023, 'IDENTICAL_IN_OTHER_BRANCH': 8285, 'IDENTICAL_IN_MAIN': 8870}`。这是文件来源映射计数，不是独立实验数量。

## 数据与未上传内容

文件处置计数：`{'ALREADY_IN_MAIN': 8734, 'DEDUPLICATED_TO_ARCHIVED_OBJECT': 317209, 'EXCLUDED_BINARY_TRACE_OR_DUPLICATE_PACKAGE': 4391, 'EXCLUDED_ENV_OR_COMPILED': 22298, 'EXCLUDED_GITHUB_100MIB_LIMIT': 4993, 'EXCLUDED_LARGE_RAW_ARTIFACT_BY_COMPACT_POLICY': 657, 'EXCLUDED_LARGE_RESEARCH_METRIC_TRACE_AND_GITHUB_LIMIT': 44, 'EXCLUDED_LARGE_RESEARCH_METRIC_TRACE_BY_COMPACT_POLICY': 144, 'EXCLUDED_LOGS_BY_REPOSITORY_POLICY': 18163, 'EXCLUDED_PERIODIC_TELEMETRY': 309, 'EXCLUDED_RAW_GENERATION_OR_DENSE_TRACE_BY_REPOSITORY_POLICY': 7939, 'EXCLUDED_SECOND_SECURITY_REVIEW': 30, 'EXCLUDED_SENSITIVE_CONTENT': 35, 'EXCLUDED_TEMPORARY': 236, 'EXCLUDED_TENSOR_TRACE_OR_CHECKPOINT_BY_REPOSITORY_POLICY': 40454, 'EXCLUDED_THIRD_PARTY_DATASET': 551, 'EXCLUDED_THIRD_PARTY_DATASET_CONTENT': 341, 'EXCLUDED_THIRD_PARTY_REFERENCE_HTML': 160, 'EXCLUDED_UNCLASSIFIED_ASSET': 7596, 'EXCLUDED_UNCONFIRMED_BINARY': 28, 'EXCLUDED_UNCONFIRMED_PROJECT_CONTEXT': 4469, 'EXCLUDED_UNCONFIRMED_ROTATION_OR_CHECKPOINT': 8, 'UPLOAD_COMPACT_ORIGINAL': 66848, 'UPLOAD_COMPACT_ORIGINAL_PARQUET_METRICS': 238, 'UPLOAD_LOSSLESS_COMPRESSED_TABLE': 57, 'UPLOAD_RESEARCH_FIGURE': 156, 'UPLOAD_SMALL_RESEARCH_DIFF_OR_NOTEBOOK': 31, 'UPLOAD_SMALL_ROTATION_PARAMETER': 59}`。

优先归档原报告、配置、协议、status/gate/manifest、逐样本紧凑结果、paired和误差诊断表、小型相关代码及核验后的小型rotation参数。逐文件原路径、目标、bytes、fresh SHA256、分支commit和解析检查在机器清单中；相同内容只存一次，不按文件名覆盖版本。

大型FP logits、full-cache和state tensor、raw token/layer traces、原始长生成、数据集副本、环境/编译缓存、logs按仓库紧凑证据规则排除；普通Git100MiB上限及更保守文本/rotation策略见ARCHIVE_POLICY.json。未上传文件仍在原服务器，清单不是它们的GitHub备份。没有购买存储或启用新平台/LFS。

另保留 23 张大型原始数值表：原始共 1,158,226,852 bytes，gzip后 351,471,336 bytes。每张压缩文件均解压复核source SHA，实际行数和压缩SHA见LOSSLESS_TABLE_REVIEW.json；这是原始数据的无损保存，不是仅上传表格摘要。

## 来源与质量

main基准commit `2311c9ab633e14897d42a01ba27ae62f33ddbf70`；只读检查了33个远端引用。固定历史分支Canonical60的commit为5a277c0988d74e9b29f8912d9934657a3db25c2d，恢复不是声称这些文件原本在当前.29研究目录完整存在。原报告H35/L6 36/L7 33和FP14/20已按原CSV重算一致，FP44/60仅旧81920协议。

原样复制逐SHA一致；读取前后size/mtime/dev/inode检查，变化/半写文件不作为稳定结果发布。JSON/CSV基本解析，原始格式异常保留并标记，不悄悄修正原报告。敏感文件/内容整份排除，不把凭据或原值写入公开报告。其他来源包的声明仅据原文件标注；未对所有研究重算评分或重新认证科学结论。物理packed/数值QDQ的区别保持原实现scope。

## 复现和使用

入口EXPERIMENT_INDEX.md链接到源包SUMMARY和原样对象。SHA对象名只为去重；原文件路径和语义由清单恢复。没有整个/data/zypan打包，没有执行科研脚本。原.gitignore未改，只对事先核验且有价值的具体归档路径单独加入版本控制。

## Parquet 分类补正（2026-11-02）

初版将Parquet按扩展名排为第三方数据集，这项分类有误。已读取全部380张唯一表的真实schema，确认其为项目指标输出；对合规紧凑表完整解码并检查字符串内容，原样补存 197 个对象、254725674 bytes，覆盖 7617889 行。大型逐token/layer指标表继续依明确的紧凑归档策略保留在服务器，并已纠正清单中的性质和排除原因。详见[逐表结构、来源和哈希](PARQUET_SCHEMA_AND_CONTENT_REVIEW.json)。未重跑模型或评分，未改写原实验结果。
