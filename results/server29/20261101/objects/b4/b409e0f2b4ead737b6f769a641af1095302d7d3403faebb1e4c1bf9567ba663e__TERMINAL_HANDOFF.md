本次科学终态为 `HISTORICAL_PORT_FP_QUALIFIED_SPEED_RECOVERED`。完整十问报告为 `FINAL_SCIENTIFIC_REPORT.md`，SHA256 `00c24c462a03bf6b6d26aaa846df254eb4fc1b31d90b50458ece18c5cd548abd`。

`ROOT.json` 与 `payload/ROOT.json` 是执行前的授权/路径快照，原有 `GPU_queries=0`、`GPU_dispatches=0` 只描述快照时点，不能当作本轮累计用量。真实终态应读取 `payload/execution/CONTROLLER_RESULT.json`、不可变事件账本和 `RESOURCE_LEDGER.json`：六个有限单元均实际派发并 PASS。四卡获授权且无 GPU 时间硬上限；本次单卡速度对照按合同在同一 GPU0 串行完成。

两条 fresh 完整执行32个token与24个GDN层；独立CPU实际比较9024组Tensor，torch.equal和byte equality均全部相等，首次差异为空。两个独立激活驱动FP64重放各完整核验768条token/layer记录并通过原冻结bound。

OLD FP三次为13.382607053095176、13.951699074157712、13.774973185423399 TPS；中位数13.774973185423399 TPS、72.5954226218164 ms/token。NEW_STRICT同卡单次为7.741047785182266 TPS；该单次对照不是多重复统计显著性证明。两者均是固定输入、128 warmup+128 measurement的短缓存FP结果。

控制器、supervisor及六个worker共8个自有进程均已核验退出，全部UUID的实际NVML记录均无这些PID。释放证明为 `analysis/POST_GPU_RELEASE_ADMISSION.json`。进程预留GPUh区间为0.4047023999608225至0.40566131705211267；精确物理GPUh仍为UNRESOLVED。

最终407项源及249个外部依赖（含四个模型分片）已重新流式SHA/bytes核验。407项中403项继承不变、4项明示修订；原生数学、dtype、fixture和benchmark计时体不改。旧CPU530、控制修订51及本任务先前CPU454资产的封存保持原样。

5252个新payload文件、6,760,421,526 bytes已完成55源双遍SHA和29独立双遍磁盘SHA/bytes核验，并原子封存于 `/data/zypan/qwen_historical_runtime_gpu_recovery_archive/GPU_EXECUTION_20261019_V1/PAYLOAD_ASSETS`。原55源未删除；`.stream.tar.incoming`只是保留的传输工程副本，永久科学资产身份以已原子发布目录及其ASSET_CATALOG/ARCHIVE_VERIFICATION为准。大型raw/模型不下载Mac。

恢复模式仍为HISTORICAL_PORT，历史完整二进制身份UNKNOWN。本轮不授予INT8/Group32/AIME/32K或81K资格；七项CPU兼容审计见 `INT8_COMPATIBILITY_AUDIT.md`。后续任务不得自动重启本轮GPU单位或添加研究条件。

小文件交付完成以 `LOCAL_DELIVERY_VERIFICATION.json` 与单独的 `operations/EXTERNAL_RECEIPT_REPLICATION.json` 为准。Mac大小写同字节路径的投影与Linux逻辑名册分开记录，不宣称三主机物理文件名册完全一致。
