# V3 attempt_002 实际恢复启动记录

北京时间 2026-10-12 11:50:21（UTC 03:50:21）启动 `qwen_v3_recovery002`。恢复提交为 `c07f5efb9bf7cefae4906d927d78e3814ea15371`，新执行使用独立 recovery freeze，旧 STOP、原报告和 POSTSTOP 保持不变。

控制器 PID 9409，唯一科学 worker PID 9415，当前只运行 Doc3 / GPU2 正式 Track B；Doc1/2/4 的有效正式结果均复用。UTC 03:55:19 检查显示 GPU2 使用约 18080 MiB，其他三卡无计算进程；日志已通过原失败位置并推进至 128 token。此时完整 Track B 尚未完成，Track A、matched long 和 Discovery 尚未开始，不能报告完整分数或科学成功。

重复调用 controller 的实测返回 `ALREADY_RUNNING_NO_DUPLICATE_LAUNCH`，worker PID 和 CHILD_STARTED 事件数未增加。该测试只证明运行中重复调用保护，不把它解释为科学资格或断电后续跑资格。

后续顺序仍由冻结依赖决定：补齐 Doc3 → 全面板完整性检查 → 原要求的 256-token matched-drive 工程检查 → 四卡真实 8192/8256 anchor → 全局 short freeze → 原实际 32K 资源门禁及 fresh-process 完整 cache parity → 同 runtime 32K matched long → 原描述性 Discovery 分析 → 新不可变 manifest。

本文件是实际启动状态记录，不是最终科学报告。最终科学交付只在所有必需单元和门禁完成后产生。
