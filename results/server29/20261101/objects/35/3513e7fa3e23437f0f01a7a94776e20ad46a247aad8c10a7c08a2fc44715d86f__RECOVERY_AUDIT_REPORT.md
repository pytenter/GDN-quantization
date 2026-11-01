# V3 attempt_002 恢复审核

原控制器和 Doc1/2/4 worker 已自然退出；没有发送终止信号。原 controller 返回失败，worker 状态为 `[0,0,1,0]`，Doc3 在模型 forward 前停止。

原 CPU-only 后封存完成，三篇已完成正式文档的结果、原日志、原 STOP 和资格证据逐项 SHA 核验通过；不能以更早生成的 FINAL_REPORT.md / SHA256SUMS.txt 替代这次闭合证据。

## 修复与等价性

`scale_science.py:279` 的文本审核由 `Path(...).read_text()` 改为 `Path(...).read_bytes().decode('utf-8')`，仅纠正 CRLF 隐式归一化。

原脚本保持原样。隔离的 repaired source snapshot 与原脚本只有上述单表达式 diff；其余函数、类、数值核心、原 teacher collector、quantizer、observer、baseline、metrics 和原身份门禁 AST 完全相同。

恢复 adapter 仅将经过静态审核的 repaired `track_b` 函数绑定到原模块 globals。其余代码和 hash guards 仍从未修改的原模块运行；adapter 和 repaired function 通过独立 recovery freeze 绑定真实源码 SHA，不伪造原文件 SHA。执行 receipt 同时保留原 revision 和修复 revision。

CPU 回归已通过：真实 Doc3 的 4962 CRLF、四篇文档的原始 SHA/token SHA、全部 32768 个精确 token、原 capture positions，以及六种换行/UTF-8 synthetic cases。没有写 frozen asset，没有模型 forward。

模型与 checkpoint 的 14 个文件还与 retained runtime identity manifest 逐字节 SHA 匹配。已有 1110-object 数值资格、三进程复现、observer parity 和 Doc1 H/C6R 工程资格通过封存身份核验，因此复用，不重跑 GPU 资格。

## 工作单元与恢复

正式单元按原协议展开为 48 个 stage/document/condition 单元：Track B 16、Track A 16、MatchedLong 16。Track B/A 的 H32/H64 是同一条持续轨迹的观察窗口，不是独立样本。

其中 Doc1/2/4 的 12 个 Track B 条件单元具备完整原子结果、结束日志、1024 QDQ 和24层、8 capture positions、原 teacher artifact、正确 source revision 和 frozen numerical contract，可复用。

只恢复 Doc3/GPU2 的四条件 bundled Track B，之后必须通过整个 32-row 面板完整性门禁，才允许进入原始 matched-history 的 256-token 工程资格、真实8192 anchor、全局 short freeze、fresh-process完整cache恢复资格和32768 continuation。

独占锁、STARTED/COMPLETED/STOP receipts 与 write-once 输出阻止重复启动、覆盖旧结果或盲目重试。新数值或科学错误仍 FAIL_AND_STOP。

## 资源与边界

Dry-run 时四张 RTX3090 均空闲；可用存储约37.54 GiB，预计新增峰值约6.89 GiB，预计剩余约30.64 GiB。该估计包含两份真实 anchor/文档、缺失 teacher artifact、短窗口 drive 和序列化/报告开销；不免除原实际8192/32K显存和 checkpoint容量门禁。

不使用历史16 GPUh或85%显存行政停止条件。不修改 model、rotation、quantizer、horizon、metric、threshold、numerical contract、GPU映射或配对定义。旧 G 不与新 short 拼接。不会训练、运行 AIME/自由生成、梯度优化或独立KL。

数值 PASS 和审核修复并不等于 surrogate 科学成功。最终只依据真实新 matched 结果作四文档 Discovery 描述；Track A functional baseline 仍为 NOT_COMPARABLE，future state error 也不能冒称 functional KL。
