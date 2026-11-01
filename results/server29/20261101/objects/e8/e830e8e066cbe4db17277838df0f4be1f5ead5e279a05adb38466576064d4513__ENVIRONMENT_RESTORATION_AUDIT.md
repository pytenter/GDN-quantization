# OLD 环境恢复审计（CPU-only）

本轮严格 `ALLOW_GPU_RESTORE=NO`，旧四卡授权不适用。本子任务未导入 Torch、未查询或初始化 GPU、未运行 benchmark 或全模型，未修改安装环境和历史资产；只通过标准库读文件、安装 metadata 与计算流式 SHA。

历史 13.742036 TPS 对应调和任务的受控 OLD 重放，不能称为已恢复原始 AIME 的全部二进制机器状态。历史 175 个交付小文件已重新逐个 SHA/bytes 校验；源码清单 `b59ee3a7417ebdae7f4f857bc3e8eb2f1644c373d6d28bf8d31da2871671c9f3` 含7执行源和194外部依赖，没有任何 `.so` 或 Python executable 的二进制 SHA 行。

| 事实 | 可采信结论 |
|---|---|
| OLD 六个裸速单位的 ENV 一致 | Python3.10.18、Torch2.5.1+cu121、CUDA build12.1、Transformers5.16.0.dev0；Torch24/24；global det=False；matmul TF32=False；cuDNN TF32=True、det=False |
| CUBLAS / OMP / MKL / CUDA_DEVICE_ORDER 字段显式为 null | 当时这些键确实 unset，不是未保留；不能由此推断所有线程库、调度或隐藏默认值 |
| LD_LIBRARY_PATH、LD_PRELOAD、部分TF32/OpenMP override没有保存 | UNKNOWN，不能将今天 SSH 的 unset 倒填到历史 |
| 当前四片权重流式 SHA/bytes 与历史共享 checkpoint 相同 | 19,306,310,880 字节，四片 before/after-stat 稳定；没有下载模型。这证明当前匹配调和批次 checkpoint，不能补原更早 AIME 缺失的完整权重身份 |
| 六个历史模型小文件 SHA 相符 | config/configuration/index/preprocessor/tokenizer_config/video_preprocessor 可复用历史绑定；tokenizer.json、merges、vocab、chat_template 当前新封定，不伪称原历史已锁定 |
| 当前26个 native 物理文件重新 SHA | 共3,992,006,873字节，全部匹配前次当前绑定；只证明现有磁盘文件，不是历史或本轮实际 loaded-map 证明 |
| 当前pip安装与conda并存 | Torch由pip安装；Transformers editable指向原源树；pip cuBLAS12.1.3.1 / runtime12.1.105 / cuDNN9.1.0.70；另有conda CUDA12.9和cuBLAS12.9.1.4。不能凭版本或库存推断实际使用哪一套 |

历史 driver/cuBLAS/cuDNN loaded-map 与完整二进制身份没有保留。今天的库 SHA、安装 metadata、mtime 或相同版本不能追溯补成历史 PASS。当前 SSH 进程 CPU affinity0..47 也不是历史 affinity。有效 Torch 当前线程和 CPU 实例资格由主任务另行收集，本审计不从环境变量猜测。

因此建议恢复副本采用 `HISTORICAL_PORT`：忠实恢复已冻结 OLD Python 数学和执行路径，仅隔离 I/O/路径并封定当前可访问依赖。它不能标为完整历史二进制 `EXACT_HISTORICAL_REPLAY`。CPU 身份通过不证明恢复版数学正确、fresh可重复或13TPS已恢复；这些 GPU 项均 `NOT_RUN`。不把 OLD 与 NEW 的结果冒称同协议匹配。

`generation_config.json` 当前不存在，明确记为 MISSING；本任务固定 token replay 不进行自由生成，今后生成协议另行冻结。历史 OLD 可以跑完 AIME，不构成未来32K/81K、Group32 Sym/Asym的长程数值资格。

主要证据：`HISTORICAL_ENVIRONMENT_REUSE.json`、`CURRENT_ENVIRONMENT_CPU_VERIFICATION.json`、`PROSPECTIVE_MODEL_AND_NATIVE_BINDINGS.json`、`ENVIRONMENT_RESTORATION_ASSESSMENT.json`，均在本目录。每项包含原始证据路径、完整 SHA、字节及历史/当前作用域；没有新科学测速或空测量结果。
