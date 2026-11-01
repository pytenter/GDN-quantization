# 启动前 CPU 包装记录

首次新目录上传仅传输了文件，远端空 `runtime` 目录尚不存在。随后 CPU self-test 包装调用 `runtime/CPU_SELFTEST.mkdir()` 没有 parents=True，真实返回 FileNotFoundError。错误发生在任何 child Popen 之前，CPU 测试和 GPU 任务均未启动，0 GPU credit。

修复仅为独立新目录的空目录创建，以及验证输出父目录的显式创建。保留此实际失败，不把它计为 CUDA 数值失败、GPU retry 或已有任务故障。原任务目录、kernel、codec 和公式未修改。

随后本地冻结包装曾把新 README 与旧目录同名 README 误当作继承科学文件比较，触发 AssertionError，发生在 SOURCE_MANIFEST 创建之前、无 GPU 启动。比较范围改为原 frozen manifest 实际复制的文件清单；新任务说明、请求和控制脚本单独归类。没有放宽对实际继承 codec/kernel/runtime/mask 的逐字节验证。
