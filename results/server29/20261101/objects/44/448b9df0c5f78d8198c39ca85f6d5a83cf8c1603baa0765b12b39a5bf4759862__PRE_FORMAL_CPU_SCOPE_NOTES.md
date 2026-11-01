# CPU 工程与门禁范围说明

所有 CPU fixture 为合成输入、mock source/archive 或纯控制生命周期，GPU credit=0，不是正式60条轨迹。真实29 canary 单独验证 SHA、fsync、NOREPLACE 与重复发布拒绝。

CPU_CONTROL_FRESH1：实际测试进程退出1，控制器在正常短子进程收尾时停止；当时临时目录被自动清除，具体 /proc 观察未保留，精确原因 UNRESOLVED。不得伪造历史内核状态。

CPU_CONTROL_FRESH3：保留 `validation/cpu_failure_scopes/normal_1793114973898874552/`，实际发生 Live argv changed 拒绝；原代码只接纳 empty cmdline 且同次 stat=Z，遗漏 argv 清空与状态采样不同时的收尾情况。empty argv 先于所采 Z 是 INFERRED 原因，而非该失败工件直接记录的 /proc 事实。

新控制只将同 starttick/PPID 的 empty argv 视为 EXIT_PENDING，最多120秒等待真实 owning Popen 整数退出；不推断退出码，不向空或错误 argv 发信号。正常不同非空 argv 仍失败。FRESH4/FRESH5 各实际60正常 +60 leaderZ cleanup +真实exit7负例闭合；正常短进程也实际观察到 EXIT_PENDING，均真退出0并回收。所有 CUDA 生产数学字节不受此控制修复影响。

CPU_DELIVERY_FRESH1/FRESH2：实际 owning exit0/7/−15、真实 leaderZ cleanup、Bash wait0/7/143，以及两fresh原FrozenV4同合成payload评分相同。PIPELINE 系列进一步执行实际 parent.finish→两fresh scorer→report finisher→CSV→manifest，archive调用仅mock；实际29资格由独立canary提供。

本地构建中文报告时，Mac Python3.9 读超长 UTF-8 行报告编码错误；文件实际UTF-8可解码，显式编码声明后生成成功。这是 CPU 文本包装问题，未改变量化公式或 GPU 结果。原 CPU prototype 跨轨迹数值 screen FAIL 也单独保留，不与上述控制 fixture 混为GPU故障。
