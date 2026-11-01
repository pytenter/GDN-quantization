本轮GPU已获授权并真实执行，但cuBLAS资格未完成。四卡各仅运行B0首个参考，提交token数252/253/253/256，合计1014。GPU3的 `NUMERICAL_INDEX.json` 的PASS只表示256输入参考构建，`actual_bytes_compared=false`；其worker最终RESULT为FAIL_CLOSED_STOP。另三卡被共享STOP终止，RESULT和完整INDEX缺失仍保留缺失，不补写PASS。

触发点是冻结worker的动态库身份门禁：basename以libtorch/libc10开头会归入torch库，故libtorchaudio.so也必须存在于绑定清单。该实际加载路径不在清单，执行在发布RESULT PASS之前停止。这不是已观察到的B0/B1误差、无收益结论、GPU时限或磁盘满事件。

没有B0第二fresh、B1自重复、B0/B1字节对照，故误差最大值和最早数值分歧均不适用，不能填0。没有任何合法128+128测量，原5%+不重叠门槛未被检验。H/Dual-M仍NOT_QUALIFIED_FOR_THIS_CODEC；条件StageD3因FP资格失败未进入。没有自动修复或重跑。
