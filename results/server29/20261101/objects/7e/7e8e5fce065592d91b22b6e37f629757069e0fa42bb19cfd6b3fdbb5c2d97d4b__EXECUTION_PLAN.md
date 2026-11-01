本轮是原CPU封存后的四卡GPU资格执行，最终状态 `FAIL_CLOSED_STOP`。

人类明确授权四卡及无GPU时间限制；累计和单元时间上限已移除，1800秒只作为无真实进度watchdog。冻结名单每卡16单位，先四个256输入FP fresh数值单位，再六个128warmup+128measurement测速，只有原5%+严格不重叠收益门槛通过才进入Profile和H资格。所有B0/B1对照绑定同一卡，不跨卡合并TPS。

实际仅四卡各派发一次 `B0_FP_Q0`，提交 [252, 253, 253, 256] 个参考token。GPU3在完整参考构建后，动态库身份门禁发现未绑定 `libtorchaudio.so`；全局停止并释放四卡。B0第二fresh、B1、性能、Profile、H及Dual-M均未执行。没有重试、热补丁或增加阶段。详细冻结计划保存在 `remote_evidence/gpuN/qualification/CUBLAS_PREREGISTRATION.json`；原CPU151文件封存完整保存在 `parent_cpu/`。
