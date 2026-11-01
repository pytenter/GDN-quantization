第一次CPU staging在首次SSH预留root之前失败：本地ControlPath已不存在，服务器拒绝无master的密钥登录。未上传源码、未创建远端执行目录、未运行远端CPU入口，更无GPU动作；原工具、bundle、状态和失败回执均保留。

随后通过交互式SSH恢复既有人类凭据对应的master，未将凭据写入文件、远端脚本或输出。实际只读确认目标root不存在后，另建连接恢复attempt02工具与独立状态/输出名；旧attempt01资产不覆盖，也不删除失败记录。第二次是连接外部状态改变后的有界CPU准入，不是重跑任何GPU科学单位或修改407源。

attempt02远端410个小载荷文件（407源+3 metadata）共22,605,499 bytes在CPU检查前后逐项SHA/bytes一致。当前--check退出0，用时61.478秒（同主机monotonic），纯stdlib流式核验当前模型四分片/配置/Tokenizer/源/必要动态库SHA；--print-env退出0，用时0.077秒，只输出历史目标设置。没有导入Torch、调用Driver/NVML、初始化CUDA context或执行forward。CPU时间没有被写成GPU时间或测速窗口。

当前成功回执为qualification/REMOTE_CPU_ADMISSION_ATTEMPT02.json；首次失败回执qualification/REMOTE_CPU_ADMISSION.json保留原字节，不被重贴成PASS。只有新成功回执用于本次当前CPU准入判定，全部真实GPU步骤仍为NOT_RUN。
