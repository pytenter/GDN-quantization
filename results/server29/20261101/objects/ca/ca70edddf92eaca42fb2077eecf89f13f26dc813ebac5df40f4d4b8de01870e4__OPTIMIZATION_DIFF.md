原CPU/历史源码、旧ledger及seal均未修改。新执行副本仅增加已授权四卡映射、无GPU时间限额控制、1800秒无进展监测、精确自身进程清理、共享首次失败、同卡对照和跨卡测速互斥、归档流式I/O。

core AST仅允许cuBLAS环境字符串及物理可见设备准入字面值；反向AST必须恢复原文语义。qualify数学/DeepRecorder/replay/Profile控制差分均有CPU断言；GDN、Codec、FP32 state/QDQ/writeback次序保持原样。B1尚未执行，这些工程变化不构成已验证提速。

停止后仅新增独立只读审计和交付程序。未把libtorchaudio补进已冻结白名单、未移除门禁、未热改程序、未重试单位。一次启动前命令模板编译错误和CPU审计工具错误保留在对应history中，均不伪装为GPU试验结果。
