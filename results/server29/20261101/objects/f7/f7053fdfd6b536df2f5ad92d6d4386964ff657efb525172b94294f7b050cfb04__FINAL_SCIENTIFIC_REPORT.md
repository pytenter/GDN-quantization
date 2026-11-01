# Qwen/GDN OSCAR-style Sym32：真实资格停止交付

状态：QUERY_EIGENBASIS_QUALIFICATION_FAIL_CLOSED_STOP。新增AIME实际0/90，P1/C1/S3均未派发，未产生本任务accuracy、评分表或科学CSV；P0没有重跑。

唯一GPU阶段为4096-token非AIME Native-FP Query采集：四篇冻结RepoBench文档各1024token，24层、每层32个actual recurrent head各4096个未中心化Query二阶矩样本，CPU FP64按固定文档/Token顺序累计。实际16个Key/Query head在递归前repeat_interleave2到32个recurrent/value head；相邻共享Query字节完全一致，不跨层Head聚合，样本时间相关性不当IID，源项目独立性未知。

首次失败为第13层、第24个recurrent head（零起点）。FP32部署A=U@H的Query敏感性对角均匀性最大相对误差2.0291937860262014e-06，预先冻结上限2e-6，超限约1.46%；A正交性误差1.4558495209371358e-07仍在2e-6以内。FP64特征分解相对残差1.7060933776589776e-15、FP64 U正交误差1.9984014443252818e-15、FP64理想A对角均匀误差7.885222288257288e-15。

该证据定位到预先约定的FP32部署精度检查，不能据此宣布OSCAR旋转数学错误或准确率失败。2e-6是本轮执行前冻结的工程阈值，不是AIME实测效应界；没有看到AIME后调阈值，也没有重新校准、重新拟合选矩阵、扩大容差或patch-and-retry。本次只能报告其未达到本轮既定资格合同。

独立CPU诊断以同一冻结Cq和原始basis函数只读重现同一首次失败，不构造新的候选矩阵、不执行GPU。原Cq、首次失败完整Cq/U/A诊断Tensor、已写入的10个未完成U层文件全部保留服务器并逐文件SHA封存；这些部分矩阵没有U合格seal，未部署到推理。原始异常、源码、固定阈值、输入身份和partial状态保留。

CPU PBR/原Sym32独立80项字段字节检查、Identity U/P、Group128排列负控和64-token FP64独立旋转递推通过；这些是CPU数学证据，不能冒充本轮三GPU短程资格、真实9B旋转FP等价或AIME证明。后续FP等价、91单位GPU资格、90条正式、FrozenV4评分和交互效应均未运行，因此不生成空结果文件。

仅GPU1上的一次Query采集，真实进程已退出且NVML释放；GPU2/3未执行本任务GPU工作，旧GPU0任务从未被停止/抢占/迁移。时间区间见RESOURCE_LEDGER和append-onlyGPU_RESOURCE_LEDGER，校准占用观测下界约501.54秒、任务保留上界约642.53秒（包括CPU准入/加载等），EXACT物理GPUh仍UNRESOLVED。模型权重不改，17模型/Tokenizer文件freshSHA与61实际外部依赖已核；原历史科研资产只读未改写。
