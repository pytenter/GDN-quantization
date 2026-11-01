# 性能因素：GPU前冻结假设

- OBSERVED：历史13.384239和9.115225来自8个raw-SHA匹配分片，共120合法生成；均值/中位/加权统计不同。重算FP均值13.415834、中位13.505565，加权13.384239；H均值9.196445、中位9.251103，加权9.115225。统计聚合差别很小。
- RULED_OUT（仅数学归因）：FP没有Dual-M或Column LS；故FP表面40.7%差异不能归因LS。
- RULED_OUT（仅调用范围）：compensated只chunk，原recurrent无cumsum。当前稳定测量窗全部recurrent；不支持“每decode反复补偿”的解释。
- OBSERVED：同一model90d文件与同版Torch/CUDA；运行编排、deterministic、TF32/cuBLAS环境、线程数、Observe/finite门禁、H矩阵生命周期、prefill/QDQ介入和计时窗口不同。
- UNVERIFIED：上述配置对TPS的各自因果贡献。不得从环境变量存在、源代码op数量或wall-active差直接分配减速百分比。
- UNVERIFIED：历史物理UUID、时钟、电源/温度、CPU负载、完整optionalpackage与权重/Tokenizer分片身份；旧runner与checkpoint配置/模型源已冻结，但历史全部系统状态不能重建。
- INFERRED：新CPU24statefinite与PythonObserve增加发射成本；是否主因须matched数据。旧H重新构建H128的数学与新memoized相同，运行成本可能不同；不预断方向。

CPU决策：统计口径不能充分解释主要差距，旧真实runner+模型源可恢复，因此执行一次受限matched GPU batch；不采用近似历史runner，不改变原FP/H/GDN/quantizer数学。主FP/H各old/new三重复，若FP范围非重叠且中位比>=1.10，才运行预先声明的两个单因素FP辅助诊断与两条FP profiler（预算准入仍须通过）。
