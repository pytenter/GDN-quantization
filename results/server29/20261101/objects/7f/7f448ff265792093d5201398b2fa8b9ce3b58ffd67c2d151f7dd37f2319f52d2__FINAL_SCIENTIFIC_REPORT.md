# Qwen/GDN Group32 资格验证实际报告

整体状态：**INT8_SHORT_RUNTIME_QUALIFIED**。新控制器终态 `FINITE_ROSTER_TERMINAL`。人类“没有预算限制啊”取消本任务GPU时间预算；原0.5GPUh运行及其预算停止封存保持不变。原微型Codec和6条32-token轨迹只读复用，新运行仅原剩余15个单位，同GPU0 UUID `GPU-a78176ed-f97f-c54f-6201-5a01b570f422`，不重跑旧前缀。

| 条件 | 配对长度 | GPU Codec | TPS | ms/token | 测速峰allocated GiB |
|---|---|---|---:|---:|---:|
| B0 | [32, 128, 512] | GPU_CODEC_QUALIFIED | 8.673151984839238 | 115.29833695385605 | 16.755578994750977 |
| B1 | [32, 128, 512] | GPU_CODEC_QUALIFIED | 8.773311488635922 | 113.98204672150314 | 16.755762100219727 |
| B2 | [32, 128, 512] | GPU_CODEC_QUALIFIED | 8.628929769729911 | 115.88922690134495 | 16.7549991607666 |

1. **B0 canonical是否复现？** CPU全24层×32heads共12,582,912元素值和字节一致；原7项canonical AST不变。已完成GPU微型37 fixture×2严格字段比对。FP32 scalar除127保留CUDA reciprocal语义及Sym负零flags，未换为另一个舍入数学。
2. **B1 Group32 Sym是否通过？** GPU Codec为GPU_CODEC_QUALIFIED；实际9B persistent与同卡fresh完成长度见表。仅Key连续32分组，每Value列4组，不跨head/layer。没有将未运行长度补造PASS。
3. **B2 Asym资格？** GPU Codec为GPU_CODEC_QUALIFIED。FP32 scale、UINT8 zero-point/codes、int16减zp和decode严格参考通过；include-zero extrema、Tensor/Tensor除255和RNE顺序不变。真实模型范围见表；没有失败后改dtype/阈值。
4. **是否统一Runtime和时序？** 三者相同HISTORICAL_PORT、模型BF16、输入、H128、种子和设备，仅Codec不同。Worker及所有科学src与旧资格逐字节相同；本次仅资源准入、生命周期和新I/O root变化。历史完整二进制EXACT仍UNKNOWN。
5. **是否token0真实persistent？** PASS canary从token0逐token×24层QDQ并写回，共96768次层写回。query长度1；多token chunk被拒绝。未量化state没有参与下一次递归。载体是native FP32 QDQ state，48MiB，不是物理packed INT8。
6. **是否真实消费QDQ state？** 已完成canary共96336次native helper initial_state真实值及完整字节检查，和上一token整数编码的临时CPU decode比较；不是指针或hash替代。每完整N-token轨迹检查(N-1)×24次。
7. **fresh稳定性？** 表中每个长度在同UUID两独立模型进程运行，真实raw全tensor值/字节、dtype/shape/index一致，包括state/preQDQ/codes/metadata/decode/cache/logits及下一kernel输入。短512不能当32K/81K稳定性或误差上界。
8. **真实TPS？** 仅表中实际完成的128warmup+128measurement、CUDA同步、无Profiler、无逐token观测的匹配窗口。测速保留QDQ与finite检查；完整writeback/next-consumption值与字节观察仅在Canary开启，测速窗口关闭该观察，不能称其逐token重复审计。未测字段NOT_RUN，不借历史9.064610或FP13.774973填入；full-raw canary的wall包含观察、hash和归档，不能当推理TPS。
9. **显存/Clipping/数值风险？** 峰值为实际worker在模型常驻后reset统计，非加载峰；不把理论编码字节当显存节省。B0 scales384KiB，B1 scales1.5MiB，B2另zp384KiB；codes各12MiB，Sym临时bool负零flags12MiB。持久载体仍48MiB。已完成范围finite通过；canary完整round前clipping统计未单独保留，不能声称长程零overflow/saturation。H GPU正交阈值2e-6通过，不声称另有CPU/GPU H变换bit证明。
10. **后续32K资格？** 具备完整512 fresh及GPU Codec资格者：['B0', 'B1', 'B2']。这些仅为后续匹配Future-KL的短程前置资格，不证明功能精度改善。仍需独立文档、同协议FP、32768输入/384位置/全248320词表FP32 KL及新用户授权；没有启动32K、AIME、训练或新研究。

资源预留观察GPUh：原运行[0.38975954150574077, 0.39091705589244763]，本次[4.9573076083966425, 4.959814415375392]，累计[5.347067149902383, 5.3507314712678395]。**EXACT_GPUh=UNRESOLVED**，轮询区间不是精确物理计算时间。所有实际dispatch均独立核验首次进程退出和全UUID释放；失败数量0。30分钟仅无真实phase/token进展watchdog，120秒仅清理观察，不是GPU运行预算。

归档：当前及所引用原raw共2016对象、366819508224字节，.29独立fresh流式SHA/bytes核验；原catalog只读，新增catalog原子发布。孤儿tmp 0。模型和大tensor未下载Mac，原CPU/FP/GPU封存未改写，失败与预算停止历史全部保留。
