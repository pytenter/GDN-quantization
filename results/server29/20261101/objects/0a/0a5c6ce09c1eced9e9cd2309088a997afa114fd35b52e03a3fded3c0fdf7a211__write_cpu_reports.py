import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads((R/p).read_text())
def save(p,v):(R/p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
a=load('qualification/CPU_PROCESS_0.json');b=load('qualification/CPU_PROCESS_1.json');assert a==b
tiny=load('qualification/TINY_CPU_PROCESS_0.json');assert tiny==load('qualification/TINY_CPU_PROCESS_1.json')
assert all(x['passed'] for x in load('qualification/INDEPENDENT_CPU_REVIEW.json')['checks'])
save('CPU_CODEC_QUALIFICATION.json',dict(status='CPU_CODEC_QUALIFIED',methods={m:'CPU_CODEC_QUALIFIED' for m in ('B0','B1','B2')},all_18_required_categories=True,unique_main_assertions=35,additional_boundary_assertions=13,real_state_heads=27,fresh_CPU_processes=2,CPU_actual_output_bytes_equal=True,reference='Independent NumPy explicit Key-group indexing; exact Fraction tie-boundary checks',arithmetic_contract='CODEC_NUMERICAL_CONTRACT.md',evidence=['qualification/CPU_PROCESS_0.json','qualification/CPU_PROCESS_1.json','qualification/ADDITIONAL_BOUNDARY_CPU.json'],GPU_claim=False))
save('CPU_REFERENCE_PARITY.json',dict(status='PASS_CPU_CANONICAL_ALL_BYTES',canonical=load('qualification/CPU_PROCESS_0.json.canonical.json'),independent_numpy_fields=['codes','scales','zero_points','negative_zero_flags','decoded'],canonical_scalar_count=24*32*128*128,Sym_CUDA_semantics='EMULATED_EXPECTATION_NOT_GPU_PROOF',GPU_NOT_RUN=True))
save('NEXT_CONSUMPTION_AUDIT.json',dict(status='PASS_TINY_REAL_CLASS_CPU_ONLY',actual_value_and_byte_comparison=True,official_tiny_fresh_processes=2,conditions_per_process=6,tokens_per_condition=8,GDN_layers=2,Key_Value_shape=[128,128],official_layer_writebacks=192,official_actual_next_kernel_checks=168,actual_cache_corruption_negatives_detected=12,token0_QDQ=True,all_24_layer_9B_GPU='NOT_RUN',evidence=['qualification/TINY_CPU_PROCESS_0.json','qualification/TINY_CPU_PROCESS_1.json']))
save('FRESH_PROCESS_DETERMINISM.json',dict(status='PASS_CPU_ONLY',codec_two_independent_processes='EXACT_OUTPUT_BYTES',tiny_real_class_two_processes='EXACT_OUTPUT_BYTES',scope='CPU FP32 and BF16 weights, FP32 recurrent carrier, each B0/B1/B2; not cross-device reproducibility',same_GPU_fresh='NOT_RUN'))
for name in ('GPU_CODEC_ONLY_QUALIFICATION.json','GPU_FULL_MODEL_CANARY.json'):
 save(name,dict(status='NOT_RUN',reason='GPU_NOT_AUTHORIZED_FOR_THIS_TASK',ALLOW_GPU_CANARY='NO',conditions={m:'NOT_RUN' for m in ('B0','B1','B2')},GPU_queries=0,GPU_dispatches=0))
save('INT8_THROUGHPUT_STATUS.json',dict(status='NOT_RUN',reason='GPU_NOT_AUTHORIZED',conditions={m:dict(TPS='NOT_RUN',ms_per_token='NOT_RUN',peak_GPU_memory='NOT_RUN') for m in ('B0','B1','B2')},CSV_not_created=True,historical_9_064610_not_reused=True,prior_FP_TPS_not_INT8=True))
save('RESOURCE_LEDGER.json',dict(status='NO_GPU_DISPATCH',GPU_queries=0,Driver_API_calls=0,NVML_calls=0,GPU_dispatches=0,model_9B_loads=0,GPU_reservation_hours=0,EXACT_GPUh='NOT_APPLICABLE_NOT_RUN',CPU_remote_tiny_real_model_only=True,parent_raw_read_only=True,large_tensor_downloaded=False,small_real_fixture_bytes=1769472,budget_proposal=dict(device='single RTX3090 GPU0',max_GPUh=.5,hard_seconds=1800,protective_seconds=1650,cleanup_seconds=150),separate_new_task_GPU_permit=None))
root=load('ROOT.json');root['status']='CPU_CODEC_QUALIFIED_GPU_NOT_AUTHORIZED';save('ROOT.json',root)
docs={
'EXECUTION_PLAN.md':'''# 执行结果与后续门禁

本次已完成：核验上一轮3753项最终交付、隔离复用407项有效源；提取7项不变 canonical AST；实现三种 Key-axis codec；冻结算术与 token0 QDQ 合同；两次独立CPU进程和独立NumPy参考；27个只读历史state样本；两次真实Qwen类小型CPU模型的FP32/BF16完整调用链、真实 next-consumption 和负控；另行静态复核。

GPU默认NO。不得把前一任务四卡/不限时授权移用于本任务。没有查询GPU/NVML/Driver，没有9B加载或FP forward。

已准备37组GPU微型fixture和逐字段独立参考。未来严格顺序：同卡微型codec全字段bit gate → 三条件各32-token两fresh → 对通过的条件128两fresh → 512两fresh → 合格条件128warmup+128measurement。有限22个进程单位，同卡串行，不重跑旧FP。不因中途数值表现改组长、阈值或计划。对应失败分支停止；共享身份/资源失败全停。

GPU新permit须绑定实际SOURCE_FREEZE和GPU_PLAN、GPU0 UUID、独立人类授权、0.5GPUh/1800总秒/1650保护/150清理。计划的保守逐单位准入可能使512或测速NOT_RUN，不把30分钟预算当作保证全矩阵完成。严格数值观测/全raw远端留存费用计入资源；测速关闭逐token观察，保留QDQ/finite检查和同步。

CPU证据已经封存，任何未来GPU运行使用隔离执行root与新permit；不回写本次CPU seal。新GPU执行前重新核验所有依赖、实际可用卡/UUID/owner/dynamic maps/库SHA。旧任务保持只读。
''',
'SOURCE_AND_RUNTIME_IDENTITY.md':'''# 源码、Runtime与历史边界

实际父交付为 QWEN_GDN_HISTORICAL_RUNTIME_GPU_RECOVERY_V1/gpu_execution_20261019_v1/delivery。3753项索引和407项有效源均已逐SHA/bytes核验。有效清单SHA为48ec3e527b28cc1a1860a72559d0e9fafad96866c9d981f6b0c80d4bf5ebe19c，父清单64ca5853e4ed65ab3e2c59d9deb85b4aa026f0f712abd5496495b7b912fcb457，403继承+4修订；没有用405+2替代。

历史模式仍为HISTORICAL_PORT；完整历史二进制EXACT身份UNKNOWN。父FP短32-token与短测速资格不证明INT8或32K/81K。历史FP三次中位13.774973185423399TPS，NEW参考7.741047785182266TPS，只作历史背景，本轮未重测、不作为INT8数字。

需要澄清附件中的时间点：前序finisher本身当时未做最终seal，随后该任务完成最终封存及三地交付。保留两个阶段，不补造finisher证明。真实最终manifest SHA cece32eb6c2f51d3b1316d8960f34406c9b4b0142a1a1a143e0de1b72c611babf5（准确值以provenance/PARENT_FINAL_DELIVERY_VERIFICATION.json为准）；seal SHA5be227f4b61eb61b641c559720e05a3d00159c8cc5e1d3ed53fafd712ee104b6。历史raw保持原位，未修改删除。

canonical Hadamard、normalize、patch、get_state、quantize_c128、seed及loader共7项AST完全不变。新适配不修改原FP forward、GDN core、Attention/MLP、权重/BF16或cache数学。全模型使用32层/24GDN层、32heads、128x128状态。小型真实CPU模型使用2个GDN层/2heads/128x128，仅检验调用链。

本地CPU Torch2.5.1，远端CPU Torch2.5.1+cu121；远端只导入CPU并显式禁止CUDA初始化。61项当前外部源/模型文件/原生库/解释器已重新流式SHA/bytes核验，未加载9B。实际GPU库映射、Driver/NVML、逻辑序号身份只可能在新授权后真实核验，目前NOT_RUN。

旧Asym128/64源码只读留存，未直接移植FP16 scale实验结果。本轮C32 Key轴、FP32 scales，独立数据与结果。Ling仅提供风险经验，没有复用其Value轴/FP16结果。
''',
'HISTORICAL_PORT_INT8_PROTOCOL.md':'''# HISTORICAL_PORT_PERSISTENT_INT8_GROUP32_V1

三条件相同H128 q/k注入、FP native helper、BF16权重和输入，采用一次一个token的原生stepwise路径。首token forward结束后即对全部24层FP32 recurrent state QDQ并原位写回，此后每token相同操作。拒绝query sequence length>1的chunk；不能把chunk末尾一次QDQ称为逐token。

在下一native GDN helper入口读取真实initial_state，临时CPU解码上一token的整数codes/FP32 scales/整数zp或negative-zero flags，然后比较实际Tensor dtype/shape/torch.equal及完整字节。不是指针或hash替代。预期编码仅诊断使用，不参与模型输入；不存在未量化recurrent shadow。观察模式对全部head/全部层检查；CPU小模型已实际验证，9B GPU仍NOT_RUN。

canary保留全部相关preQDQ、整数fields、postQDQ/writeback、实际kernel initial_state、logits与全cache raw。无损重复字节可共用packet offset，须实际字节相等后去重。fresh0流式永久归档29、独立SHA/bytes原子发布后提交receipt；fresh1直接读取参考实际raw比较，首差异完整保留当前raw，禁止前缀重跑取证。临时CPU解码/观测不成为GPU recurrent buffer。

实际持久carrier仍为原生48MiB FP32 QDQ cache，不能称物理packed INT8。测量关闭观察/逐tokenhash/远端raw传输，保留必要codec和finite检查；128 warmup后同步计时128token，末尾再次同步。不沿用历史OLD首token漏QDQ或AIME整段未QDQ prefill协议。
''',
'GROUP32_FUTURE_KL_NEXT_STAGE_PLAN.md':'''# 后续方案，仅准备，禁止自动启动

拟名QWEN_GDN_GROUP32_FUTURE_KL_PILOT_V1。当前没有任何条件获得真实9B GPU完整资格，因此没有候选可直接准入。

未来先完成对应codec数学、32/128/512真实persistent及fresh/资源资格。再以独立新授权/预算/文档冻结同协议FP_NATIVE、H+C128 Sym、H+C32 Sym、H+C32 Asym：32768输入、384冻结位置、完整248320词表FP32 KL(FP_NATIVE||Student)、math.fsum/384、文档级聚合。全部学生从token0每token每GDN层写回QDQ。

主要差值：Delta_group=KL(H+C128)-KL(H+C32 Sym)；Delta_asym=KL(H+C32 Sym)-KL(H+C32 Asym)。正值表示该匹配比较中KL下降；两者分开，不用短canary预测结果。不以历史NEW_STRICT KL替代新FP/学生对照。N、候选/排除、项目独立性、模型/runtime/source/输入/位置身份、算术分辨率与统计计划需在评分前另行冻结。当前不选择科学方向，不启动任何32K/AIME/训练。
''',
'FINAL_SCIENTIFIC_REPORT.md':'''# Group32资格报告：CPU_CODEC_QUALIFIED；GPU_NOT_AUTHORIZED

本次授权范围内的CPU工作已实际完成，B0/B1/B2均为CPU_CODEC_QUALIFIED。9B GPU、同卡fresh及吞吐全部NOT_RUN。没有启动32K、AIME、训练或重跑FP恢复。

1. **B0是否复现canonical？** 是，CPU真实验证24层×32heads，共12,582,912元素；重建值与字节完全一致、maxabs=0。原7项canonical AST不变。FP32 scale、safe floor、舍入、clip及负零均保留。CUDA标量除法语义另有预登记模拟参考，尚未获得GPU证明。
2. **B1是否通过GPU与真实QDQ？** 尚未；CPU Key轴连续4×32分组、独立参考与写回检查通过。GPU Codec和9B persistent均NOT_RUN。
3. **B2 scale/zp/codes/decode资格？** CPU全部字段通过独立NumPy参考、边界与真实样本检查，UINT8 codes/zp、FP32 scales、int16相减。GPU全部NOT_RUN。张量除法顺序已冻结，遇跨度溢出拒绝，不事后改数学。
4. **统一Runtime及时序？** 已实现同一适配器和H128，不修改FP helper/权重/Attention/MLP；三条件只变codec。小模型真实调用链通过，9B GPU合同待执行。
5. **首token persistent？** 小型真实Qwen CPU从token0执行，FP32/BF16权重下三条件均通过；没有沿用旧首token漏QDQ或整段prompt不量化。9B GPU尚未证明。
6. **下一kernel实际消费？** 两fresh CPU过程共192个合法层写回、168次实际helper initial_state值及字节比较通过；12个故意破坏真实cache负控全部检测。不是hash/指针代替；没有未量化递归shadow。规模仅2个GDN层/2heads/128x128。
7. **同卡fresh？** 同卡GPU NOT_RUN。CPU两独立进程的codec完整输出和真实Qwen类FP32/BF16轨迹完全相同。不能把CPU稳定性升级为GPU稳定性。
8. **INT8 TPS？** B0=NOT_RUN，B1=NOT_RUN，B2=NOT_RUN，显存峰同样NOT_RUN。未创建空壳INT8_THROUGHPUT_RESULTS.csv；没有借用9.064610或13.774973作为当前结果。
9. **Group32风险？** CPU全18类资格和额外13项边界检查通过；极小scale有1e-12安全floor，跨度溢出明确FAIL。全24层尺度元数据B0=384KiB、B1=1.5MiB，B2另加384KiB zero-point；三者codes均12MiB。Sym为保留canonical负零另有12MiB bool flags（临时编码/审计元数据）。实际FP32 recurrent carrier仍48MiB，不是packed INT8，也不据此声称显存加速。clipping/saturation只能作为编码计数解释，GPU发生率、实际峰值与长程误差未知。短资格不证明32K稳定或精度收益。
10. **谁可进入Future-KL？** 当前没有任何条件完成全部9B GPU门禁，三者均不能直接进入。后续方案已准备；必须新授权先完成短GPU资格，再独立授权32K。

CPU证据包括：35项主断言（两独立进程重复）、13项额外边界检查、27固定真实state小样本、独立NumPy逐字段参考、精确Fraction tie检查，以及两fresh真实Qwen类小模型。独立静态/负控复核38项通过。61项当前外部依赖已fresh SHA/bytes核验；实际GPU库/设备门禁没有执行。

| 条件 | CPU | GPU Codec | 9B persistent | 同卡fresh | INT8短Runtime/TPS | 32K准入 |
|---|---|---|---|---|---|---|
| B0 H+C128 Sym | CPU_CODEC_QUALIFIED | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | 未具备 |
| B1 H+C32 Sym | CPU_CODEC_QUALIFIED | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | 未具备 |
| B2 H+C32 Asym | CPU_CODEC_QUALIFIED | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | 未具备 |

没有NUMERICAL_QUALIFICATION_FAIL、PERSISTENCE_CONTRACT_FAIL或GPU_RESOURCE_FAIL科学事件；合成负控的预期拒绝不是正式GPU失败。GPU尚未授权是NOT_RUN，不是数值失败。整体仍为CPU阶段合法完成、完整GPU资格未完成。

前序finisher当时未封存，随后最终seal与三地核验已完成；本次从实际最终交付核验407源和3753索引，保留两阶段事实，没有修改或删除历史文件/raw。详见源码身份与provenance锚点。

附件D1明确：默认ALLOW_GPU_CANARY=NO，且“此提示词不代表用户已经授权GPU”。本任务尚无另行GPU grant/permit，因此本次GPU查询/Driver/NVML/dispatch/9B加载均0。准备的22有限单位不保证在0.5GPUh内全完成；预算不足必须保留NOT_RUN，不延长或借历史数值填表。
'''}
for name,text in docs.items():(R/name).write_text(text)
# Avoid transcribing a long hash manually in prose: exact anchor is injected.
p=R/'SOURCE_AND_RUNTIME_IDENTITY.md';text=p.read_text();wrong='cece32eb6c2f51d3b1316d8960f34406c9b4b0142a1a1a143e0de1b72c611babf5'
correct=load('provenance/PARENT_FINAL_DELIVERY_VERIFICATION.json')['anchors']['SHA256_MANIFEST.json']['sha256'];p.write_text(text.replace(wrong,correct))
print('CPU reports and explicit NOT_RUN receipts written')
