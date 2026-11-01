from pathlib import Path
import json
R=Path(__file__).resolve().parent.parent
C=json.loads((R/'analysis/STRICT_BASELINE_CONTRACT.json').read_text())
source_rows='\n'.join(f"|{k}|`{v['path']}`|`{v['sha256']}`|" for k,v in C['source_bindings'].items())
strict='''# NEW_STRICT Runtime合同

状态：`CPU_RESTORED_STRICT_CONTRACT_GPU_QUALIFICATION_NOT_RUN`。本轮`ALLOW_GPU_DIAG=NO`；本文件恢复可核验的Strict定义，不宣称本轮已完成GPU字节等价、吞吐或资源资格。可机读全部路径/bytes/完整SHA、接口和边界在[STRICT_BASELINE_CONTRACT.json](analysis/STRICT_BASELINE_CONTRACT.json)。Strict参考是当前NEW，不把旧detFalse/24线程设置搬作优化。

## 1. 身份、模型与执行方式

原调和manifest `b59ee3a7417ebdae7f4f857bc3e8eb2f1644c373d6d28bf8d31da2871671c9f3`及Speedup manifest `d0069f87aafa9ab461827000e4322c0d4758c61a04d1d296da2b495e1c9ba90d`以各自实际绑定源为依据。前序根因root完整80文件/81行SHA索引本轮通过；原cuBLAS预登记raw SHA `404e2b2e07519434e589acfe71a2a1355b224a038d3d8d707f2eb94885ea1884`已精确复制，新请求对矩阵/先后顺序的覆盖由本任务独立addendum说明，不修改原合同。

|角色|实际远端路径|SHA256|
|---|---|---|
'''+source_rows+'''

模型目录`/data/zypan/modelscope_models/Qwen3.5-9B`，完整Qwen3.5-9B、32 decoder层、BF16参数、device_map `{"":0}`、eval、requires_grad=False、local_files_only=True。GDN 24层FP32 state `[1,32,128,128]`；full attention8层。加载器是原formal runner AST，禁止换runtime、offload、精度或算法。

权重4分片SHA/bytes来自原共享checkpoint CPU流式资格；本轮没有重新读取20GB模型或下载模型。它们作为prospective期望身份，未来任何GPU dispatch前须fresh hash全部4片；原早期AIME没有完整权重SHA，不能倒填历史。config/configuration/index/tokenizer_config等已有历史绑定；另外本轮只读stream SHA补齐tokenizer.json、merges、vocab与chat_template完整当前身份，不下载它们。这些新hash用于未来B0/B1同一现行目录资格，不能反证原历史Tokenizer字节身份，详见`analysis/contract_current_model_tokenizer_hashes.json`。

本任务固定token replay，不进行分词、prompt模板渲染或采样。原非AIME Doc1输入文件32768 token，raw SHA `18faa0ca311362181c5c3340d19d72ac39422d1ee07846e516bf34074a51703d`，canonical SHA `8ccd76d9556130c4dac436ac8d99aa7fef98a5d7c767bbd8299a79c4e276dde3`。原路径SPEED/fixtures/prospective_doc1_input_ids_32768.json。取前256个输入，seed=20261015（Python/Torch CPU/Torch CUDA）；numpy/sampler不参与本合同。首次1token prefill，累计128 forward warmup且保留cache，再测128 forward，实际RoPE128–255，8路attention KV128→256。不是32K cache，不外推81K。

## 2. 冻结Strict配置

|项目|Strict值|证据边界|
|---|---|---|
|Python/Torch/CUDA build/Transformers|3.10.18 /2.5.1+cu121 /12.1 /5.16.0.dev0|旧ENV实录；未来须实际运行确认，当前未导入torch|
|Transformers源码|commit a353632607c59463e6ced86a44c2de3c2cd62d5e，架构raw90d929…|按源manifest，不以当前HEAD猜历史|
|CUBLAS_WORKSPACE_CONFIG|`:4096:8`|B1仅改为`:16:8`，必须子进程导入torch前设置|
|deterministic|True，warn_only=False|禁止关globaldet换速度|
|fill_uninitialized_memory|本轮prospective Strict明确True，B0/B1都True|历史动态值未单独留存；原源无setter，不能把现在True回填为历史遥测|
|TF32|matmul=False、cuDNN=False|保持NEW合同；OLD原cuDNN=True仅历史比较|
|cuDNN deterministic|原记录False|不得误写为全局detTrue即此字段True|
|线程|intraop=1、interop=1、OMP=1、MKL=1|禁止同时改线程作workspace对照|
|GPU mapping|CUDA_VISIBLE_DEVICES=0，PCI_BUS_ID，原GPU0 UUID GPU-a78176ed-f97f-c54f-6201-5a01b570f422|未来exclusive/PID/UUID/显存资源门禁尚NOT_RUN|
|离线环境|HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1、PYTHONDONTWRITEBYTECODE=1|不下载新权重、不向历史树写pycache|
|driver/动态库原历史选择|UNKNOWN/NOT_RETAINED|不能以版本字符串或本轮磁盘文件存在推测过去loaded binary|

本轮增加CPU-only物理库候选SHA清单时，只能定义后续绑定，不会变成历史执行证明。实际B0/B1加载路径须未来获授权后观测并核对，尤其pip与conda并存的cuBLAS候选。源、库、设备资格失配必须停止，不能悄悄回退。

## 3. GDN数学、缓存与量化时序

1. `core.configure()`/`core.load()`取原AST：原native架构、补偿`compensated_scan/chunk_binding`仅替换chunk的cumsum；同原FP32 recurrent helper，不能换回旧native chunk当优化。首输入chunk24次，后255×24 recurrent，计时窗口128×24；每次完整模型32层，计249个linear/GEMM模板。
2. 输入cache由本次forward消费；GDN当前更新和native readout产生当前logits；量化条件在forward后对state QDQ并in-place写回，下一token必须消费该存储/字节。prefill t0也QDQ；不能移动写回到更晚、跳过token或复用高精度未量化state。
3. `Observe(on=False)`仍保留24层call/dtype/shape/statefinite，state bool汇总与logits finite后一次主机标量检查。FP前一步previous为空，H时next-consumption pointer/hash按资格模式检查。不能删safety以改善TPS；完整hash/采集只在资格窗执行，不能混进裸TPS。
4. FP路径无Hadamard、无QDQ；H路径在原GDN core前归一化q/k（按原flag）并右乘FP32归一H128、关闭core重复归一化，保留value/readout dtype转换；Hadamard矩阵memoize，不改公式。
5. H+C128原`quantize_c128`：FP32 source、`amax(dim=-2).clamp_min(1e-12)/127`，`round().clamp(-127,127)`，`dequant=codes*scale`，finite后`state.copy_`；24层均写回。资格须保留codes/scales/dequant/pre/post及下一token消费状态，FP通过不自动通过C128。
6. H+Dual-M只读原`dual`和`codec`AST：FP16 row/column scales、Column LS与final re-encode不改；原scale风险门禁、decode/copyback不改。codes/rows/columns/dequant/pre/post、signed-zero差异及next-consumption均独立核验；当前候选`NOT_QUALIFIED_FOR_THIS_CODEC`。
7. **存储披露**：原实现是24层合计48MiB native FP32 recurrent-cache的逐tokenQDQ carrier；编码codes为INT8、Dual-M尺度FP16，但不是persistent packed INT8 CUDA recurrent kernel。优化资格不得把carrier误宣称物理压缩INT8，也不得允许未QDQ高精度状态跨token消费。

## 4. 完整数值接口与拒绝规则

- 可复用：`core.tsha/cache_tensors/cache_hash`、原worker `Observe`、`describe/deep_hash`、`original_c128`源码行trace、`dual(trace=True)`。这些是tensor bytes与结构观察接口，原算法保持不变；资格过程需同时记录shape/dtype/finite及全cache身份，不能只比较最终logits。
- 每token需观察q/k、initial state、updated state、current readout、logits、conv/recurrent state、8路KV；H时追加codes/scales/writeback及下一token消费。B0各两次/B1各两次fresh进程的256输入轨迹先做组内确定性和组间严格字节比较；首次分歧须给token、tensor角色、SHA及合法可观测误差，不放宽阈值。
- 裸TPS固定128warmup+128measurement、边界synchronize；不额外插入hash、Profiler或逐token全局sync，原数学/finite/QDQ本来存在的同步不删。数值资格与TPS必须分开。
- FP、H+C128、H+Dual-M分别资格；没测到就是`NOT_RUN`/`NOT_QUALIFIED_FOR_THIS_CODEC`。本轮没有任何新GPU数值/性能证据，不授予FAST_STRICT_EQUIVALENT或FAST_NEW_PROTOCOL。
- 严格重复性、byte parity、安全性、设备资源、稳定>=预登记收益阈值是不同门禁。短程byte相等不是证明所有32K/81K长度等价，输出相等也不是未初始化读安全证明。

## 5. 当前可用性

已建立可审阅的NEW_STRICT文件/环境/输入/API合同；GPU执行准入、B0/B1字节轨迹、fresh determinism、真实单卡TPS及各Codec适用性均NOT_RUN。原1800秒cuBLAS合同未授权；fill是另一个假设，不能合并预算或关闭开关。`runtime_presets/strict_runtime.sh`由本轮启动器按此合同封装，默认只能CPU校验；未生成可使用的`fast_validated_runtime.sh`。
'''
reuse='''# 历史证据复用与本轮边界

前序根因任务`QWEN_GDN_RUNTIME_THROUGHPUT_ROOT_CAUSE_V1`终态`DIRECTIONALLY_SUPPORTED_ONLY`。本轮先重新计算其全部80个文件SHA/bytes和81条SHA256SUMS（多一条是manifest自身），全部PASS；不是仅阅读终态。原预登记完整raw SHA `404e2b2e07519434e589acfe71a2a1355b224a038d3d8d707f2eb94885ea1884`已精确复制到`evidence/previous/ROOTCAUSE_GPU_DIAGNOSTIC_PREREGISTRATION.json`，没有修改原任务。证明见[contract_prior_evidence_verification.json](analysis/contract_prior_evidence_verification.json)。

## 已闭合事实直接复用

|历史观察|OLD|NEW|资格与限制|
|---|---:|---:|---|
|FP无Profiler TPS，3次中位|13.742036|7.776711|历史同卡/固定输入/128+128短缓存；本轮新测量NOT_RUN|
|H+C128无Profiler TPS，3次中位|9.064610|6.650598|有prefill QDQ/H缓存差异，不是Codec严格单变量|
|mm CPU self，ms/token|7.076412|55.827730|历史instrumented统计，非纯CPU算力或独占墙钟份额|
|mm开始至首次launch，ms/token|6.807047|55.391320|历史trace249ordinal普遍增大；具体native库/调度帧UNKNOWN|
|关联GPU GEMM，ms/token|20.928374|20.939048|同名/同数GPU活动；不能证明全部数学/参数或整机critical path等价|
|kernel/token|2710|3557|额外847已对账：451FP32fill+271BF16fill+2longfill+96finite组件+25boolall+2cat/and|

已有detFalse诊断中位7.996658 TPS，仅比NEW +2.828%；intraop24中位7.655933，−1.553%。这些干预未恢复旧速度、未证明可采用的严格等价优化。NEW的CUBLAS/OMP/interop/safety等在detOFF保持；不能将单开关观察扩大成所有确定性或线程因素已排除。

已有源码/trace共同反对“新增24个逐层主机同步”；两FP都是每token一次合并标量读取。补偿chunk在prefill、计时窗口中零chunk调用，不直接解释稳态降速。额外724fill具有确定性初始化来源机制，但历史fill动态flag未显式记录，也无单变量fill-off因果/完整全写后读证明，不批准全局关闭。

## 来源与复制策略

- 复用前序201+175源/依赖fresh SHA记录：全部376 PASS；本轮没有重复访问全部源或重新解析815MB/978MB原trace。原完整proof作为精确小文件复制，SHA与原seal绑定；这叫复用前次fresh资格，不声称本轮每一源又fresh了一遍。
- 原大trace永久留55；原trace完整SHA已由前序stream核验。本轮复用小归约、kernel delta与callpath，不下载大trace/model、不重新占GPU。
- 原历史任务本地369个资产和Speedup270个资产的SHA验证记录随前序proof保留；本轮重点验证前序root80文件的传递身份，没有无限重复所有底层文件读取。
- 原Strict目前权重4片SHAs来自调和共享checkpoint资格；未来GPU必须fresh全部4片。Tokenizer/config等现行文件另外CPU-only stream SHA供后续同文件绑定；不能追认更早历史全Tokenizer SHA。
- 当前CPU补齐Torch/CUDA/cuBLAS/cuDNN物理库候选SHA时，明确区分安装文件存在与实际loaded选择；历史driver/全so身份仍UNKNOWN。本轮不导入torch、不读NVML、不操作进程。

## 前序预登记与本请求

原完整cuBLAS合同真实可访问，但包括OLD/NEW先裸速和2条数值单位；最新请求强调NEW内部B0/B1、数值资格先行及两边fresh重复性，因此本任务以独立addendum明确覆盖顺序/重复矩阵。原文件raw保持不变，单卡1800秒、1650保护+150清理不扩大。新的严格矩阵及其CPU审查不构成GPU授权。第二fill假设要独立预登记、预算、人类授权，不偷并。

本轮所有候选结果均`NOT_RUN_GPU_NOT_AUTHORIZED`，`NEW_GPUh=0`。历史7.776711/6.650598/6.801314是历史参考，不能放入当前新STRICT/FAST成对测量列冒充新结果。没有Fast validated脚本，也没有已验证可采用的Workspace或fill收益。
'''
readme='''# Runtime presets 使用边界

当前`ALLOW_GPU_DIAG=NO`；状态是CPU准备完成、GPU资格未运行。Strict配置参考[STRICT_RUNTIME_CONTRACT.md](../STRICT_RUNTIME_CONTRACT.md)及机读[STRICT_BASELINE_CONTRACT.json](../analysis/STRICT_BASELINE_CONTRACT.json)。候选`:16:8`尚未通过数值/性能/Codec资格，不是可投入使用的Fast Runtime。

- `strict_runtime.sh`只封装NEW_STRICT：独立Python子进程前设置`:4096:8`、OMP/MKL1、离线模式及bytecode禁写；默认CPU-only资格/帮助，任何GPU执行仍必须本任务人类明确YES、有效冻结授权和所有准入PASS。
- `runtime_identity.json`或对应机读合同绑定实际源/参数；不得悄悄回退。完整weight/tokenizer/依赖身份须按计划核验；未来loaded动态库和GPU实际环境必须记录。
- 未生成`fast_validated_runtime.sh`。也不能把候选配置改名作Fast脚本或直接应用已封存AIME Runtime。
- FP通过不自动通过H+C128或H+Dual-M。原Dual-M当前候选为`NOT_QUALIFIED_FOR_THIS_CODEC`，Fast应拒绝它；Group32、Asymmetric等未测试方法同样不能继承资格。
- 不自动运行AIME、32K Future-KL、训练或长程评测；短资格不保证32K/81K全路径一致。

资格worker接口（只说明程序入口，不提供当前执行授权）：

```text
scripts/cublas_worker.py --root ROOT --unit-id ID --mode qualify|benchmark --condition B0|B1 --codec FP|H_C128 --authorization AUTHORIZATION_JSON
```

正常使用先阅读启动器帮助与冻结授权，不直接调用worker绕过控制器；raw worker也必须自行拒绝未授权/身份失配。没有通过字节等价、fresh重复性、安全/资源和稳定收益全部门禁前，Strict与Fast之间不存在已批准切换。

科研协议选择：Strict用于原协议定义内复现与对照；只有未来`FAST_STRICT_EQUIVALENT_VALIDATED`且注明具体Codec/长度范围时，Fast才可作为相应新实验等价候选。若仅`FAST_NEW_PROTOCOL_REPRODUCIBLE`，必须整套匹配基线在新协议中重建，不能直接与旧Future-KL/AIME作Codec因果比较。

原native recurrent-cache是FP32逐tokenQDQ carrier，不能宣称persistent packedINT8。配置快并不替代量化状态/数值语义资格。
'''
strict=strict.replace('本轮增加CPU-only物理库候选SHA清单时，只能定义后续绑定，不会变成历史执行证明。实际B0/B1加载路径须未来获授权后观测并核对，尤其pip与conda并存的cuBLAS候选。源、库、设备资格失配必须停止，不能悄悄回退。','本轮CPU流式读取20条物理库路径（16个独立物理文件，共3,843,736,483 bytes），对Torch CUDA/CPU/c10、pip cuBLAS/cuDNN/CUDA runtime及conda cuBLAS候选绑定SHA；未导入或下载库。它们只定义后续绑定，不是历史执行证明。pip与conda存在不同cuBLAS副本；实际B0/B1加载路径须未来获授权后观测并核对，同一B0/B1必须绑定相同selected库。缺失conda cuDNN候选如实UNKNOWN，不导致CPU资格失败。源、库、设备资格失配必须停止，不能悄悄回退。')
reuse=reuse.replace('当前CPU补齐Torch/CUDA/cuBLAS/cuDNN物理库候选SHA时，明确区分安装文件存在与实际loaded选择；历史driver/全so身份仍UNKNOWN。','当前CPU已补齐20条Torch/CUDA/cuBLAS/cuDNN物理库路径SHA（16个独立物理文件）；明确区分磁盘文件与实际loaded选择，pip/conda不同cuBLAS副本同时存在，需未来实际mapped路径资格；历史driver/全so身份仍UNKNOWN。')
readme=readme.replace('- `strict_runtime.sh`只封装NEW_STRICT：独立Python子进程前设置`:4096:8`、OMP/MKL1、离线模式及bytecode禁写；默认CPU-only资格/帮助，任何GPU执行仍必须本任务人类明确YES、有效冻结授权和所有准入PASS。','- `strict_runtime.sh`是冻结NEW_STRICT配置与身份包装；只接受`--help`、`--check`（CPU-only本地source manifest检查与不导入torch的package metadata记录）、`--print-env`（打印冻结exports）。它不接受任意命令、不加载模型，也不启动GPU/长任务。预置`:4096:8`、OMP/MKL1、离线模式及bytecode禁写。CPU检查不是GPU数值PASS。')
readme=readme.replace('正常使用先阅读启动器帮助与冻结授权，不直接调用worker绕过控制器；raw worker也必须自行拒绝未授权/身份失配。','GPU入口仅`python scripts/run_cublas_controller.py --run --allow-gpu-diag YES --authorization AUTHORIZATION_JSON`，且必须已有本任务人类明确YES以及所有真实准入PASS；这里的命令说明不是授权。正常使用先阅读启动器帮助与冻结授权，不直接调用worker绕过控制器；raw worker也必须自行拒绝未授权/身份失配。')
for name,text in [('STRICT_RUNTIME_CONTRACT.md',strict),('HISTORICAL_EVIDENCE_REUSE.md',reuse),('runtime_presets/README.md',readme)]:
 libs=[x for x in C.get('prospective_library_bindings',[]) if x['status']=='STABLE_CURRENT_CPU_HASH'];physical={x['resolved_path']:x['bytes'] for x in libs}
 text=text.replace('20条物理库路径（16个独立物理文件，共3,843,736,483 bytes）',f"{len(libs)}条物理库路径（{len(physical)}个独立物理文件，共{sum(physical.values()):,} bytes）")
 text=text.replace('20条Torch/CUDA/cuBLAS/cuDNN物理库路径SHA（16个独立物理文件）',f"{len(libs)}条Torch/CUDA/cuBLAS/cuDNN物理库路径SHA（{len(physical)}个独立物理文件）")
 for rel in ['analysis/STRICT_BASELINE_CONTRACT.json','analysis/contract_prior_evidence_verification.json','../STRICT_RUNTIME_CONTRACT.md','../analysis/STRICT_BASELINE_CONTRACT.json']:
  target=(R/'runtime_presets'/rel).resolve() if rel.startswith('../') else R/rel
  text=text.replace(']('+rel+')',']('+str(target)+')')
 (R/name).write_text(text)
print('written')
