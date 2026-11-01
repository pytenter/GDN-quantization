# Qwen3.5-9B STEPQuant@6 与 Hadamard+Dual-M：用户主动停止交付

最终运行状态：**FAIL_CLOSED_STOP**。科学状态：**INCOMPLETE_USER_REQUESTED_SCOPE_CHANGE**。

用户明确改为测试 AIME26 中 Hadamard+Dual-M 的吞吐，并允许停止 STEPQuant。协调器收到 SIGTERM 15 后依冻结清理逻辑结束四个自有 worker；四进程均 returncode=-15、真实进程不存在、NVML 已释放，旧 tmux 会话已结束。本轮停止原因为用户改范围，不能描述成模型工程或数值故障。停止时仅有 **1/120** 条完整生成（STEPQuant seed1，H+Dual-M 为0），尚不满足独立 Frozen V4 评分条件。没有执行评分，没有生成科学结果 CSV，也不把未完成请求记作答错。

原始失败校准的247文件封存及其源码/失败历史永久保留；其 manifest SHA为 `dc0e70df32b2ff64ab6230409657589fdaacad99500e526c8583241d61424e34`。本目录是修复运行的新 STOP 交付，不覆盖该父封存。另行授权的 H 吞吐测试应在独立目录，不能回填本次120生成比较。

## 原请求12问

1. **STEPQuant@6 是否在9B合法移植？** 已完成官方 commit `f2deac58131219e6aecff274f8eabe88e08d2e3c` 的真实32×2048 WikiText非AIME校准、完整9B Plan、CPU参考检查及四卡128-token fresh/写回/消费资格；一条正式12640-token STEP生成正常EOS。最小修复仅放行原生 `use_cache=True` 控制参数并要求 `cu_seqlens=None`，没有改GDN/归一化/统计/codec数学。可以称限定范围的实际移植资格通过，不能宣称完成60条评测或81K数值资格。原失败因缺少该控制kwarg准入，保留不抹除。
2. **Hadamard+Dual-M 是否复现原数学与QDQ？** 原H128、INT8、impact=1、pivot=0、FP16行列scale与原column LS/final re-encode/floor/clipping/rounding路径在CPU及四卡非AIME128-token canary/fresh资格通过，真实Hadamard调用和写回后下一kernel消费已检查。新共同原生90d GDN路径与历史补偿实现、历史prefill QDQ时序不同，因此不宣称历史完整生成逐字节等价。H本轮没有完整AIME生成。
3. **四张3090完整生成多少？** STEPQuant **1条**；Hadamard+Dual-M **0条**。五次请求已发起，其中四次被用户停止（GPU0的H及GPU1–3的STEP），另115条未发起。完整记录在 `formal/worker0/00_STEPQUANT6.json`；canonical payload与文件SHA已独立检查。四卡按固定UUID各单模型、独立cache/Generator运行，不是TP4。
4. **STEPQuant AIME26正确率？** **未知，未评分。** 只有1/60完整，不能给出完整60样本准确率，也不将该单条当成代表性估计。
5. **Hadamard+Dual-M正确率？** **未知，未评分。** 完整0/60。
6. **两个seed各正确多少？** 均未知。只有seed1的一条STEP完整；seed2正式完整0。真实GPU seed1随机Generator已推进并fresh复现，seed2仅冻结参数化采样器和CPU fixture资格，不伪称另有seed2GPU canary。
7. **哪些题两方法不同？** 无完整方法配对，不能计算正确性分歧、配对胜负或主要准确率差。没有查看答案来决定停止、替换或重采样。
8. **长度与截断？** 唯一完整STEP生成12640 token，EOS结束，耗时3288.776626秒，实际3.843374 token/s；不是81920长度截断。全60的均值/中位/p95/EOS/截断率均不可计算。停止前日志H已到6400 token/936.815661秒，其他STEP到14464/15872/15232，均只是最后128-token日志边界，不是完整输出。Frozen V4提取失败数未知，因为未评分。128资格和这些前缀不能当成81K完整性能/数值证明。
9. **scale/pivot/clipping/数值风险？** 有实际floor命中，不能说无floor。STEP128资格每条column_floor=524288，资格记录的受影响energy/relativeMSE最大值为0；decode-fit数值maxabs=0但字节SHA存在差异，未证明一定由signed zero引起。唯一完整STEP记录column_floor=51773441，row floor/ceiling和column ceiling=0，nonfinite=false，未触发冻结风险STOP；全程独立energy/MSE最大标量未另保存，不能沿用128零值覆盖12640或81K。中断请求未达到结束审计，不作全程安全结论。Frozen clipping/rounding规则保留，无新风险阈值修改。
10. **state位宽与成本？** STEP768head的4/6/8/16-bit分别299/170/267/32；32个全局FP16 pivot head，真实data平均6.333333bit（不含metadata），FP16行列scale。每请求24层编码payload10,338,304B，其中integercodes8,912,896B、pivot1,048,576B、scale376,832B；逐层临时编码最大801,280B。H为INT8、无pivot，编码payload12,976,128B（codes12,582,912B+scales393,216B），原helper批量24层。两方法原生persistent state均是FP32承载重构QDQ值，50,331,648B=48MiB，不是物理packed INT8驻留，也没有未量化state shadow绕过反馈。上述编码成本不含共享Plan/索引/Python分配器；CUDA峰值另报。
11. **支持哪个方法？** **没有准确率方向证据。** 比较未完整，主要指标、配对统计与30题cluster不确定性均NOT_RUN。历史52/41/40/33/18等只作历史参考：原生GDN/确定性/prefill时序差异尚不支持当作本轮严格对照。非AIME128资格、合成82293-context显存压力和单条STEP生成不等于全60或81K保证。
12. **对codec-aware rotation训练启示？** 本次只建立有限工程可运行证据和真实存储/吞吐成本，不能凭未完成准确率判断训练收益或选择目标。本轮不训练C8/C6R或新旋转，不租A800，不新增FP/其他基线，也不自动补seed。用户另行授权的H吞吐测试只回答速度问题。

## 资格、资源与停止回执

真实官方Plan SHA：`a0613ebdf249f90f6ef128b283733ca4368baca8fbeef6e09b1f4db25cd0fa79`；statistics SHA：`92522eb402cc84ff0d8f3757bfe9ce57d6a056c2ff958ba82cf8fbcac9d0b35c`。数据65536 token，WikiText revision b08601e04326c79dfdd32d625aee71d232d685c3，seed0、gamma0.25、horizon2048、sample_every8、max_snapshots64，值组32。原冻结设计文件保留“校准前待生成”状态，实际Plan身份在校准回执及正式AUTH中绑定，没有覆盖冻结设计来伪造事前状态。

四卡12个限定门禁PASS。真实82293总context的live-model KV/cat/expanded workspace/native SDPA压力：每卡peak allocated22,289,758,720B、reserved22,334,668,800B、free2,786,787,328B，大于1GiB门槛；这是容量压力而非81K GDN数值轨迹。唯一完整STEP的实际CUDA allocated峰值18,446,313,984B、reserved19,356,712,960B。

固定采样：thinking=true，do_sample=true，temperature1、top_p0.95、top_k20（原tie-aware）、min_p0、presence1.5仅generated output、repetition1，max_new_tokens81920。唯一完整请求Generator状态前后不同；正式采样源与资格相同，没有greedy替代。每方法独立cache，prefill后立即QDQ、decode每token写回。完整STEP12640次QDQ、303360层写回、303336次下一kernel消费检查；formal关闭逐tokenCPU tensor hash，仅128资格有完整字节trace。

最终账本累计 **6.257937735323 GPUh**，包含父失败0.006847216835 GPUh；本修复运行实际 **6.251090518489 GPUh**，正式阶段四卡合计 **4.737968294786 GPUh**。冻结账本 `this_run_GPU_hours=0` 是未维护初值，独立 reconciliation按累计减父失败计算，没有修改原账本/回执SHA。

本STOP交付保存所有停止时小文件、原子完整raw、attempt、真实进度日志、源码/身份/限定资格及独立审计。未完成请求不伪造raw completion；`GENERATION_RESULTS.jsonl/PER_PROBLEM_RESULTS.csv/PER_SEED_RESULTS.csv/ACCURACY_SUMMARY.json`不创建。模型、Plan和3GiB statistics保留在服务器原位，未下载大型tensor或删除任何历史资产。`STOP_STATE_SNAPSHOT.json`固定用户停止后的身份/释放证据；mutable ROOT/HANDOFF不纳入最终seal。最终独立source/tensor复核及小文件逐SHA/bytes核验见随附报告和manifest。

最终独立只读复核 `independent_review/USER_STOP_FINAL_IDENTITY_VERIFICATION.json` SHA `bb6a83237740358d7e34c50afed77147b2da90e6775c369650a665704fd7316c`：214份冻结任务源与2952份外部依赖逐内容SHA/bytes全匹配；Plan1,209,542B和statistics3,222,054,270B远端稳定stat流式freshSHA全匹配；原父247封存全247份fresh校验通过，旧coordinator及4worker均真实不存在/NVML释放。模型17份权重/tokenizer本次引用先前全内容freshSHA证明，未宣称此次重新流式读取全部权重。该PASS仅为身份/释放核验，不改变用户停止且未评分的科学状态。
