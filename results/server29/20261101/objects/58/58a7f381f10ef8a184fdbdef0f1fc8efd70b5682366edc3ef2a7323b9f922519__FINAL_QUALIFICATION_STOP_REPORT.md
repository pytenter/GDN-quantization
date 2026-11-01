# Ling/KDA V3 最终资格停止报告

本轮真实终态：GPU6完整512×2 PASS；GPU3 fresh-process exact-repeat FAIL。流水线已fail-closed，micro/32K均未启动。

## 五项状态

```json
{
  "ENGINEERING_STATUS": "STOP",
  "NUMERICAL_QUALIFICATION_STATUS": "FAIL",
  "MICROBENCHMARK_STATUS": "NOT_RUN",
  "FORMAL_32K_STATUS": "NOT_RUN",
  "P1_VS_L7_SCIENTIFIC_VERDICT": "NOT_EVALUATED_NO_QUALIFIED_32K_RESULT"
}
```

## 已定位的数值差异

GPU3两次资格canonical的原数值fingerprint分别为：

`37d50bd364da206c76aa782e441e02d456e25df8196c0406f220c022dccc8e3f`

`a480e120dceef18b0e781b3cfe678f0447391b2a8a0e4360c8277bf315a631f3`

原fingerprint不含PID/时间，独立重算与记录一致。FP和两个student的full-logit SHA均不一致。两个student分别有509/512个KL scalar不同，首个差异为零基位置3；最大绝对差分别为1.146123245358467和0.12891923636198044，均在位置218。原数值payload共1072处差异。

这些是资格期失败诊断量，不是正式32K P1/L7效果。原逐token FP logits未归档，无法由SHA还原每token logit误差。具体kernel/硬件根因尚未确定，不能把deterministic=True当作fresh-process确定性已被证明。

## 原状态报告的明确更正

原生V3_FINAL_STATUS.json误记NUMERICAL_QUALIFICATION_STATUS=NOT_RUN：terminal_report先读取缺失的aggregate文件，再写STOP占位。这是状态汇报顺序缺陷，不改变数值失败。原文件与STOP永久保留；以V3_TERMINAL_STATUS_CORRECTION.json追加声明FAIL，未覆盖原证据。原生FINAL_STOP_REPORT.md中对应NOT_RUN及未冻结numerical uncertainty的概述也不能视为有效资格或已执行阈值冻结。

## 十五问

1. V2独占检查的问题：源码把目标GPU compute PID与fuser全部holder合并，仅允许worker自身PID，可能把非compute fd持有者误判为目标计算占用。历史失败瞬间未保存硬件快照，因此不能确认历史具体触发者。

2. V3运行中独占：按物理UUID查询NVML compute context，并绑定PID/starttime/boot_id/UID/NSpid；逐次保存raw查询、fuser与分类。四次512 worker的采样guard均通过。退出后快照不能代替运行时证据。

3. 最小修复是否有效：CPU regression及两卡tiny real CUDA probe PASS；运行时guard通过，V2 worker/runtime及数值engine字节不变。但guard有效不等于整体数值资格通过。

4. 串行还是并行：GPU6 probe→GPU6两个fresh 512→GPU3 probe→GPU3两个fresh 512，全部串行。并行轻量probe/micro/formal均未执行。GPU3启动前短暂外部占用仅等待自然释放，未发信号。

5. 两卡512×2资格：GPU6 PASS；GPU3 FAIL。GPU3两次各自512过程机械门禁及同进程FP replay通过，但fresh-process FP/student logit SHA与数值轨迹不同；整体资格FAIL，不能用各单次canonical的PASS替代。

6. 8448 micro：NOT_RUN。资格失败后未启动。

7. 32K真实物理可行性：NOT_ESTABLISHED。仅512显存记录不能证明8192/32768缓存增长可行；未执行8448或physical resource gate。没有16 GPUh/85%行政上限，也不能凭此宣称物理可行。

8. 4/4 student trajectories：0/4正式32K。资格期的两条512 student不是正式科学单元。

9. 2条same-GPU FP：0/2正式32K FP。资格期512 live FP和同进程replay存在，但GPU3跨fresh process FP字节不一致，不能据此批准32K。

10. 文档860的P1/L7 KL：正式primary[8192,32768)与secondary[2048,32768)均未测得（missing，不是0）。不把512资格KL当作正式KL。

11. 文档5956的P1/L7 KL：正式primary/secondary均未测得（missing，不是0）。资格期已有真实数值差异，仅用于失败诊断，不支持科学排名。

12. 两文档支持P1还是L7：均NOT_EVALUATED；没有合格长期结果。未改变document/pair/rotation或容差，没有重跑凑结果。

13. 独立预测增量证据：本轮没有新增合格32K增量证据；不推翻或扩张既有Phase A/B的有限结论，不宣称统计确认、梯度等价或训练收益。

14. GPUh与显存：原ledger按receipt→canonical计0.26360433081786 GPUh，含启动/loading/JIT/probe/FP。精确process exit时刻未记录；另按下一stage/STOP已确认worker退出计保守上界0.26973963843452 GPUh，避免漏计canonical后的短暂退出开销，CPU与外部等待不计。CUDA allocated峰值16154826240 bytes (15.045354 GiB)，reserved 16282288128 bytes (15.164062 GiB)。两卡分别求和，历史V1/V2成本另列。NVML只提供采样显存，不冒充连续峰值。

15. 是否值得做P1梯度可优化性：本轮不足以支持直接进入梯度验证。下一项有依据的工作是单独定位GPU3 fresh-process确定性问题，需新指令批准；本轮不自动修复/retry、不训练或optimizer.step。

## 复现与封存

测量commit：`31646483521ce6732be80b7cd64675b3a7b0663a`。

原数值engine SHA256：`d448121735fd5ed7eff3a62689d359f17eca4b5467972ff69d299073a2b865c8`。

所有6条launch receipt、STARTED/process binding、canonical+sidecar、raw GPU_EXCLUSIVITY、日志及冻结代码archive均保留。相关非权重输入重新SHA核验；32权重分片仅新鲜metadata核验并复用已有全SHA资格，不重复全部历史深度审计。V1/V2 STOP哈希未变。

追加交付程序仅CPU分析与打包，未修改测量HEAD、engine、worker、容差或冻结科学选择。未生成未执行的科学CSV、FP manifest、32K run manifest或空壳结果。核验所有本轮worker/supervisor自然退出后封存，未使用signal/kill。
