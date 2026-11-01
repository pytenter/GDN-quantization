# QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3 最终报告

## 阶段状态

```json
{
  "NUMERICAL_CLOSURE_STATUS": "PASS",
  "TRACK_B_STATUS": "FAIL_AND_STOP",
  "TRACK_A_STATUS": "NOT_RUN",
  "LONG_HORIZON_STATUS": "NOT_RUN_OR_INCOMPLETE",
  "SURROGATE_DISCOVERY_STATUS": "ENGINEERING_OR_NUMERICAL_STOP"
}
```

## 数值闭环

第一条历史 fixed-order FP32 路线：算子 PASS；8-token 全模型状态相对L2=0.0011951210145，高于原冻结0.001上限，判为数值不合格，证据保留。
第二条预登记 FP32 Neumaier 补偿路线：算子 PASS；2/8/32/64/256-token 全模型共1110个对象与离线数学scan参考逐字节相同；三个独立进程一致。未放宽旧数值上限。
FP64只在离线数学oracle和只读指标归约中使用；正式缓存/scan仍FP32。旧历史native byte parity未恢复，也未被改为PASS。

## 15个问题

1. 原strict CUDA cumsum执行阻断：在新的隔离数值合同下已解决；不宣称旧运行的历史等价恢复。
2. 先复用原fixed-order实现；其全模型资格失败后只使用第二条事前登记的FP32补偿路线，无第三条实现。
3. 新数学、数值、完整模型资格全部PASS，离线scan数学参考与正式candidate有清晰角色区分。
4. 旧145对象：historical32 145/145 MATCH；dev256 12/145 MATCH，历史full-model FAIL保持不变，旧logits最大误差0.375/相对L2约0.0099804173未忽略。
5. 新full-model三进程逐字节一致，覆盖每次prefill及随后两token的全部24层输出/状态/cache、final hidden及logits。
6. B0/B1保留原chunk-to-position collector；独立逐token clean scale轨迹没有冒充原训练teacher。
7. S1/S2 observer：两次fresh process，OFF/ON codes、完整cache、logits、next-token消费和writeback均逐字节一致。
8. Track B：未完成正式全套测量；保留已通过的数值/observer/局部资格证据。
9. Track A：NOT_RUN或未完成；不得填入虚构分数。
10. Matched长期G：NOT_RUN或未完成；没有与旧G无资格配对。
11. S1区分H/C6R/C7：没有完整合格结果，不能判断。
12. S1相对C6 Functional：未形成完整可比较证据。
13. S2相对S1：未形成完整增量证据。
14. 已记录阶段GPUh=0.408894，峰值allocated=17.101GiB，reserved=17.256GiB；保留产物=7.557GiB。详见RESOURCE_LEDGER的测量口径。
15. 独立full-model persistent KL：本轮未授权执行；先处理当前实际STOP并完成合格测量，暂无充分科学依据启动。

## 首个新停止证据

```json
{
  "exception": "RuntimeError",
  "message": "original text/tokenizer first1024 context pairing",
  "retry": false,
  "status": "FAIL_AND_STOP",
  "traceback": "Traceback (most recent call last):\n  File \"/data/zypan/worktrees/qwen-gdn-deterministic-scale-discovery-v3/experiments/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3/scripts/scale_science.py\", line 429, in main\n    elif a.phase=='track_b':track_b(a.gpu,a.doc,a.engineering)\n  File \"/data/zypan/worktrees/qwen-gdn-deterministic-scale-discovery-v3/experiments/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3/scripts/scale_science.py\", line 279, in track_b\n    N.check(tokenizer(Path(d['text_path']).read_text(),add_special_tokens=False).input_ids[:1024]==ids, 'original text/tokenizer first1024 context pairing')\n  File \"/data/zypan/worktrees/qwen-gdn-deterministic-scale-discovery-v3/experiments/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3/scripts/numerical_recovery.py\", line 41, in check\n    raise RuntimeError(why)\nRuntimeError: original text/tokenizer first1024 context pairing\n",
  "utc": "2026-10-12T03:11:55.833789+00:00"
}
```

## 执行边界

未训练C8或新rotation，未执行AIME、自由生成、梯度优化或独立KL。旧实验状态和失败证据均保持不变。未执行的科学CSV不生成空壳。
