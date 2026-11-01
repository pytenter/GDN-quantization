"""Chinese handoff of already frozen/validated results; no metric selection."""
import csv
import json
from pathlib import Path
from run_short_horizon_proxy_v2 import ROOT, sha256, atomic_json

assert json.loads((ROOT / "FINAL_VERIFICATION.json").read_text())["status"] == "PASS"
analysis = json.loads((ROOT / "ANALYSIS_RESULTS.json").read_text())
decision = json.loads((ROOT / "NEXT_EXPERIMENT_DECISION.json").read_text())
observed = json.loads((ROOT / "OBSERVED_PROCESS_COST_LEDGER.json").read_text())
ranks = list(csv.DictReader((ROOT / "ALL_ROTATION_RANKS.csv").open()))
delivery = "C:/Users/admin/Desktop/操作系统实验/lab2/ling_short_surrogate/results_resume_v2"
messages = {
    "SHORT_HORIZON_READOUT_SURROGATE_SUPPORTED_FOR_VALIDATION": "有候选满足冻结的自然排序及增量判据；仅支持另行独立验证，不支持直接开始训练。",
    "CHEAPER_PROXY_PREFERRED": "较便宜的 P2 满足冻结规则，优先于更复杂的 P3；尚未验证可训练性。",
    "SHORT_HORIZON_READOUT_SIGNAL_NOT_STABLE": "没有 readout 候选同时满足冻结的稳定性与全部 baseline 增量要求；不建议据此推进新的 rotation objective。正相关本身不等于增量价值。",
    "INSUFFICIENT_DATA_OR_RUNTIME_BUDGET": "资料或预算不足，只能交付实际完成的部分结果。"
}
text = f"""# 本轮结果汇总

最终科研状态：`{decision['status']}`。

{messages[decision['status']]}

实际完成：两篇文档 × 16 个自然 rotation，共 32 个 rotation measurements；每个覆盖起点 256/1024、H32/H128、18 个 KDA 层和全部 16 heads。CSV 含 64 个原始窗口行及 32 个聚合行。四个旧 anchor panels 只是两篇文档的两种目标，不是 64 篇独立样本。

新增实测进程 GPU 消耗上界约 **{observed['total_observed_gpu_hours_upper_bound']:.6f} GPU-hours**，包含旧失败 micro、本轮重复 micro、加载、FP reference、计算与清理。预算控制器更保守的口径是 **{decision['gpu_hours']:.6f}/4.0 GPU-hours**；其顺序等待会高估较早结束的 GPU 的用时。两张卡时间求和，不是墙钟时间。

## 主排名结果

表格单元为 Spearman rho / pairwise concordance；分数越高表示预测损伤越大。

| 指标 | 37702 Primary | 37702 Secondary | 5396 Primary | 5396 Secondary |
|---|---:|---:|---:|---:|
"""
for metric in ["P1","P2_W128","P3_H32","P3_H128","B0","B1_H32","B1_H128","B2_H32","B2_H128"]:
    text += "| " + metric + " | " + " | ".join(f"{p['statistics'][metric]['spearman_rho']:.4f} / {p['statistics'][metric]['concordance']:.4f}" for p in analysis["panels"]) + " |\n"
text += "\n## H / L6 / L7 的自然排序检查\n\n下表 rank 在完整 16-condition panel 内计算，低 rank 为低损伤；不是事后重新只在三个条件内拟合。\n\n| 文档/anchor | rotation | Future-KL rank | B0 rank | P2 rank | P3_H32 rank |\n|---|---|---:|---:|---:|---:|\n"
for panel in analysis["panels"]:
    for condition in ["H","L6","L7"]:
        values = {}
        target_rank = None
        for metric in ["B0","P2_W128","P3_H32"]:
            row = next(x for x in ranks if x["document"]==panel["document"] and x["anchor_role"]==panel["anchor_role"] and x["condition"]==condition and x["metric"]==metric)
            values[metric] = float(row["proxy_rank"])
            target_rank = float(row["target_rank"])
        text += f"| {panel['document']}/{panel['anchor_role']} | {condition} | {target_rank:g} | {values['B0']:g} | {values['P2_W128']:g} | {values['P3_H32']:g} |\n"
text += "\n## 判据、反例与边界\n\n"
for metric, value in analysis["incremental_tests"].items():
    text += f"- {metric}：四面板正相关={analysis['stable_positive_ranking'][metric]}；同时赢过每个 baseline 的面板数={value['beats_each_baseline_panels']}；控制 JPROP 后正 rank residual 的面板数={value['positive_residual_panels']}/4；增量判据通过={value['passed']}。\n"
text += f"""

所有自然 pair 排名反例、B0 相差不超过 10% 的 contrasts 及 per-rotation ranks 均保留在 CSV，没有删除不利结果或改窗口/层。自然方向区分的结论是 `{analysis['natural_directional_discrimination']}`；不使用此前人工 VISIBLE/LOW 的极端比值替代自然验证。

重要限制：新代理分数是 driver580.178.04，旧标签是580.173.02，只能作探索性跨 runtime 关联。旧 Future-KL 实际是自然量化前缀到 anchor 后的 pulse continuation，并在后续不再 QDQ；不能冒充“持续量化到32K”的直接验证。两个限制都不能通过本轮相关性消除。

可训练性尚未 PASS：当前 evaluator 为 inference-only，真实历史 K3 helper 会 detach error。数学目标可讨论 Cayley/STE，但要另行保持前向语义并检查梯度、BPTT 和 backward 显存。本轮未训练、未 optimizer.step、未新增 checkpoint。B2 为短程 full-logit KL 对照，不是原 L7 的 local o_proj relative-MSE loss 的逐字复刻。

未来设计仅为条件性规划：两篇未参与 selection 的新文档，每篇四个 calibration-only 选出的自然条件，同一 runtime、同 UUID 的 FP pairing，真正 every-token QDQ 持续到32K。按短前缀吞吐外推约 {decision['prospective_plan_only']['gpu_hours_planning_estimate']:.3f} GPU-hours，未运行也未获本轮执行授权。

## 七项指定交付

"""
for filename in ["SURROGATE_PROTOCOL.md", "TRAIN_TIME_INFORMATION_AUDIT.md", "SHORT_HORIZON_PROXY_SCORES.csv",
                 "NATURAL_ROTATION_RANKING_REPORT.md", "COST_AND_TRAINABILITY_AUDIT.md",
                 "NEXT_EXPERIMENT_DECISION.json", "REPRODUCIBILITY_REPORT.md"]:
    text += f"- [{filename}]({delivery}/{filename})\n"
text += f"\n独立 NumPy 复核、所有冻结文件哈希、聚合行/原始行对应关系和预算检查均通过：[FINAL_VERIFICATION.json]({delivery}/FINAL_VERIFICATION.json)。数学坐标说明见 KDA_MATHEMATICAL_ALIGNMENT.md。历史 Attempt04/05、80个 canonical 及其 FAIL 均未改变。\n"
with (ROOT / "RESULTS_SUMMARY_ZH.md").open("x", encoding="utf-8") as f:
    f.write(text)
paths = sorted(p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in str(p) and p.suffix != ".log" and p.name != "DELIVERY_SHA256SUMS.txt")
with (ROOT / "DELIVERY_SHA256SUMS.txt").open("x") as f:
    f.write("".join(f"{sha256(p)}  {p.relative_to(ROOT)}\n" for p in paths))
print("CHINESE_RESULT_HANDOFF_READY")
