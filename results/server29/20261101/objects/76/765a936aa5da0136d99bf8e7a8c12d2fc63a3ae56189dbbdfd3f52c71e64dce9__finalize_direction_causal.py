#!/usr/bin/env python3
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path("/data/zypan/experiment_runs/LING_KDA_LONG_HORIZON_DIRECTION_CAUSAL_V1")

docs = [
    {
        "document": "pg19-test-37702", "role": "development",
        "gpu": "GPU-de53f243-b271-521c-d875-d96611ed8574", "gpu_index": 6,
        "run": "pg19-test-37702__gpu6__development__formal_v1",
        "complete_sha": "990e59fc4d7301f40fdd0a117bac230615f7306633c54546bb08d9909b6bc431",
        "phase_sha": "3758e9add1f8bafb551bc09bb721235010baf4fa13963d666c38b07ebe971b0b",
        "hours": 3.42579116874271, "elapsed": 12332.848207473755,
        "D": 70.26856830568101, "Dpos": 90.84905121443126, "Dneg": 49.688085396930774,
        "visible_endpoint": 26.71466064453125, "low_endpoint": 68.00331115722656,
        "sensitivity_ratio": 14088376.173211796,
        "branches": {
            "NATURAL": [0.09319955773054103, 715.7726033705551, 73.20488739013672, 2.805407805851193e-07, 100.57564544677734, 127.79819040724438],
            "VISIBLE_POS": [0.09268116332333456, 711.7913343232094, 73.20490264892578, 0.007256275860058236, 26.530532836914062, 83.43354538848142],
            "VISIBLE_NEG": [0.0870710933836997, 668.7059971868136, 73.20489501953125, 0.007256290855225028, 26.898788452148438, 83.31442084987683],
            "LOW_POS": [0.08085185977978883, 620.9422831087782, 73.20490264892578, 5.14459220926752e-10, 71.55559539794922, 106.66665872147192],
            "LOW_NEG": [0.080601290597641, 619.0179117898829, 73.20486450195312, 5.13650036297515e-10, 64.4510269165039, 101.91133012556399],
        },
    },
    {
        "document": "pg19-test-28988", "role": "validation",
        "gpu": "GPU-de53f243-b271-521c-d875-d96611ed8574", "gpu_index": 6,
        "run": "pg19-test-28988__gpu6__validation__formal_v1",
        "complete_sha": "34df8bf6b142ef1c1dc18d2f969682556b8cbf49468e729b6837437af5363bbc",
        "phase_sha": "2948769cb0a9d731945fbf7be4c41dcb8cd4f77a418842fccdfd5b6c04c7ce8e5",
        "hours": 3.222493746744262, "elapsed": 11600.977488279343,
        "D": 141.17580235871424, "Dpos": 96.59369978479526, "Dneg": 185.7579049326332,
        "visible_endpoint": 30.568222999572754, "low_endpoint": 87.63090515136719,
        "sensitivity_ratio": 8045351.78827388,
        "branches": {
            "NATURAL": [0.10092891232995059, 775.1340466940205, 89.2534408569336, 5.099996398930911e-07, 117.08758544921875, 136.5891734108116],
            "VISIBLE_POS": [0.10010909294947776, 768.8378338519892, 89.25347137451172, 0.007051688379088495, 29.96723175048828, 76.04837429406578],
            "VISIBLE_NEG": [0.11117381235090296, 853.8148788549347, 89.25345611572266, 0.007051780622376237, 31.169214248657227, 76.69161200879965],
            "LOW_POS": [0.08753178828999922, 672.244134067194, 89.25343322753906, 8.754367888791655e-10, 77.34040069580078, 104.39429578885606],
            "LOW_NEG": [0.08698658514613301, 668.0569739223015, 89.25341033935547, 8.755591487280093e-10, 97.9214096069336, 120.3066825498377],
        },
    },
    {
        "document": "pg19-test-26493", "role": "validation",
        "gpu": "GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae", "gpu_index": 3,
        "run": "pg19-test-26493__gpu3__validation__formal_v1",
        "complete_sha": "4d5f95167883fef6a0a54e4dda2a00e0da3c7c58322ea9103f50e0bac9303dc7",
        "phase_sha": "6a664b163a96a025e26104dccc3e979352470a11ce415e030e224413900d09efe",
        "hours": 3.4807790553569795, "elapsed": 12530.804599285126,
        "D": 52.38295468272747, "Dpos": 73.88568192486252, "Dneg": 30.880227440592535,
        "visible_endpoint": 27.262639045715332, "low_endpoint": 84.2618637084961,
        "sensitivity_ratio": 11206070.989388932,
        "branches": {
            "NATURAL": [0.08033113490355187, 616.9431160592784, 81.990966796875, 3.5206198728242815e-07, 97.59138488769531, 120.6612243389647],
            "VISIBLE_POS": [0.0805198765221235, 618.3926516899085, 81.99100494384766, 0.0071610925405805334, 27.10887908935547, 76.15400576209296],
            "VISIBLE_NEG": [0.07729000336428686, 593.587225837723, 81.990966796875, 0.007161058613148827, 27.416399002075195, 76.2485960199307],
            "LOW_POS": [0.07089934502149035, 544.5069697650459, 81.99097442626953, 6.375905311122402e-10, 74.10528564453125, 102.56561039336941],
            "LOW_NEG": [0.07326914041629304, 562.7069983971305, 81.9909439086914, 6.384801831721021e-10, 94.41844177246094, 118.00700818947425],
        },
    },
]

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def append_final(name, heading, body):
    p = ROOT / name
    text = p.read_text()
    marker = f"\n## {heading}\n"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n"
    p.write_text(text.rstrip() + marker + "\n" + body.strip() + "\n")

for d in docs:
    c = ROOT / "runs" / d["run"] / "COMPLETE.json"
    if sha(c) != d["complete_sha"]:
        raise SystemExit(f"COMPLETE hash mismatch: {c}")
    parsed = json.loads(c.read_text())
    if parsed.get("gate") != "PASS":
        raise SystemExit(f"formal gate not PASS: {c}")

append_final("EXPERIMENT_PLAN.md", "Final execution status", """
The frozen protocol executed without changing checkpoint, layer, heads, directions, metrics, or gate thresholds. Development completed first; after actual-cost re-projection, both prospective validations were launched in the frozen order on their fixed GPUs and ran concurrently. All three formal runs completed with gate `PASS`; no formal run was retried, migrated, split, or overwritten. Attempt04/05 history and their canonical/FAIL evidence remained outside this experiment and unmodified.

Final status: **COMPLETE_AND_VALIDATED**. Scientific decision: `PROCEED_TO_SHORT_HORIZON_SURROGATE_VALIDATION`. This authorizes only validation of candidate training-time surrogates; it does not authorize new rotation training.
""")

audit_lines = [
    "All three formal runs preserved the frozen physical-coordinate intervention and original L6 rotated-QDQ/writeback path. The post-QDQ norm, signed-direction, sensitivity-separation, finite-value, branch-identity, and fork-parity gates all passed.",
    "",
    "| Document | Natural target norm | Visible/low sensitivity ratio | Mean visible endpoint target norm | Mean low endpoint target norm |",
    "|---|---:|---:|---:|---:|",
]
for d in docs:
    audit_lines.append(f"| {d['document']} | {d['branches']['NATURAL'][2]:.6f} | {d['sensitivity_ratio']:.3f}x | {d['visible_endpoint']:.6f} | {d['low_endpoint']:.6f} |")
audit_lines += [
    "",
    "The visible intervention produced *smaller*, not larger, endpoint target-state error than LOW on every document while still producing larger Future-KL AUC. Thus larger endpoint error magnitude cannot explain the observed contrast. The permitted claim remains trajectory-level: initial error direction changes the subsequent recurrent/quantized trajectory and functional damage. These runs do not isolate a readout-only direct effect from mediated recurrent and quantizer feedback.",
]
append_final("STATE_AND_READOUT_OPERATOR_AUDIT.md", "Formal outcome audit", "\n".join(audit_lines))

parity = """Formal checkpoint-24576 parity passed independently for all three documents.

| Document | Cache tensors | Initial cache | Alias pointers | 8-step logits | Final cache | Gate |
|---|---:|---|---:|---|---|---|
| pg19-test-37702 | 84 | bitwise equal, max diff 0 | 0 | bitwise equal, max diff 0 | bitwise equal, max diff 0 | PASS |
| pg19-test-28988 | 84 | bitwise equal, max diff 0 | 0 | bitwise equal, max diff 0 | bitwise equal, max diff 0 | PASS |
| pg19-test-26493 | 84 | bitwise equal, max diff 0 | 0 | bitwise equal, max diff 0 | bitwise equal, max diff 0 | PASS |

Each formal branch used its same-run FP reference chunks. Every formal run produced 60 hashed FP logit chunks (7,680 scored rows) and two hashed cache snapshots; the sidecar size/SHA checks passed."""
append_final("FORK_PARITY_REPORT.md", "Formal parity results", parity)

intervention = """All formal engineering gates passed on the frozen 24K checkpoint.

| Document | Role | D | D(+) | D(-) | Norm gate | Direction gate | Sensitivity gate | Fork gate |
|---|---|---:|---:|---:|---|---|---|---|
"""
for d in docs:
    intervention += f"| {d['document']} | {d['role']} | {d['D']:.6f} | {d['Dpos']:.6f} | {d['Dneg']:.6f} | PASS | PASS | PASS | PASS |\n"
intervention += "\nBoth sign-specific contrasts were positive on development and on both prospective validation documents. No outcome-dependent branch, threshold, direction, checkpoint, or document choice was made."
append_final("DIRECTION_INTERVENTION_TESTS.md", "Formal intervention results", intervention)

csv_path = ROOT / "CAUSAL_BRANCH_RESULTS.csv"
fields = ["document","evidence_role","gpu_uuid","checkpoint","calibration_start","calibration_end","evaluation_start","evaluation_end","branch","sign","future_kl_mean","future_kl_auc","initial_error_norm","initial_readout_sensitivity_per_norm2","endpoint_target_error_norm","endpoint_global_kda_error_norm","all_finite","direction_gate","fork_gate"]
with csv_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for d in docs:
        for branch, vals in d["branches"].items():
            sign = "0" if branch == "NATURAL" else ("+" if branch.endswith("POS") else "-")
            w.writerow(dict(document=d["document"], evidence_role=d["role"], gpu_uuid=d["gpu"], checkpoint=24576, calibration_start=24576, calibration_end=25088, evaluation_start=25088, evaluation_end=32768, branch=branch, sign=sign, future_kl_mean=vals[0], future_kl_auc=vals[1], initial_error_norm=vals[2], initial_readout_sensitivity_per_norm2=vals[3], endpoint_target_error_norm=vals[4], endpoint_global_kda_error_norm=vals[5], all_finite="PASS", direction_gate="PASS", fork_gate="PASS"))

formal_hours = sum(d["hours"] for d in docs)
micro_pass = 0.008230973680814107
# Failed mechanical probes are charged conservatively by STARTED-to-failure wall clock.
# Attempt02 includes the entire disconnected-client interval, deliberately overcharging possible GPU use.
failed_upper = (15.6157 + 1662.9448 + 14.6028 + 19.8252) / 3600.0
ledger = {
    "task": "LING_KDA_LONG_HORIZON_DIRECTION_CAUSAL_V1",
    "budget_gpu_hours": 16.0,
    "accounting_policy": "completed runs use reported GPU hours; failed mechanical probes use STARTED-to-failure wall-clock upper bounds, including the full attempt02 disconnect interval",
    "entries": [
        {"label":"micro_attempt01","role":"MECHANICAL_ONLY_ENGINEERING_FAIL","gpu":6,"gpu_hours_upper_bound":15.6157/3600},
        {"label":"micro_attempt02","role":"MECHANICAL_ONLY_CLIENT_SESSION_DISCONNECT","gpu":6,"gpu_hours_upper_bound":1662.9448/3600},
        {"label":"micro_attempt03","role":"MECHANICAL_ONLY_ENGINEERING_FAIL","gpu":6,"gpu_hours_upper_bound":14.6028/3600},
        {"label":"micro_attempt04","role":"MECHANICAL_ONLY_ENGINEERING_FAIL","gpu":6,"gpu_hours_upper_bound":19.8252/3600},
        {"label":"micro_attempt05","role":"MECHANICAL_ONLY_PASS","gpu":6,"gpu_hours":micro_pass},
    ] + [
        {"label":d["document"],"role":d["role"].upper(),"gpu":d["gpu_index"],"gpu_uuid":d["gpu"],"elapsed_seconds":d["elapsed"],"gpu_hours":d["hours"],"status":"COMPLETE","gate":"PASS","complete_sha256":d["complete_sha"]}
        for d in docs
    ],
    "formal_gpu_hours": formal_hours,
    "completed_gpu_hours_including_micro_pass": formal_hours + micro_pass,
    "failed_probe_gpu_hours_conservative_upper_bound": failed_upper,
    "conservative_total_gpu_hours_upper_bound": formal_hours + micro_pass + failed_upper,
    "remaining_gpu_hours_lower_bound": 16.0 - (formal_hours + micro_pass + failed_upper),
    "budget_gate": "PASS",
    "budget_decision": "ALL_FROZEN_FORMAL_RUNS_COMPLETED_WITHIN_16_GPU_HOURS",
}
(ROOT / "GPU_COST_LEDGER.json").write_text(json.dumps(ledger, indent=2) + "\n")

report = f"""# Causal Mechanism Report

Status: **COMPLETE_AND_VALIDATED**.

## Result

The frozen directional intervention produced a positive primary contrast on development and replicated prospectively on both validation documents:

| Document | Role | GPU | D: visible minus low Future-KL AUC | D(+) | D(-) |
|---|---|---|---:|---:|---:|
| pg19-test-37702 | development | GPU6 | 70.268568 | 90.849051 | 49.688085 |
| pg19-test-28988 | validation | GPU6 | 141.175802 | 96.593700 | 185.757905 |
| pg19-test-26493 | validation | GPU3 | 52.382955 | 73.885682 | 30.880227 |

Every document passed fork parity, initial post-QDQ magnitude matching, signed-direction alignment, visible/low sensitivity separation, branch identity, finite-value, and sidecar-integrity gates. Both signs were positive in every document; the finding is therefore not carried by a single sign choice.

## Causal interpretation and boundary

At matched initial post-QDQ error magnitude, alignment with the independently frozen future-query-sensitive direction caused greater long-horizon Future-KL AUC than alignment with the low-sensitivity direction. This is a replicated pilot causal signal that initial error direction matters for downstream functional damage.

The effect is not explained by larger final propagated error magnitude: mean endpoint target-state norms were visible/low 26.715/68.003, 30.568/87.631, and 27.263/84.262, respectively. However, the branch intervention changes the recurrent state trajectory, and every later step includes the real L6 quantizer/writeback. Therefore the evidence supports a direction-mediated trajectory effect, not a claim that direct readout visibility is the sole mediator or that propagation and quantizer feedback are irrelevant.

This is a three-document mechanistic pilot, not a population-level statistical estimate. It does not validate a training-time surrogate, authorize a new rotation, or establish generality beyond the frozen L6 setting.

## Integrity and budget

- Frozen runner SHA256: `fb49e377b6b2786deedd694c4489115f08d4cb0bb95cb7f45a93b0a44914ed20`.
- Frozen L6 SHA256: `51a532aad371e9af1d65ed2d2d5d0c41b1369f0635ba2c11adb713283038e964`.
- Formal GPU-hours: {formal_hours:.12f}.
- Completed GPU-hours including the passing micro: {formal_hours + micro_pass:.12f}.
- Conservative total including upper-bounded failed probes: {formal_hours + micro_pass + failed_upper:.12f} / 16.0 (PASS).
- Historical micro attempts 01–04 remain engineering failures/disconnect evidence and were not retroactively relabeled. Attempt04/05 experiment history remained isolated and unmodified.

## Scientific conclusion

`DIRECTIONAL_CAUSAL_SIGNAL_REPLICATED_IN_PILOT`.
"""
(ROOT / "CAUSAL_MECHANISM_REPORT.md").write_text(report)

decision = """# Next Rotation Objective Decision

Status: **FINAL**.

## Decision

`PROCEED_TO_SHORT_HORIZON_SURROGATE_VALIDATION`

## Why

The primary visible-minus-low Future-KL AUC contrast was positive in development and both prospective validations, for both intervention signs, with all frozen mechanical and integrity gates passing. The result also survives the endpoint-magnitude sanity check in the conservative direction: visible branches ended with substantially smaller target-state error norms than LOW branches.

The evidence is strong enough to stop asking whether direction can matter in this pilot and start asking whether an outcome-free, training-time-available short-horizon proxy can rank harmful directions. It is not yet evidence to train a new rotation.

## Minimal next validation

Using the existing frozen artifacts and these three documents, pre-register a compact candidate set of local/short-horizon query-readout proxies and test whether their direction ranking agrees with the frozen long-horizon causal `D`/branch ranking. Keep long-horizon outcomes blinded until proxy definitions and scoring rules are frozen. Require prospective document-level agreement and retain propagation diagnostics as covariates; do not optimize a new rotation in that validation.

If later work needs to distinguish direct readout mediation from recurrent/quantizer feedback, run a separately frozen endpoint-state-controlled mediation experiment. That is optional follow-up evidence, not a prerequisite for the short-horizon surrogate validation selected here.

No L9, AIME, free generation, Qwen work, old 48/128 recovery, or new rotation training is authorized by this decision.
"""
(ROOT / "NEXT_ROTATION_OBJECTIVE_DECISION.md").write_text(decision)

artifact_names = [
    "EXPERIMENT_PLAN.md", "STATE_AND_READOUT_OPERATOR_AUDIT.md", "FORK_PARITY_REPORT.md",
    "DIRECTION_INTERVENTION_TESTS.md", "CAUSAL_BRANCH_RESULTS.csv", "GPU_COST_LEDGER.json",
    "CAUSAL_MECHANISM_REPORT.md", "NEXT_ROTATION_OBJECTIVE_DECISION.md",
    "FROZEN_CONFIG.json", "PREMEASUREMENT_FREEZE.json", "SOURCE_IDENTITIES.json",
    "MODEL_WEIGHT_SHA256.txt", "scripts/run_direction_causal.py",
]
manifest = []
for name in artifact_names:
    p = ROOT / name
    manifest.append(f"{sha(p)}  {name}")
for d in docs:
    manifest.append(f"{d['complete_sha']}  runs/{d['run']}/COMPLETE.json")
(ROOT / "FINAL_ARTIFACT_SHA256.txt").write_text("\n".join(manifest) + "\n")

status = {
    "task": "LING_KDA_LONG_HORIZON_DIRECTION_CAUSAL_V1",
    "status": "COMPLETE_AND_VALIDATED",
    "scientific_conclusion": "DIRECTIONAL_CAUSAL_SIGNAL_REPLICATED_IN_PILOT",
    "decision": "PROCEED_TO_SHORT_HORIZON_SURROGATE_VALIDATION",
    "formal_documents_complete": 3,
    "all_formal_gates": "PASS",
    "formal_gpu_hours": formal_hours,
    "conservative_total_gpu_hours_upper_bound": formal_hours + micro_pass + failed_upper,
    "budget_gpu_hours": 16.0,
    "artifact_manifest_sha256": sha(ROOT / "FINAL_ARTIFACT_SHA256.txt"),
}
(ROOT / "FINAL_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
print(json.dumps(status, indent=2))
