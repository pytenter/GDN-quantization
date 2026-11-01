"""Bind already observed qualification evidence; never run a model or alter science sources."""
import datetime
import hashlib
import json
from pathlib import Path

B = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((B / name).read_text())


def sha(name):
    return hashlib.sha256((B / name).read_bytes()).hexdigest()


def publish(name, data):
    payload = (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    target = B / name
    if target.exists():
        raise RuntimeError("Refuse replacing an existing admission artifact: " + name)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def main():
    review = read("independent_review/FORMAL_ADMISSION_REVIEW.json")
    assert review["status"].startswith("PASS"), review
    prereg = read("PREREGISTRATION.json")
    ledger = read("independent_review/PRE_FORMAL_LEDGER_SNAPSHOT.json")
    assert ledger["all_GPUs_released"] and ledger["status"] == "STAGE_COMPLETE_RESOURCE"
    assert not (B / "FIRST_FAILURE.json").exists()
    for mode in ("calibration", "canary", "fresh", "resource"):
        assert read(f"execution/{mode}/RESULT.json")["status"] == "PASS"
    for name in ("CALIBRATION_PLAN", "CANARY_STAGE", "FRESH_STAGE", "RESOURCE_STAGE"):
        assert read(f"independent_review/{name}_INDEPENDENT_REVIEW.json")["status"] == "PASS"
    assert read("PRE_FORMAL_RESOURCE_PREDICTION.json")["status"] == "PASS"
    task, run = prereg["TASK_ID"], read("ROOT.json")["RUN_ID"]
    canary = read("independent_review/CANARY_STAGE_INDEPENDENT_REVIEW.json")
    assert len(canary["actual_record_metadata"]) == 8
    assert all(r["output_tokens"] == 128 for r in canary["actual_record_metadata"])
    assert all(r["next_consumption_checks"] == 3048 for r in canary["actual_record_metadata"])
    plan = read("calibration/PLAN_CPU_QUALIFICATION.json")
    assert plan["status"] == "PASS"
    assert plan["plan_sha256"] == "a0613ebdf249f90f6ef128b283733ca4368baca8fbeef6e09b1f4db25cd0fa79"

    def gate(number, name, evidence, scope):
        return dict(number=number, name=name, status="PASS", evidence={p: sha(p) for p in evidence}, scope=scope)

    cr = "independent_review/CANARY_STAGE_INDEPENDENT_REVIEW.json"
    fr = "independent_review/FRESH_STAGE_INDEPENDENT_REVIEW.json"
    rr = "independent_review/RESOURCE_STAGE_INDEPENDENT_REVIEW.json"
    gates = [
        gate(1, "Official STEPQuant codec reference parity and actual full9B Plan", ["calibration/PLAN_CPU_QUALIFICATION.json", "independent_review/CALIBRATION_PLAN_INDEPENDENT_REVIEW.json", cr],
             "Actual32x2048 nonAIME calibration; official f2deac58131219e6aecff274f8eabe88e08d2e3c fullPlan24layers/768heads. ActualGPU codes/scales/pivots exact; decode uses pre-frozen official atol=rtol1e-6, not byte-equivalence to fit_state."),
        gate(2, "Original H128 and Dual-M reference parity", ["RUNTIME_CPU_CLOSURE_FINAL.json", "H_DUAL_M_IDENTITY.json", cr],
             "Original columnLS/finalencode/FP16floor/clip/round codec, H key-coordinate path. Actual GPU reference parity and3072 H calls per128tokens. Historical compensated whole-model path is not claimed identical."),
        gate(3, "Common original native GDN integration", ["RUNTIME_BINDING_CONTRACT.md", cr, fr],
             "Both use unchanged original90d native chunk/recurrent fallback and same BF16 model; actual closure metadata and traced prefill/decode boundaries verified. No alternate recurrence or unquantized state shadow."),
        gate(4, "Actual per-token QDQ and writeback", [cr, fr],
             "Each actual128-token record has128QDQcalls and3072layerwritebacks, immediate postprefill plus127decode updates; same QDQ source used formal."),
        gate(5, "Next native kernel consumes written state", [cr, fr],
             "3048 actual perlayer transitions per128record byte-SHA checked, decoded/post/cache identical. Formal retains object/storage/availableversion/count checks, not per-token CPU tensor hashing."),
        gate(6, "FP16 scales and pivots safety", [cr, fr],
             "Observed finite positive FP16 scales/pivots and zero affected-energy/relativeMSE maxima in prefix. STEP columnfloor524288 per128record retained; no no-floor claim. Formal source keeps finite checks and frozen perlayer/head FP16 risk STOP (affectedenergy>=1e-4 and relativeMSE>1e-2), not a proven longhorizon guarantee."),
        gate(7, "128-token fresh-process numeric qualification", [fr],
             "All4GPUs x bothmethods x two independent processes; same-GPU entire128tokens, initial/finalGenerator, sampling/H/forward/QDQ trace exact. PREFIX_ONLY; no claim81920numericproof or universal CUDA determinism."),
        gate(8, "Effective original stochastic sampler and seed", ["protocol_recovery/CPU_PROTOCOL_QUALIFICATION.json", cr, fr, "FROZEN_UNITS.json"],
             "RealGPU seed1Generator advances and same-GPU fresh repeats exactly; original samplerAST and CPU seed1vs2/non-greedy fixture pass. Seed2 GPU canary was NOT independently run; formal fixed units use the same parameterized manual_seed(seed) and independentGenerator. Both seeds must later pass complete-record checks."),
        gate(9, "81920 configuration allowed", ["PREREGISTRATION.json", "MODEL_CONFIG_READONLY.json", rr],
             "Fixed81920 newtokens + maximum373prompt =82293, below model262144 configured positions; actual longKV/nativeSDPA capacity pressure performed. Long recurrent trajectory numeric status NOT_TESTED_NEW_METHODS, numerical qualification PREFIX_ONLY128; no guaranteed completion inferred."),
        gate(10, "Four RTX3090 memory/runtime resource admission and forecast", [rr, "PRE_FORMAL_RESOURCE_PREDICTION.json"],
             "Each realGPU live-model82293KV +cat/expandedworkspace+nativeSDPA finite, allocated22289758720B,reserved22334668800B, driverfree2786787328B >frozen1GiB. Conditional timing scenarios explicitly separate audited128 rates from not-yet-measured formal rates. No taskGPUh hardcap."),
        gate(11, "Exact FrozenV4 independent post-generation scorer", ["protocol_recovery/CPU_PROTOCOL_QUALIFICATION.json", "protocol_recovery/OFFLINE_FINISHER_CPU_GUARD_TEST_FINAL.json", "source/score_complete_generations.py"],
             "Exact historical fa5dd904 source and CPU extraction fixtures; frozen independent finisher requires120 unique legal COMPLETE, both60, fourworkers each30, source/Plan/GPU/payload/termination and actual release. No currentAIME output scored; incomplete yields no empty scienceCSV."),
        gate(12, "FourGPU model/source/Plan/environment identities", ["MODEL_TOKENIZER_FRESH_IDENTITY.json", "SOURCE_MANIFEST.json", "RUNTIME_SOURCE_MANIFEST.json", "PREREGISTRATION.json", cr, fr, rr],
             "Shared17model/tokenizer files streamingSHA verified;214task+2952external sourceSHA rechecked. Actual4physical3090UUID/CVDlogical0/BF16/native90d/Plan/runtime/environment metadata bound across canary/fresh/resource. Formal repeats frozen source/admission and physical checks before actual use."),
    ]
    bindings = {"prereg_sha256": sha("PREREGISTRATION.json"), "source_manifest_sha256": sha("SOURCE_MANIFEST.json"),
                "runtime_source_manifest_sha256": sha("RUNTIME_SOURCE_MANIFEST.json"),
                "model_tokenizer_identity_sha256": sha("MODEL_TOKENIZER_FRESH_IDENTITY.json"), "plan_sha256": plan["plan_sha256"],
                "units_sha256": sha("FROZEN_UNITS.json")}
    qualification = dict(task_id=task, RUN_ID=run, status="PASS", scope="TWELVE_BOUNDED_PRE_FORMAL_GATES_ONLY",
                         gates=gates, gate_count=12, bindings=bindings, fixed_gpu_UUIDs=prereg["GPU_binding"],
                         actual_formal_generations_at_admission=0, formal_sampling_cap=81920,
                         numeric_qualification="PREFIX_ONLY_128", scientific_numeric_81920="NOT_TESTED_NEW_METHODS",
                         persistent_carrier="Bothmethods nativeFP32 reconstructedQDQ values; no unquantized shadow; not physical packedINT8",
                         STEP_actual_head_bits={"4":299,"6":170,"8":267,"16":32},
                         STEP_actual_mean_data_bits_excluding_metadata=6.333333333333333,
                         known_disclosures=["STEP decode-fit maxabs0 but3072layer byteSHA differences per128record; no byteexact fit-reference claim or unverified signedzero diagnosis",
                                            "STEP524288columnfloorhits per128record retained; zero observed affectedenergy/relativeMSE; FP16pivot placeholder explanation source-level only",
                                            "Historical prefillQDQFalse versus currentTrue and compensated/detTrue versus currentoriginalnative/detFalse: old AIME results referenceonly",
                                            "Actual81K capacity pressure is not an81K recurrent numerical trajectory or proof of future stability"],
                         previous_failed_run="Parent original STOP247file seal permanently retained; original0.006847216834624609GPUh inherited",
                         independent_admission_review=dict(path="independent_review/FORMAL_ADMISSION_REVIEW.json",sha256=sha("independent_review/FORMAL_ADMISSION_REVIEW.json")),
                         admission_builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    qsha = publish("GPU_QUALIFICATION.json", qualification)
    auth = read("AUTHORIZATION_RESOURCE.json")
    auth.update(allowed_modes=["formal"], scope="Existing human-authorized fixed120 actual AIME26generations; no new scientific conditions", **bindings)
    for name in ("canary_input_sha256", "max_prompt_tokens", "resource_min_free_bytes"):
        auth.pop(name, None)
    for name in ("execution/resource/RESULT.json", cr, fr, rr, "PRE_FORMAL_RESOURCE_PREDICTION.json", "GPU_QUALIFICATION.json"):
        assert read(name)["status"] == "PASS"
        auth["required_gate_receipts"][name] = sha(name)
    auth["dispatch_GPUs"] = [0,1,2,3]
    auth["no_automatic_retry"] = True
    auth["watchdog_seconds"] = 1800
    auth["recorded_utc"] = qualification["recorded_utc"]
    asha = publish("AUTHORIZATION_FORMAL.json", auth)
    print(json.dumps(dict(status="LOCAL_FORMAL_ADMISSION_PREPARED_NO_GPU_DISPATCH",gates=12,qualification_sha256=qsha,authorization_sha256=asha)))


if __name__ == "__main__":
    main()
