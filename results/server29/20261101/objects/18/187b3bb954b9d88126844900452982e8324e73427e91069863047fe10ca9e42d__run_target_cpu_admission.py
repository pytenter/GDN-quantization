"""Two-fresh target CPU admission entrypoint; never load weights or CUDA state.

Usage: CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python this.py
       --root /exact/new/PACE/root --run fresh1

All output is under operations/TARGET_CPU_<run>; formal validation files are
untouched. Actual installed native functions are used, rather than AST oracles.
"""
import argparse
import collections
import hashlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
import unittest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    require(re.fullmatch(r"[A-Za-z0-9_-]{1,40}", args.run) is not None, "Unsafe run name")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU admission requires literal empty CUDA_VISIBLE_DEVICES")
    root = args.root.resolve()
    output = root / "operations" / ("TARGET_CPU_" + args.run)
    output.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(root / "src"))
    calls = collections.Counter()
    weight_attempts = []
    started = time.monotonic()

    def profile(frame, event, _arg):
        if event == "c_call" and getattr(_arg, "__name__", "") == "_cuda_init":
            calls["native_cuda_init"] += 1
            raise RuntimeError("Native CUDA context initialization forbidden in CPU admission")
        if event == "call" and "/torch/cuda/" in frame.f_code.co_filename.replace("\\", "/"):
            name = frame.f_code.co_name
            calls[name] += 1
            if name in ("_lazy_init", "init", "set_device", "synchronize", "mem_get_info"):
                raise RuntimeError("CUDA context operation forbidden in CPU admission: " + name)

    def audit(event, values):
        if event == "open" and values and isinstance(values[0], (str, bytes, os.PathLike)):
            name = os.fsdecode(values[0])
            if "/Qwen3.5-9B/" in name and (name.endswith(".safetensors") or name.endswith(".bin")):
                weight_attempts.append(name)
                raise RuntimeError("Model weights may not be read during CPU admission")

    sys.addaudithook(audit)
    sys.setprofile(profile)
    report = dict(status="FAIL", run=args.run, scope="ACTUAL_TARGET_CPU_ONLY_NO_GPU_QUALIFICATION_CREDIT",
                  CUDA_VISIBLE_DEVICES="", model_weight_loads=0, GPU_qualification_credit=False)
    try:
        import torch
        torch.set_num_threads(1)
        # Read the flag, do not call any torch.cuda query/API to establish it.
        require(not bool(torch.cuda._initialized), "CUDA was initialized before CPU admission")
        import pace_runtime as runtime
        report["vendor_manifest_sha256"] = runtime.verify_vendor(root)
        reference = runtime.load_native_reference(root)
        report["native_reference_identity"] = reference.source_identity
        require(reference.source_sha256 == runtime.NATIVE_SHA256, "Native callback SHA metadata differs")

        def progress(phase, **values):
            print(json.dumps(dict(phase=phase, **values)), flush=True)

        math_tests = module(root / "tests/test_pace_math.py", "target_cpu_pace_math")
        math_report = math_tests.run_cpu_gates(output / "MATH", progress=progress, native_reference=reference)
        write(output / "MATH_SUMMARY.json", math_report)
        require(math_report["status"] == "PASS" and math_report["actual_cuda"] is False,
                "CPU mathematical integration gate failed")

        first_chunk_test = module(root / "tests/first_chunk_adapter_parity.py", "target_cpu_first_chunk_adapter")
        first_chunk_report = first_chunk_test.run_adapter_parity(
            root, device="cpu", native_module=reference.native_module,
            native_identity=dict(reference.source_identity, kind="ACTUAL_TARGET_INSTALLED_NATIVE_CPU_HELPERS"),
            progress=progress)
        write(output / "FIRST_CHUNK_ADAPTER_PARITY.json", first_chunk_report)
        require(first_chunk_report["status"] == "PASS" and first_chunk_report["actual_cuda"] is False and
                first_chunk_report["fixture_count"] == 44 and first_chunk_report["token_comparisons"] == 440,
                "CPU actual installed native first-chunk adapter parity failed")
        report["first_chunk_adapter_parity"] = dict(status=first_chunk_report["status"],
            native_reference_identity=first_chunk_report["native_identity"],
            fixture_count=44, paired_period_cases=88, token_comparisons=440,
            thresholds=first_chunk_report["thresholds"],
            report_sha256=sha(output / "FIRST_CHUNK_ADAPTER_PARITY.json"))

        binding_tests = module(root / "tests/test_pace_runtime_cpu.py", "target_cpu_runtime_bindings")
        suite = unittest.defaultTestLoader.loadTestsFromModule(binding_tests)
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        (output / "RUNTIME_TESTS.log").write_text(stream.getvalue())
        report["runtime_tests"] = dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                                       status="PASS" if result.wasSuccessful() else "FAIL")
        require(result.wasSuccessful(), "CPU mock-model binding tests failed")

        # Synthetic masks validate the post-calibration checker itself. These
        # masks are never copied into calibration or used for model evaluation.
        fixture_masks = {}
        for layer in runtime.GDN_LAYERS:
            high = sorted((layer + head) % 32 for head in range(8))
            fixture_masks[str(layer)] = dict(int8_heads=high, int4_heads=sorted(set(range(32))-set(high)))
        fixture = dict(schema="CPU_SYNTHETIC_MASK_FIXTURE_NOT_CALIBRATION", selection_decode_period=8,
                       head_count=32, int8_per_layer=8, layers=fixture_masks)
        fixture_path = output / "SYNTHETIC_MASK_FIXTURE.json"
        write(fixture_path, fixture)
        mask_test = module(root / "tests/actual_mask_gate.py", "target_cpu_actual_mask_checker")
        mask_report = mask_test.run_actual_mask_check(root, device="cpu", progress=progress,
                          native_reference=reference, head_mask_path=fixture_path)
        mask_report["scope"] = "CPU_SYNTHETIC_MASK_ROUTING_FIXTURE_NOT_CALIBRATION_OR_GPU_GATE"
        write(output / "MASK_CHECKER_CPU_FIXTURE.json", mask_report)
        require(mask_report["status"] == "PASS" and mask_report["actual_cuda"] is False,
                "Actual-mask checker CPU integration failed")

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("/data/zypan/modelscope_models/Qwen3.5-9B",
                                                   trust_remote_code=True, local_files_only=True)
        windows = json.loads((root / "protocol/calibration_windows.json").read_text())
        require(windows["AIME_data_used"] is False and len(windows["windows"]) == 8,
                "Frozen calibration panel differs")
        prefix_rows = []
        for row in windows["windows"]:
            source = root / row["source_text_path"]
            require(sha(source) == row["source_text_sha256"], "Calibration text SHA differs")
            ids = tokenizer(source.read_text(), add_special_tokens=False)["input_ids"][:2048]
            require(ids == row["input_ids"] and runtime.canonical_sha(ids) == row["input_ids_sha256"],
                    "Actual target tokenizer prefix differs")
            prefix_rows.append(dict(document_id=row["document_id"], tokens=len(ids),
                                    input_ids_sha256=runtime.canonical_sha(ids), status="PASS"))
            progress("TARGET_CPU_TOKENIZER_PREFIX_PASS", document_id=row["document_id"])
        write(output / "TOKENIZER_8_PREFIXES.json", dict(status="PASS", records=prefix_rows))

        binding = module(root / "vendor/protocol_binding.py", "target_cpu_protocol_binding")
        functions, ast_rows = binding.sampler(root, torch)
        expected = json.loads((root / "protocol/INHERITED_AIME26_SEED1_PROTOCOL_FREEZE.json").read_text())
        require(ast_rows == expected["sampler_AST"], "Sealed sampler AST identity differs")
        warpers = functions.build_warpers()
        require([type(x).__name__ for x in warpers] ==
                ["TemperatureLogitsWarper", "TopKLogitsWarper", "TopPLogitsWarper"], "Original warper ordering differs")
        data = json.loads((root / "AIME26_DATASET_FREEZE.json").read_text())
        require(len(data["records"]) == 30 and [x["problem_idx"] for x in data["records"]] == list(range(1, 31)),
                "Canonical30 roster differs")
        prompt_rows = []
        for row in data["records"]:
            messages, ids = functions.render_input(tokenizer, row["problem"])
            require(ids.device.type == "cpu" and ids[0].tolist() == row["input_ids"],
                    "Actual target frozen prompt tokens differ")
            text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True, enable_thinking=True)
            require(hashlib.sha256(text.encode()).hexdigest() == row["prompt_text_sha256"]
                    and functions.ids_sha256(ids[0].tolist()) == row["input_ids_sha256"],
                    "Actual target frozen prompt hashes differ")
            prompt_rows.append(dict(problem_id=row["problem_id"], input_tokens=ids.shape[1],
                                    input_ids_sha256=row["input_ids_sha256"], status="PASS"))
        write(output / "TOKENIZER_30_FROZEN_PROMPTS.json", dict(status="PASS", records=prompt_rows))
        write(output / "SAMPLER_AST_AND_WARPERS.json", dict(status="PASS", AST=ast_rows,
              warpers=[dict(name=type(x).__name__, path=inspect.getsourcefile(type(x)),
                            sha256=sha(inspect.getsourcefile(type(x)))) for x in warpers]))

        expected_bindings = {}
        for filename in ("EXTERNAL_PYTHON_BINDINGS.json", "EXTERNAL_NATIVE_BINDINGS.json"):
            catalog = json.loads((root / "analysis" / filename).read_text())
            for row in catalog["files"]:
                expected_bindings[str(Path(row["path"]).resolve())] = row
        module_rows = {}
        for name, loaded in sorted(sys.modules.items()):
            path = getattr(loaded, "__file__", None)
            if path and Path(path).is_file() and name.startswith(("torch", "transformers", "tokenizers", "safetensors",
                                                                 "accelerate", "huggingface_hub", "numpy")):
                path = str(Path(path).resolve())
                if path not in module_rows:
                    expected_row = expected_bindings.get(path)
                    digest, size = sha(path), Path(path).stat().st_size
                    require(expected_row is not None and digest == expected_row["sha256"]
                            and size == expected_row["bytes"], "Actual loaded module differs from frozen environment: " + path)
                    module_rows[path] = dict(path=path, sha256=digest, bytes=size,
                                             matches_frozen_environment=True, modules=[])
                row = module_rows[path]
                row["modules"].append(name)
        require(not any("qwen_gdn_group64" in p.lower() for p in module_rows),
                "Unexpected inherited GROUP64 private runtime resolution")
        write(output / "ACTUAL_LOADED_MODULE_PATHS_SHA.json", dict(status="PASS", count=len(module_rows),
                                                                  files=list(module_rows.values())))
        require(not weight_attempts and not bool(torch.cuda._initialized), "CPU admission touched weights/CUDA context")
        report.update(status="PASS", torch=torch.__version__, CPU_threads=torch.get_num_threads(),
                      actual_installed_native=True, AST_native_used_for_math_gate=False,
                      math_gates={k: dict(status=v["status"], tests=v["tests"]) for k, v in math_report["gates"].items()},
                      tokenizer_prefixes=8, frozen_prompts=30, actual_loaded_modules=len(module_rows),
                      CUDA_initialized_flag=False,
                      source_sha256={str(p.relative_to(root)): sha(p) for p in
                          [root / "src/pace_runtime.py", root / "src/pace_buffer.py", root / "src/pace_quantizer.py",
                           root / "tests/test_pace_math.py", root / "tests/test_pace_runtime_cpu.py",
                           root / "tests/actual_mask_gate.py", root / "tests/first_chunk_adapter_parity.py",
                           root / "tests/first_chunk_parity.py", Path(__file__).resolve()]})
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
    finally:
        sys.setprofile(None)
        report.update(seconds=time.monotonic()-started, model_weight_read_attempts=weight_attempts,
                      observed_internal_torch_cuda_python_functions=dict(calls),
                      CUDA_context_operation_calls=sum(calls[n] for n in
                          ("_lazy_init", "init", "set_device", "synchronize", "mem_get_info", "native_cuda_init")),
                      seed_note="CPU runtime fixtures may queue native lazy seed callbacks; no CUDA context is initialized")
        write(output / "REPORT.json", report)
        print(json.dumps(dict(status=report["status"], report=str(output / "REPORT.json"),
                              error=report.get("error")), ensure_ascii=False), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
