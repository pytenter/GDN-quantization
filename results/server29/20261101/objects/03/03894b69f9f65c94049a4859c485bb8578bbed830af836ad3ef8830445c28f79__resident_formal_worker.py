#!/usr/bin/env python3
"""One admitted C1/S3 sample, one fresh process, never a launcher or retry.

All Torch/CUDA-capable imports follow the stdlib-only control admission. State
remains in UH/UHP through the unchanged SHA-pinned resident adapter. No scoring
or answer-dependent control occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def objsha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def atomic_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def progress(target, phase, **values):
    path = Path(target) / "PROGRESS.json"
    temp = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temp.open("wb") as stream:
        stream.write(canonical(dict(phase=phase, monotonic=time.monotonic(), **values)) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def append_chunk(target, tokens, count):
    with (Path(target) / "TOKEN_CHUNKS.jsonl").open("ab") as stream:
        stream.write(canonical(dict(event="GENERATED_TOKEN_CHUNK", monotonic=time.monotonic(),
                                     generated_token_count=count, token_ids=tokens)) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def storage_check():
    stat = os.statvfs("/data")
    require(stat.f_bavail * stat.f_frsize >= 8 * 1024 ** 3, "Data free headroom fell below8GiB")


def generate(runtime, job, qualification, resident, validation, active_tokens):
    torch = qualification.torch
    codec = resident._load_codec()
    root, target = runtime.root, runtime.target
    f, sampler_ast = runtime.sampler()
    require(sha(root / "src/resident_rotation.py") == validation.CORE_SHA
            and sha(root / "protocol/sampler.py") == validation.SAMPLER_SHA
            and sha(root / "protocol/aime26_scorer_v4.py") == validation.SCORER_SHA,
            "Formal resident core, original sampler or FrozenV4 bytes differ")
    protocol = read(root / "AIME26_SEED1_PROTOCOL.json")
    require(sampler_ast == protocol["sampler_AST"], "Original sampler AST binding changed")
    frozen = next(row for row in read(root / "AIME26_DATASET_FREEZE.json")["records"]
                  if row["problem_id"] == job["problem_id"])
    messages, rendered = f.render_input(runtime.tokenizer, frozen["problem"])
    input_ids = rendered[0].tolist()
    prompt_text = runtime.tokenizer.apply_chat_template(messages, tokenize=False,
                           add_generation_prompt=True, enable_thinking=True)
    require(input_ids == frozen["input_ids"] and len(input_ids) == job["input_tokens"]
            and f.ids_sha256(input_ids) == job["input_ids_sha256"]
            and hashlib.sha256(prompt_text.encode()).hexdigest() == frozen["prompt_text_sha256"],
            "Actual full Prompt render/token payload differs")
    require(0 < len(input_ids) <= 373 and job["seed"] == 1 and job["max_new_tokens"] == validation.CAP,
            "Sample Prompt/seed/cap contract differs")
    f.setup_seed(1)
    generator = torch.Generator(device="cuda:0").manual_seed(1)
    initial_rng = hashlib.sha256(qualification.tensor_bytes(generator.get_state())).hexdigest()
    warpers, stops = f.build_warpers(), f.stop_ids(runtime.tokenizer)
    mask = torch.zeros((1, runtime.vocab), dtype=torch.bool, device="cuda:0")
    require(bool(stops), "Original EOS roster is empty")
    writes = {str(layer): 0 for layer in resident.GDN_LAYERS}
    expected_next, latest_gpu_encoded, consumption_evidence = {}, {}, []
    total = 0

    def fields(encoded):
        return [dict(name=name, dtype=str(getattr(encoded, name).dtype),
                     shape=list(getattr(encoded, name).shape),
                     bytes=len(qualification.tensor_bytes(getattr(encoded, name))),
                     sha256=hashlib.sha256(qualification.tensor_bytes(getattr(encoded, name))).hexdigest())
                for name in ("codes", "scales", "negative_zero_flags")]

    def check_cache(actual, encoded, index, layer, *, final_writeback=False):
        expected = codec.decode(encoded)
        actual_cpu = actual.detach().cpu()
        raw, wanted = qualification.tensor_bytes(actual_cpu), qualification.tensor_bytes(expected)
        equal = (actual_cpu.shape == expected.shape and actual_cpu.dtype == expected.dtype
                 and torch.equal(actual_cpu, expected) and raw == wanted)
        if not equal:
            name = "RESIDENT_FINAL_WRITEBACK_FIRST_DIFFERENCE" if final_writeback else "RESIDENT_NEXT_CONSUMPTION_FIRST_DIFFERENCE"
            qualification.save_tensor_diagnostic(target, name,
                [("actual", actual), ("expected", expected)], input_index=index, layer=layer,
                scope=("Final persistent QDQ writeback, not a next input" if final_writeback
                       else "Actual resident post-QDQ next input; no continuation or retry"))
            raise RuntimeError("Resident actual values/bytes differ from the previous encoded writeback")
        return dict(input_index=index, layer=layer, actual_values_and_bytes_equal=True,
                    bytes=len(raw), actual_writeback_sha256=hashlib.sha256(raw).hexdigest(),
                    expected_decoded_sha256=hashlib.sha256(wanted).hexdigest(), encoded_fields=fields(encoded))

    def observe(event, token, layer, payload):
        require(token == total, "Resident callback/model input index drifted")
        if event == "NEXT_CONSUMPTION" and validation.sparse_position(token):
            require(layer in expected_next, "Sparse resident encoded evidence missing")
            consumption_evidence.append(check_cache(payload["actual"], expected_next.pop(layer), token, layer))
        elif event == "POST_QDQ":
            writes[str(layer)] += 1
            # These are encoded GPU tensors from the actual codec, not an
            # unquantized State clone. Keep at most one set per resident layer.
            encoded = codec.Encoded("B1", tuple(payload["writeback"].shape),
                          payload["codes"].detach(), payload["scales"].detach(),
                          None, payload["negative_zero_flags"].detach())
            latest_gpu_encoded[layer] = encoded
            if validation.sparse_position(token + 1):
                require(layer not in expected_next, "Unconsumed sparse evidence would be overwritten")
                expected_next[layer] = encoded.cpu()

    cache, out = None, None
    survivors, chunk = [], []
    p = len(input_ids)
    started = time.monotonic()
    prompt_start = started
    torch.cuda.reset_peak_memory_stats()
    patch = runtime.patch(job["condition"], True, observer=observe, audit=False)

    def step(token):
        nonlocal total, cache
        out, cache = runtime.forward(token, cache)
        total += 1
        if total % 32 == 0:
            progress(target, "FORMAL_FORWARD_COMMITTED", input_forwards=total,
                     generated_tokens=len(active_tokens), problem_id=job["problem_id"], condition=job["condition"])
        if total % 256 == 0:
            storage_check()
        return out

    with torch.inference_mode(), patch:
        for token in input_ids:
            out = step(token)
        prompt_seconds = time.monotonic() - prompt_start
        decode_start = time.monotonic()
        while len(active_tokens) < validation.CAP:
            current, alive = f.sample_next(warpers, out.logits[:, -1, :], mask, generator)
            require(1 <= alive <= runtime.vocab, "Original sampler has no valid survivors")
            mask.scatter_(1, current, True)
            token = int(current.item())
            active_tokens.append(token)
            survivors.append(alive)
            chunk.append(token)
            if len(active_tokens) % 128 == 0 or token in stops or len(active_tokens) == validation.CAP:
                append_chunk(target, chunk, len(active_tokens))
                chunk = []
                progress(target, "FORMAL_GENERATED_TOKENS_COMMITTED", input_forwards=total,
                         generated_tokens=len(active_tokens), problem_id=job["problem_id"], condition=job["condition"])
            if token in stops or len(active_tokens) == validation.CAP:
                break  # This final output token is never consumed by the model.
            out = step(token)
        decode_seconds = time.monotonic() - decode_start
        n, audit = len(active_tokens), patch.audit()
        require(total == p + n - 1 and all(value == total for value in writes.values()),
                "Actual Prompt/Decode p+n-1 writeback timing differs")
        qualification.assert_audit(audit, total, True, full_next=False)
        require(len(consumption_evidence) == 24 * len(validation.sparse_positions(total)),
                "Sparse actual next-consumption evidence incomplete")
        final_evidence = []
        for layer in resident.GDN_LAYERS:
            require(layer in latest_gpu_encoded, "Final encoded resident writeback missing")
            encoded = latest_gpu_encoded[layer].cpu()
            final_evidence.append(check_cache(runtime.legacy.get_state(cache, layer), encoded,
                                             total - 1, layer, final_writeback=True))
        # A possible encoded sample for input index==total is not consumed:
        # that next model input deliberately does not exist after EOS/cap.
        expected_next.clear()
    elapsed = time.monotonic() - started
    tokens = list(active_tokens)
    return dict(status="COMPLETE", condition=job["condition"], problem_id=job["problem_id"],
                seed=1, gpu=job["gpu"], gpu_uuid=job["gpu_uuid"],
                input_tokens=p, input_ids_sha256=f.ids_sha256(input_ids),
                prompt_messages_sha256=objsha(messages), prompt_text_sha256=hashlib.sha256(prompt_text.encode()).hexdigest(),
                generated_token_ids=tokens, generated_token_ids_sha256=objsha(tokens), generated_token_count=n,
                response=runtime.tokenizer.decode(tokens, skip_special_tokens=True),
                finish_reason="eos" if tokens[-1] in stops else "length", stop_token_ids=sorted(stops),
                max_new_tokens=validation.CAP, runtime_error=None, nonfinite=False,
                actual_model_forward_calls=total, quantizer_audit=audit, writebacks_by_layer=writes,
                prompt_layer_QDQ=24 * p, decode_layer_QDQ=24 * (n - 1),
                sampler_calls=n, last_output_token_fed_back=False,
                sampling_parameters=dict(temperature=1., top_p=.95, top_k=20, min_p=0., presence_penalty=1.5,
                                         repetition_penalty=1., generated_only_presence=True, seed=1, thinking=True),
                sampling_audit=dict(top_k_parameter=20, min_survivors=min(survivors), max_survivors=max(survivors),
                      above20_due_to_original_native_tie_semantics=sum(value > 20 for value in survivors),
                      original_warper_unchanged=True),
                initial_generator_sha256=initial_rng,
                final_generator_sha256=hashlib.sha256(qualification.tensor_bytes(generator.get_state())).hexdigest(),
                sparse_next_consumption_positions=validation.sparse_positions(total),
                sparse_next_consumption_checks=len(consumption_evidence),
                sparse_next_consumption_evidence=consumption_evidence,
                sparse_evidence_sha256=objsha(consumption_evidence), final_writeback_evidence=final_evidence,
                final_writeback_is_not_an_extra_next_consumption=True,
                prompt_seconds=prompt_seconds, decode_and_sampling_seconds=decode_seconds,
                wall_seconds=elapsed, TPS=n / elapsed,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                resident_core_sha256=sha(root / "src/resident_rotation.py"),
                sampler_source_sha256=sha(root / "protocol/sampler.py"),
                scorer_source_sha256=sha(root / "protocol/aime26_scorer_v4.py"),
                dataset_freeze_sha256=sha(root / "AIME26_DATASET_FREEZE.json"),
                consumption_contract_sha256=sha(root / "RESIDENT_FORMAL_CONSUMPTION_CONTRACT.json"),
                environment=runtime.environment)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    # This control module and record validator must contain only stdlib imports.
    control = importlib.import_module("resident_execution_control")
    validation = importlib.import_module("resident_record_validation")
    admission = control.admit_formal(root, args.unit, args.gpu, os.getpid())
    validation.verify_consumption_contract(root)
    require(admission["admission"]["formal_permit_sha256"] == sha(root / "FORMAL_GPU_PERMIT.json")
            and admission["admission"]["authorization_sha256"] == sha(root / "AUTHORIZATION_FORMAL.json"),
            "Formal permit/authorization changed after control admission")
    job, target = admission["job"], Path(admission["target"])
    require(job["id"] == args.unit and job["gpu"] == args.gpu
            and target == root / "formal" / args.unit, "Control returned a different formal sample")
    require(not (target / "STARTED.json").exists() and not (target / "RESULT.json").exists(),
            "Sample already attempted; no restart or overwrite")
    atomic_new(target / "STARTED.json", dict(identity=admission["identity"], monotonic=time.monotonic(),
               admission=admission["admission"], source_freeze_sha256=sha(root / "SOURCE_FREEZE.json"),
               permit_sha256=sha(root / "GPU_PERMIT.json"),
               formal_permit_sha256=sha(root / "FORMAL_GPU_PERMIT.json"),
               formal_authorization_sha256=sha(root / "AUTHORIZATION_FORMAL.json")))
    runtime, active_tokens = None, []
    try:
        # No Torch import, model load, or CUDA context exists above admission.
        sys.path.insert(0, str(root / "vendor"))
        import torch
        import resident_rotation as resident
        qualification = importlib.import_module("qualify_gpu")
        require(Path(qualification.__file__).resolve() == root / "scripts/qualify_gpu.py",
                "Qualification/runtime import escaped the independent root")
        qualification.torch, qualification.resident = torch, resident
        runtime = qualification.Runtime(root, target, args.gpu, job["condition"])
        atomic_new(target / "PREFLIGHT_LIBRARY.json", control.check_after_model(root, args.gpu, os.getpid()))
        result = generate(runtime, job, qualification, resident, validation, active_tokens)
        atomic_new(target / "POSTFLIGHT_LIBRARY.json", control.check_after_model(root, args.gpu, os.getpid()))
        control.verify_sources(root)
        record = dict(status="PASS", unit=job["id"], result=result, identity=admission["identity"],
                      source_freeze_sha256=sha(root / "SOURCE_FREEZE.json"), permit_sha256=sha(root / "GPU_PERMIT.json"),
                      formal_permit_sha256=sha(root / "FORMAL_GPU_PERMIT.json"),
                      formal_authorization_sha256=sha(root / "AUTHORIZATION_FORMAL.json"),
                      monotonic=time.monotonic())
        record["payload_sha256"] = objsha(record)
        validation.validate(record, job, root)
        atomic_new(target / "RESULT.json", record)
        progress(target, "FORMAL_SAMPLE_PASS", generated_tokens=len(active_tokens),
                 input_forwards=result["actual_model_forward_calls"])
        return 0
    except BaseException as exc:
        atomic_new(target / "PARTIAL_TOKENS.json", dict(generated_token_ids=active_tokens,
                   count=len(active_tokens), score_eligible=False, continuation_allowed=False))
        if runtime is not None:
            try:
                cache = runtime.last_cache if runtime.last_cache is not None else runtime.last_input_cache
                fields = [("resident_state/" + str(layer), runtime.legacy.get_state(cache, layer))
                          for layer in resident.GDN_LAYERS] if cache is not None else []
                if runtime.last_out is not None:
                    fields.append(("last_completed_forward_logits", runtime.last_out.logits))
                if fields:
                    qualification.save_tensor_diagnostic(target, "ACTUAL_RESIDENT_CACHE_AT_FAILURE", fields,
                        scope="Actual GDN State and last COMPLETED logits only; not a resumable cache",
                        last_attempt_token=runtime.last_attempt_token, continuation_allowed=False)
            except BaseException as diagnostic_error:
                atomic_new(target / "DIAGNOSTIC_FAILURE.json", dict(error=repr(diagnostic_error)))
        atomic_new(target / "FIRST_FAILURE.json", dict(status="FORMAL_FAIL_CLOSED_STOP", unit=job["id"],
                    error=repr(exc), traceback=traceback.format_exc(), identity=admission["identity"],
                    monotonic=time.monotonic(), no_retry=True, score_eligible=False))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
