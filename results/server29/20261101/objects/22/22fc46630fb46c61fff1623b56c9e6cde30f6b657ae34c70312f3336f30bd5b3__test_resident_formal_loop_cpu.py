#!/usr/bin/env python3
"""Execute the actual formal generate() loop against a stdlib mock runtime.

No Torch import, numerical model, CUDA operation, AIME input or GPU admission.
The three deliberately short cases test only EOS/cap/control semantics. They
cannot establish numerical correctness, model quality, GPU readiness or TPS.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from types import SimpleNamespace as NS


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(root):
    worker = import_file("_resident_formal_loop_under_test", root / "scripts/resident_formal_worker.py")
    validation = import_file("_resident_formal_record_under_test", root / "src/resident_record_validation.py")
    qualifier = import_file("_resident_qualifier_audit_under_test", root / "scripts/qualify_gpu.py")
    checks, case_rows, generator_instances, masks = [], [], [], []
    bytes_cache = {}

    def check(name, ok):
        if not ok:
            raise AssertionError(name)
        checks.append(dict(name=name, status="PASS"))

    class Tensor:
        def __init__(self, data, shape=(), dtype="mock"):
            self.data, self.shape, self.dtype = data, tuple(shape), dtype

        def detach(self): return self
        def cpu(self): return self
        def clone(self): return Tensor(self.data, self.shape, self.dtype)
        def __getitem__(self, _key): return self
        def tolist(self): return list(self.data)

    class Current:
        def __init__(self, token): self.token = int(token)
        def item(self): return self.token

    class Mask:
        def __init__(self):
            self.seen = set()
            masks.append(self)

        def scatter_(self, _dim, current, value):
            if value is not True:
                raise AssertionError("Presence mask must stay Boolean")
            self.seen.add(current.item())

    class Generator:
        def __init__(self, device):
            if device != "cuda:0":
                raise AssertionError("Production selector changed; this fixture mocks rather than initializes it")
            self.seed, self.draws = None, 0
            generator_instances.append(self)

        def manual_seed(self, seed):
            self.seed, self.draws = seed, 0
            return self

        def get_state(self):
            return Tensor(f"mock-generator-seed{self.seed}-draws{self.draws}".encode())

    torch = NS(Generator=Generator, bool="mock_bool", inference_mode=contextlib.nullcontext,
               zeros=lambda *args, **kwargs: Mask(),
               equal=lambda a, b: a.shape == b.shape and a.dtype == b.dtype and a.data == b.data,
               cuda=NS(reset_peak_memory_stats=lambda: None,
                       max_memory_allocated=lambda: 123456, max_memory_reserved=lambda: 234567))

    def payload(value, size):
        key = (value, size)
        if key not in bytes_cache:
            bytes_cache[key] = value * (size // len(value))
        return bytes_cache[key]

    class Encoded:
        def __init__(self, method, shape, codes, scales, zero_points, negative_zero_flags):
            assert method == "B1" and zero_points is None
            self.shape, self.codes, self.scales = shape, codes, scales
            self.negative_zero_flags = negative_zero_flags

        def cpu(self): return self

    def decode(encoded):
        value = encoded.codes.data[0]
        return Tensor(payload(struct.pack("<f", float(value)), 2097152), (1,32,128,128), "torch.float32")

    codec = NS(Encoded=Encoded, decode=decode)
    resident = NS(_load_codec=lambda: codec, GDN_LAYERS=validation.LAYERS)
    qualifier.torch, qualifier.resident = torch, resident
    qualifier.tensor_bytes = lambda tensor: tensor.data
    qualifier.save_tensor_diagnostic = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("No mock State mismatch is expected"))
    brief_validation = NS(**{name: getattr(validation, name) for name in
        ("CORE_SHA", "SAMPLER_SHA", "SCORER_SHA", "sparse_position", "sparse_positions")}, CAP=5)

    class Patch:
        def __init__(self, runtime, observer):
            self.runtime, self.observer = runtime, observer

        def __enter__(self):
            self.runtime.current_patch = self
            return self

        def __exit__(self, *_): self.runtime.current_patch = None

        def audit(self):
            total = len(self.runtime.inputs)
            return dict(processed_tokens=total, qk_batched_matmul_calls=24*total,
                unique_qk_vectors_rotated=32*24*total, original_gdn_helper_calls=24*total,
                qdq_calls=24*total, full_token_finite_checks=total, state_rotation_matmul_calls=0,
                P_inside_QDQ=False, chunk_prefill_calls=24, recurrent_decode_calls=24*(total-1),
                high_precision_continuation_shadow=False, state_dtype="FP32", packed_INT8_persistent_storage=False,
                next_consumption_full_value_byte_checks=0)

    prompt_ids = [101, 102, 103]
    problem, prompt_text = "SYNTHETIC_NON_AIME_PROTOCOL_FIXTURE", "SYNTHETIC_THINKING_TEMPLATE\n"

    class Runtime:
        def __init__(self, fixture_root, case, sequence):
            self.root, self.target = fixture_root, fixture_root / case
            self.target.mkdir()
            self.vocab, self.inputs, self.sequence = 248320, [], sequence
            self.environment = dict(scope="STDLIB_MOCK_RUNTIME_NO_TORCH_NO_MODEL_NO_GPU")
            self.legacy = NS(get_state=lambda cache, layer: cache[layer])
            self.tokenizer = NS(apply_chat_template=lambda *a, **k: prompt_text,
                                decode=lambda tokens, **k: "synthetic output " + str(tokens))
            self.drawn = []

        def sampler(self):
            def sample(_warpers, _logits, mask, generator):
                check(self.target.name + "_boolean_generated_only_presence_" + str(len(self.drawn)),
                      mask.seen == set(self.drawn) and not mask.seen.intersection(prompt_ids))
                check(self.target.name + "_sampler_generator_sequence_" + str(len(self.drawn)),
                      generator.seed == 1 and generator.draws == len(self.drawn))
                token = self.sequence[len(self.drawn)]
                self.drawn.append(token)
                generator.draws += 1
                return Current(token), 25  # Native top-k ties may exceed20.
            return NS(render_input=lambda *a: ([dict(role="user",content=problem)], Tensor(prompt_ids)),
                      ids_sha256=validation.objsha, setup_seed=lambda seed: check(self.target.name + "_fresh_seed1", seed==1),
                      build_warpers=lambda: object(), stop_ids=lambda _t: {42}, sample_next=sample), [dict(mock="AST")]

        def patch(self, _condition, _qdq, observer, audit):
            check(self.target.name + "_core_full_cpu_copy_disabled", audit is False)
            return Patch(self, observer)

        def forward(self, token, cache):
            index = len(self.inputs)
            self.inputs.append(int(token))
            updated = {}
            for layer in resident.GDN_LAYERS:
                if cache is not None:
                    self.current_patch.observer("NEXT_CONSUMPTION", index, layer, dict(actual=cache[layer]))
                value = 1 + index % 7
                encoded = Encoded("B1", (1,32,128,128),
                    Tensor(payload(bytes([value]),524288),(1,32,4,32,128),"torch.int8"),
                    Tensor(payload(struct.pack("<f",1.),65536),(1,32,4,128),"torch.float32"),None,
                    Tensor(payload(b"\0",524288),(1,32,4,32,128),"torch.bool"))
                state = decode(encoded)
                updated[layer] = state
                self.current_patch.observer("POST_QDQ", index, layer,
                    dict(codes=encoded.codes,scales=encoded.scales,negative_zero_flags=encoded.negative_zero_flags,writeback=state))
            return NS(logits=Tensor(b"mock logits")), updated

    with tempfile.TemporaryDirectory(prefix="resident_formal_cpu_protocol_") as temporary:
        fixture_root = Path(temporary)
        for relative in ("src/resident_rotation.py", "protocol/sampler.py", "protocol/aime26_scorer_v4.py"):
            destination = fixture_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, destination)
        (fixture_root / "AIME26_SEED1_PROTOCOL.json").write_text(json.dumps(dict(sampler_AST=[dict(mock="AST")])))
        (fixture_root / "AIME26_DATASET_FREEZE.json").write_text(json.dumps(dict(records=[
            dict(problem_id="synthetic",problem=problem,input_ids=prompt_ids,
                 prompt_text_sha256=hashlib.sha256(prompt_text.encode()).hexdigest())])))
        (fixture_root / "RESIDENT_FORMAL_CONSUMPTION_CONTRACT.json").write_text('{"fixture":"no authorization"}\n')
        for name, sequence, expected_reason in (
            ("first_EOS", [42], "eos"), ("middle_EOS", [8,8,42], "eos"),
            ("length_cap", [8,9,8,9,8], "length")):
            runtime = Runtime(fixture_root, name, sequence)
            job = dict(id=name,condition="C1",problem_id="synthetic",seed=1,gpu=1,
                       gpu_uuid=validation.UUIDS[1],input_ids_sha256=validation.objsha(prompt_ids),
                       input_tokens=len(prompt_ids),max_new_tokens=5)
            tokens = []
            row = worker.generate(runtime,job,qualifier,resident,brief_validation,tokens)
            expected_inputs = prompt_ids + sequence[:-1]
            check(name + "_actual_inputs_p_plus_n_minus1", runtime.inputs == expected_inputs)
            check(name + "_final_output_not_fed_back", row["last_output_token_fed_back"] is False
                  and row["actual_model_forward_calls"] == len(prompt_ids)+len(sequence)-1)
            check(name + "_exact_sampler_calls_and_termination", tokens == sequence
                  and row["sampler_calls"] == len(sequence) and row["finish_reason"] == expected_reason)
            check(name + "_24_QDQ_per_actual_input", row["quantizer_audit"]["qdq_calls"] == 24*len(expected_inputs))
            check(name + "_actual_sparse_coverage", row["sparse_next_consumption_checks"] == 24*(len(expected_inputs)-1))
            case_rows.append(dict(case=name, generated_tokens=tokens, actual_inputs=runtime.inputs,
                                  expected_p_plus_n_minus1=len(expected_inputs), initial_generator_sha256=row["initial_generator_sha256"],
                                  finish_reason=row["finish_reason"]))
    check("fresh_generator_and_mask_objects_for_each_sample",
          len({id(x) for x in generator_instances}) == len({id(x) for x in masks}) == 3)
    check("same_seed1_initial_generator_state_each_sample",
          len({row["initial_generator_sha256"] for row in case_rows}) == 1)
    return dict(status="PASS_ACTUAL_GENERATE_LOOP_STDLIB_MOCK_THREE_TERMINATION_CASES",
                scope="Actual production generate function, mocked tensors/runtime/sampler; not numerical,9B,GPU or realSampler proof",
                GPU_dispatches=0, AIME_generations=0, Torch_imported="torch" in sys.modules,
                synthetic_short_cap=5, production_cap_unchanged=validation.CAP,
                cases=case_rows, checks=checks,
                worker_sha256=validation.sha(root / "scripts/resident_formal_worker.py"),
                test_sha256=validation.sha(Path(__file__)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=run(args.root.resolve())
    if result["Torch_imported"]:
        raise AssertionError("CPU protocol test imported Torch unexpectedly")
    output=args.output if args.output.is_absolute() else args.root/args.output
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as stream:
        json.dump(result,stream,indent=2,sort_keys=True)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"],checks=len(result["checks"]))))


if __name__ == "__main__":
    main()
