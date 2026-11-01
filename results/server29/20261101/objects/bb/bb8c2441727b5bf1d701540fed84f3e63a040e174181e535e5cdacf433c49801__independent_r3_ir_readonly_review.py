"""Read-only CPU review of already compiled text: no Triton or Torch imports."""
from pathlib import Path
from collections import Counter
import datetime
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
IR = ROOT / "operations/SM86_CPU_COMPILE_R3_INDEPENDENT_IR"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(text):
    defs = {}
    splits = {}
    adds = []
    for line in text.splitlines():
        match = re.match(r"\s*(%\w+) = (.*)", line)
        if match:
            defs[match[1]] = match[2]
        match = re.match(r"\s*(%\w+), (%\w+) = tt.split (%\w+) : (.*)", line)
        if match:
            splits[(match[1], match[2])] = (match[3], match[4])
        match = re.match(r"\s*(%\w+) = arith.addf (%\w+), (%\w+) : tensor<(\d+)x(\d+)xf32,", line)
        if match:
            adds.append((match[1], match[2], match[3], int(match[4]), int(match[5])))
    nodes = []
    for output, low, high, half, width in adds:
        convert, shape = splits[(low, high)]
        assert f"tensor<{half}x{width}x2xf32" in shape
        trans = re.match(r"triton_gpu.convert_layout (%\w+) :", defs[convert])[1]
        reshape = re.match(r"tt.trans (%\w+) \{order = array<i32: 1, 2, 0>\}", defs[trans])[1]
        match = re.match(r"tt.reshape (%\w+) \{allow_reorder = false\} : tensor<(\d+)x(\d+)xf32,.* -> tensor<2x(\d+)x(\d+)xf32,", defs[reshape])
        assert match and tuple(map(int, match.groups()[1:])) == (2 * half, width, half, width)
        nodes.append({"output": output, "input": match[1], "extent": 2 * half, "width": width})
    roots = [x for x in nodes if x["input"] not in {y["output"] for y in nodes}]
    by_input = {x["input"]: x for x in nodes}
    chains = []
    for root in roots:
        chain = [root]
        while chain[-1]["output"] in by_input:
            chain.append(by_input[chain[-1]["output"]])
        chains.append(chain)
    return defs, chains


def mul_args(defs, name):
    match = re.match(r"arith.mulf (%\w+), (%\w+) :", defs[name])
    assert match, (name, defs[name])
    return match[1], match[2]


def group_objectives(defs, chains, use_row):
    result = []
    x_ids = set()
    for chain in chains:
        term = chain[0]["input"]
        weight, square = mul_args(defs, term)
        diff, diff2 = mul_args(defs, square)
        assert diff == diff2
        match = re.match(r"arith.subf (%\w+), (%\w+) :", defs[diff])
        assert match
        x, decoded = match[1], match[2]
        assert defs[x].startswith("tt.load ")
        x_ids.add(x)
        decoded_original_coordinates = decoded
        row = None
        if use_row:
            decoded, row = mul_args(defs, decoded)
            assert defs[row].startswith("tt.broadcast ")
        match = re.match(r"arith.select (%\w+), (%\w+), (%\w+) :", defs[decoded])
        assert match
        sign, negative_zero, ordinary_decode = match[1], match[2], match[3]
        code_float, scale = mul_args(defs, ordinary_decode)
        assert defs[code_float].startswith("arith.sitofp ")
        assert defs[scale].startswith("tt.broadcast ")
        assert defs[weight].startswith("tt.broadcast ")
        result.append({"root": term, "weight": weight, "square": square, "difference": diff, "original_X_load": x, "decoded_in_original_coordinates": decoded_original_coordinates, "row_restore_after_Q_times_scale": row, "negative_zero_select": decoded, "plain_code_times_scale": ordinary_decode})
    assert len(x_ids) == 1
    return result


def run():
    process_path = ROOT / "operations/SM86_CPU_COMPILE_R3_PROCESS_RECEIPT.json"
    process = json.loads(process_path.read_text())
    report = process["report"]
    assert process["failure"] is None and report["variant_count"] == 20
    assert report["status"] == "PASS_EXPLICIT_SM86_CPU_COMPILATION_AND_IR_STRUCTURE_NO_GPU"
    for flag in ("GPU_calls", "kernel_launches", "model_loads", "torch_imports", "driver_context_calls", "actual_GPU_qualification_credit"):
        assert report[flag] == 0
    assert report["driver_access_attempts"] == report["torch_import_attempts"] == []
    assert report["options"] == {"num_warps": 4, "enable_fp_fusion": False}
    assert report["source_sha256"]["src/unified_codec.py"] == sha(ROOT / "src/unified_codec.py") == "3331a4f7ebd4118a078e70326bf07d291629a766cd06f50fff12e6c9985e5c34"
    assert report["source_sha256"]["reference/math_reference.py"] == sha(ROOT / "reference/math_reference.py") == "da1155ff3405a2151452404c469438df9b55f042bbc7504461888b0f3a4370fc"
    variants = {x["name"]: x for x in report["variants"]}
    assert len(variants) == 20 and all(x["status"] == "PASS_CPU_EXPLICIT_TARGET_COMPILE_ONLY" for x in variants.values())
    direct = {}
    source = (ROOT / "src/unified_codec.py").read_text().splitlines()
    for folder in sorted(IR.iterdir()):
        if not folder.is_dir():
            continue
        variant = variants[folder.name]
        files = {}
        for path in sorted(folder.iterdir()):
            expected = variant["files"][path.suffix[1:]]
            assert path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"]
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha(path)}
        kernel = "_row_kernel" if folder.name.startswith("row_") else "_group_kernel"
        tg = (folder / (kernel + ".ttgir")).read_text()
        ll = (folder / (kernel + ".llir")).read_text()
        pt = (folder / (kernel + ".ptx")).read_text()
        defs, chains = parse(tg)
        regular_fp = "\n".join(line for line in ll.splitlines() if re.search(r"= (?:fadd|fmul|fsub|fdiv)\b", line))
        assert not re.search(r"\b(?:fast|reassoc|contract|afn|nsz|nnan|ninf|arcp)\b", regular_fp)
        reductions = re.findall(r'"tt\.reduce"\(.*?\n    \}\) : .*?\n', tg, re.S)
        assert all("arith.addf" not in region for region in reductions)
        fmas = [line for line in ll.splitlines() if re.search(r"\bcall\b.*@[^ (]*fma[^ (]*\(", line)]
        ptx_fmas = [line for line in pt.splitlines() if re.search(r"\bfma\.rn", line)]
        objectives = None
        fma_counts = {}
        if kernel == "_row_kernel":
            assert sorted((c[0]["extent"], c[0]["width"], len(c)) for c in chains) == [(32, 1, 5), (32, 1, 5), (128, 32, 7)]
            meta = dict(re.findall(r"^!(\d+) = (!DILocation\(.*)$", ll, re.M))
            count = Counter()
            for line in fmas:
                identifier = re.search(r"!dbg !(\d+)", line)[1]
                match = re.fullmatch(r"!DILocation\(line: (\d+), column: (\d+), scope: !(\d+)\)", meta[identifier])
                assert match and int(match[1]) in (116, 117, 125, 126)
                assert "tl.log(" in source[int(match[1]) - 1]
                assert re.search(r'^!' + match[3] + r' = distinct !DISubprogram\(name: "_row_kernel"', ll, re.M)
                count[int(match[1])] += 1
            assert dict(count) == {116: 24, 117: 24, 125: 24, 126: 24}
            assert len(ptx_fmas) == 48
            fma_counts = dict(count)
        else:
            search = "_W_" in folder.name or "_RW_" in folder.name
            use_row = "_R_" in folder.name or "_RW_" in folder.name
            assert len(chains) == (5 if search else 1)
            assert all(c[0]["width"] == 16 and [n["extent"] for n in c] == [32, 16, 8, 4, 2] for c in chains)
            assert len(fmas) == len(ptx_fmas) == 0
            objectives = group_objectives(defs, chains, use_row)
        direct[folder.name] = {"files": files, "actual_TTGIR_SSA_addition_chains": chains, "objective_and_decode_association": objectives, "LLVM_FMA_calls": len(fmas), "PTX_FMA_instructions": len(ptx_fmas), "permitted_library_log_FMA_by_source_line": fma_counts, "fast_reassoc_or_contract_flags": 0, "generic_float_add_reductions": 0}
    assert len(direct) == 6
    out = {
        "status": "PASS_INDEPENDENT_R3_COMPILED_REDUCTION_DAG_AND_OBJECTIVE_REVIEW",
        "UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "codec_sha256": sha(ROOT / "src/unified_codec.py"),
        "script_sha256": sha(Path(__file__)),
        "process_receipt": {"path": str(process_path.relative_to(ROOT)), "sha256": sha(process_path), "reported_variants": 20, "reported_GPU_calls": 0},
        "direct_text_review_scope": "Six actual variants / 24 IR text files independently read and byte/SHA checked against process receipt. Other 14 variants' compile/structure status is the runtime auditor's report, not a second direct IR review here.",
        "variants_directly_reviewed": direct,
        "R2_independent_log_special_branch_receipt": {"path": "operations/INDEPENDENT_R2_FLOAT_TREE_IR_REVIEW.json", "sha256": sha(ROOT / "operations/INDEPENDENT_R2_FLOAT_TREE_IR_REVIEW.json")},
        "R2_receipt_metadata_path_correction": "Its preserved_reference.path says reference/quantization_math.py in error. The actual unchanged independent oracle is reference/math_reference.py, SHA da1155ff3405a2151452404c469438df9b55f042bbc7504461888b0f3a4370fc; no nonexistent file was read or verified by R2.",
        "conclusion": "Explicit halves-first additions remain ordered SSA dependencies with no reassociation flags. Group loss uses original X minus restored decoded state, then square, then omega, followed by five Key32 addition stages. Decode is code*scale, signed-zero selection, then positive row restore where enabled. Row FMA is confined to the four unchanged tl.log library bodies, including special-input handling; no reduction or ordinary pointwise multiply-add fusion was found.",
        "boundaries": "This review executes no compiler, GPU kernel, model, controller, finisher or scoring. Compiled IR is not GPU field qualification; exact actual-device codes/scales/signedzero/lambda/weighted-loss/decode gates still must pass unchanged. The old failed run remains 0/52 and 0/90.",
        "GPU_calls": 0, "compiler_calls": 0, "model_loads": 0, "controller_calls": 0, "finisher_calls": 0, "production_edits": 0,
    }
    target = ROOT / "operations/INDEPENDENT_R3_COMPILED_REDUCTION_DAG_REVIEW.json"
    assert not target.exists()
    target.write_text(json.dumps(out, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "path": str(target), "sha256": sha(target)}, sort_keys=True))


if __name__ == "__main__":
    run()
