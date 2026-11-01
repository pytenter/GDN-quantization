"""CPU-only inspection of existing R2 compiler text; never invokes a compiler."""
from pathlib import Path
from collections import Counter
import datetime
import hashlib
import json
import re


ROOT = Path(__file__).resolve().parents[1]
IR = ROOT / "operations/CPU_COMPILE_R2_IR_REVIEW/c03c0724429741c9d89e48432bd16e71e8baf58b23d00d6c09defcdc414f2ad4"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    tg = (IR / "_row_kernel.ttgir").read_text()
    ll = (IR / "_row_kernel.llir").read_text()
    pt = (IR / "_row_kernel.ptx").read_text()
    source = (ROOT / "src/unified_codec.py").read_text()
    assert sha(ROOT / "src/unified_codec.py") == "3331a4f7ebd4118a078e70326bf07d291629a766cd06f50fff12e6c9985e5c34"
    defs = {}
    splits = {}
    add_lines = []
    for line in tg.splitlines():
        match = re.match(r"\s*(%\w+) = (.*)", line)
        if match:
            defs[match[1]] = match[2]
        match = re.match(r"\s*(%\w+), (%\w+) = tt.split (%\w+) : (.*)", line)
        if match:
            splits[(match[1], match[2])] = (match[3], match[4])
        match = re.match(r"\s*(%\w+) = arith.addf (%\w+), (%\w+) : tensor<(\d+)x(\d+)xf32,", line)
        if match:
            add_lines.append((match[1], match[2], match[3], int(match[4]), int(match[5])))
    # Check the actual SSA edges and logical transform at every reduction level,
    # not merely an operation count or list of tensor dimensions.
    levels = []
    for output, low, high, half, width in add_lines:
        convert, split_type = splits[(low, high)]
        assert f"tensor<{half}x{width}x2xf32" in split_type
        match = re.match(r"triton_gpu.convert_layout (%\w+) :", defs[convert])
        assert match
        trans = match[1]
        match = re.match(r"tt.trans (%\w+) \{order = array<i32: 1, 2, 0>\}", defs[trans])
        assert match
        reshape = match[1]
        match = re.match(r"tt.reshape (%\w+) \{allow_reorder = false\} : tensor<(\d+)x(\d+)xf32,.* -> tensor<2x(\d+)x(\d+)xf32,", defs[reshape])
        assert match
        previous = match[1]
        assert (int(match[2]), int(match[3]), int(match[4]), int(match[5])) == (2 * half, width, half, width)
        levels.append({"output": output, "input": previous, "input_extent": 2 * half, "width": width, "operation": "FP32 low + high"})
    assert len(levels) == 17
    by_input = {x["input"]: x for x in levels}
    roots = [x for x in levels if x["input"] not in {y["output"] for y in levels}]
    chains = []
    for first in roots:
        chain = [first]
        while chain[-1]["output"] in by_input:
            chain.append(by_input[chain[-1]["output"]])
        chains.append(chain)
    assert sorted((c[0]["input_extent"], c[0]["width"], len(c)) for c in chains) == [(32, 1, 5), (32, 1, 5), (128, 32, 7)]
    for chain in chains:
        assert [x["input_extent"] for x in chain] == ([128, 64, 32, 16, 8, 4, 2] if len(chain) == 7 else [32, 16, 8, 4, 2])
    reduce_lines = re.findall(r'"tt\.reduce"\(.*?\n    \}\) : .*?\n', tg, re.S)
    assert len(reduce_lines) == 2 and all("xi32," in x and "arith.addi" in x and "arith.addf" not in x for x in reduce_lines)
    assert not re.search(r"\b(?:fast|reassoc|contract|afn|nsz|nnan|ninf|arcp)\b", "\n".join(line for line in ll.splitlines() if re.search(r"= (?:fadd|fmul|fsub|fdiv)\b", line)))
    fmas = [line.strip() for line in ll.splitlines() if re.search(r"\bcall\b.*@[^ (]*fma[^ (]*\(", line)]
    allowed = {"35": (116, 50), "39": (117, 50), "49": (125, 28), "52": (126, 30)}
    counts = Counter()
    for line in fmas:
        metadata = re.search(r"!dbg !(\d+)", line)[1]
        assert metadata in allowed
        row, col = allowed[metadata]
        assert f"!{metadata} = !DILocation(line: {row}, column: {col}, scope: !7)" in ll
        assert "tl.log(" in source.splitlines()[row - 1]
        counts[metadata] += 1
    assert dict(counts) == {key: 24 for key in allowed}
    assert re.search(r'^!7 = distinct !DISubprogram\(name: "_row_kernel"', ll, re.M)
    ptx_lines = pt.splitlines()
    current_loc = None
    ptx_counts = Counter()
    exceptional = []
    assert "mov.f32 \t%f249, 0f7F800000;" in pt
    for index, line in enumerate(ptx_lines):
        match = re.search(r"\.loc\s+(\d+)\s+(\d+)\s+(\d+)", line)
        if match:
            current_loc = tuple(int(x) for x in match.groups())
        if re.search(r"\bfma\.rn", line):
            assert current_loc
            ptx_counts[current_loc] += 1
            if current_loc[1] == 0:
                # LLVM emitted both ftz/non-ftz implementations; these four final
                # PTX instructions are log(+/-inf/NaN) library special handling.
                assert re.search(r"fma\.rn\.ftz\.f32\s+%f38[2-5], %f(?:67|72|79|84), %f249, %f249;", line)
                assert "bra" in ptx_lines[index - 2]
                assert "2139095040" in "\n".join(ptx_lines[index - 4:index])
                exceptional.append({"ptx_line": index + 1, "loc": list(current_loc), "instruction": line.strip(), "meaning": "Library log nonfinite-input x * +inf + +inf branch; no reduction operands"})
    assert sum(ptx_counts.values()) == 48
    assert len(exceptional) == 4
    assert {x for x in ptx_counts if x[1]} == {(1, 116, 50), (1, 117, 50), (1, 125, 28), (1, 126, 30)}
    assert all(ptx_counts[(1, line, col)] == 11 for line, col in allowed.values())
    # The pointwise multiply and following add remain separate even after LLVM.
    assert re.search(r"= fmul float %\d+, 7\.812500e-03, !dbg !\d+\n\s*%\d+ = fadd float", ll)
    report = {
        "status": "PASS_INDEPENDENT_EXISTING_R2_ROW_COMPILED_FLOAT_TREE_REVIEW",
        "UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Only existing R2 row_h1 compiler text. No claim about remaining variants or CUDA execution.",
        "codec_sha256": sha(ROOT / "src/unified_codec.py"),
        "review_script_sha256": sha(Path(__file__)),
        "files": {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha(path)} for path in sorted(IR.iterdir()) if path.is_file()},
        "explicit_SSA_dependency_chains": chains,
        "TTGIR_logical_reduction_order": "At each level reshape [N,W] to [2,N/2,W] without reordering, transpose to [N/2,W,2], split ordered low/high, FP32 add. Seven Value128 stages, five stages for each Key32 GM. No floating generic tt.reduce.",
        "generic_reductions": reduce_lines,
        "LLVM_plain_FP_operation_counts": dict(Counter(re.search(r"= (fadd|fmul|fsub|fdiv)\b", line)[1] for line in ll.splitlines() if re.search(r"= (fadd|fmul|fsub|fdiv)\b", line))),
        "LLVM_fast_reassociation_or_contract_flags": 0,
        "LLVM_FMA_executing_calls": len(fmas),
        "LLVM_FMA_debug_closed_set": {key: {"count": counts[key], "source_line": value[0], "column": value[1], "scope": "_row_kernel", "origin": "Existing tl.log library implementation"} for key, value in allowed.items()},
        "PTX_FMA_executing_instructions": 48,
        "PTX_FMA_by_loc": {" ".join(map(str, key)): value for key, value in ptx_counts.items()},
        "PTX_line0_FMA_individually_classified": exceptional,
        "ordinary_pointwise_multiply_add_and_reduction_FMA": 0,
        "preserved_reference": {"path": "reference/quantization_math.py", "note": "No reference or log/exp implementation changes in this review; original 13x2 CPU proofs and narrow GM compatibility supplement remain separate evidence."},
        "qualification_boundary": "Compiler text verifies emitted ordering and provenance; only the existing future actual GPU qualification can prove device fields. Original failed execution remains 0/52 qualification and 0/90 formal; no gate relaxed.",
        "GPU_calls": 0, "model_loads": 0, "compiler_calls": 0, "controller_calls": 0, "finisher_calls": 0, "production_edits": 0,
    }
    target = ROOT / "operations/INDEPENDENT_R2_FLOAT_TREE_IR_REVIEW.json"
    assert not target.exists(), "Append-only receipt already exists"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {"status": report["status"], "path": str(target), "sha256": sha(target)}


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
