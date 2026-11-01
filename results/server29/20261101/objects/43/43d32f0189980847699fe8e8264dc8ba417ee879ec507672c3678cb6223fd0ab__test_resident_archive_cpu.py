#!/usr/bin/env python3
"""CPU-only archive codec/filter/byte-preservation tests; no SSH or GPU.

Generated archive programs run against isolated temporary files with a mocked
space observation and mocked NOREPLACE operation. This does not qualify the
remote host, its filesystem, atomic syscall or available capacity: the actual
29 positive/negative renameat2 and free-space tests remain mandatory at publish.
"""
from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import errno
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import types


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    spec = importlib.util.spec_from_file_location("_test_resident_archive", root / "src/resident_archive.py")
    archive = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archive)
    checks = []

    def check(name, predicate):
        if not predicate:
            raise AssertionError(name)
        checks.append(name)

    def rejects(name, function):
        try:
            function()
        except (AssertionError, RuntimeError, ValueError, OSError):
            checks.append(name)
        else:
            raise AssertionError("Negative case was accepted: " + name)

    def fake_noreplace(source, destination):
        if Path(destination).exists():
            raise OSError(errno.EEXIST, "CPU_TEST_MOCK_NOREPLACE")
        os.rename(source, destination)

    def execute(program, *, stream=None):
        """Execute generated statements; imports/defs first, then inject CPU mocks."""
        tree = ast.parse(program)
        definitions, statements = [], []
        for node in tree.body:
            (definitions if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)) else statements).append(node)
        namespace = {}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "<archive_cpu_defs>", "exec"), namespace)
        proxy = types.SimpleNamespace(**{name: getattr(os, name) for name in dir(os)})
        proxy.statvfs = lambda path: types.SimpleNamespace(f_bavail=200 * 1024 ** 3, f_frsize=1)
        namespace.update(os=proxy, rename_noreplace=fake_noreplace)
        if stream is not None:
            namespace["sys"] = types.SimpleNamespace(stdin=types.SimpleNamespace(buffer=io.BytesIO(stream)))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(ast.Module(body=statements, type_ignores=[]), "<archive_cpu_statements>", "exec"), namespace)
        return json.loads(stdout.getvalue()) if stdout.getvalue() else namespace

    def tar_bytes(source, rows, *, extra=None):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w") as stream:
            for row in rows:
                stream.add(source / row["path"], arcname=row["path"], recursive=False)
            if extra is not None:
                info = tarfile.TarInfo(extra)
                info.size = 1
                stream.addfile(info, io.BytesIO(b"x"))
        return output.getvalue()

    check("canonical_has_actual_newline", archive.canonical({"a": 1}) == b'{"a":1}\n')
    namespace = {}
    exec(archive.REMOTE_COMMON, namespace)
    check("55_29_canonical_exact_bytes", archive.canonical({"a": "中", "b": [1, 2]}) == namespace["canonical"]({"a": "中", "b": [1, 2]}))
    for name in ("a", "nested/file.bin", "nested/FINAL_SEAL_RECEIPT.json"):
        check("safe_positive_" + name, archive.safe_relative(name) and namespace["safe"](name))
    for name in ("", "/absolute", "../escape", "a/../b", "./a", "a/./b", "a\nfile", "a\rfile", "a\0file"):
        check("safe_negative_" + repr(name), not archive.safe_relative(name) and not namespace["safe"](name))
    pretty = b'{\n  "b": 2,\n  "a": 1\n}\n'
    encoded = dict(base64=base64.b64encode(pretty).decode(), bytes=len(pretty), sha256=hashlib.sha256(pretty).hexdigest())
    check("preserves_original_control_serialization", archive.decode_control(encoded) == pretty)
    check("json_equal_does_not_imply_original_sha_equal", json.loads(pretty) == json.loads(archive.canonical({"a": 1, "b": 2})) and hashlib.sha256(pretty).hexdigest() != hashlib.sha256(archive.canonical({"a": 1, "b": 2})).hexdigest())
    rejects("control_byte_count_tamper", lambda: archive.decode_control(dict(encoded, bytes=len(pretty) + 1)))
    rejects("control_sha_tamper", lambda: archive.decode_control(dict(encoded, sha256="0" * 64)))
    rejects("control_base64_tamper", lambda: archive.decode_control(dict(encoded, base64="invalid!?")))
    rejects("canonical_refuses_nonfinite", lambda: archive.canonical({"nan": float("nan")}))

    with tempfile.TemporaryDirectory(prefix="resident_archive_CPU_ONLY_") as temp:
        folder = Path(temp)
        source = folder / "science"
        source.mkdir()
        (source / "SOURCE_FREEZE.json").write_bytes(archive.canonical({"files": [{"path": "src.py", "bytes": 8, "sha256": hashlib.sha256(b"science\n").hexdigest()}]}))
        (source / "src.py").write_bytes(b"science\n")
        (source / "nested").mkdir()
        (source / "nested/FINAL_ARCHIVE_VERIFICATION.json").write_bytes(pretty)
        (source / "nested/FINAL_SEAL_RECEIPT.json").write_bytes(pretty)
        (source / "nested/FINISHER_RESULT.json").write_bytes(pretty)
        (source / "new_tensor.bin").write_bytes(bytes(range(256)))
        (source / "logs").mkdir()
        (source / "logs/finalizer.log").write_bytes(b"live\n")
        for excluded in archive.CONTROLS | archive.POST_SEAL:
            path = source / excluded
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not catalogued\n")
        rows, exclusions = archive.catalog(source, {"logs/finalizer.log": "exact CPU fixture live descriptor"})
        names = {row["path"] for row in rows}
        check("dynamic_science_count_not_legacy_68_or_104", len(rows) == 6)
        check("exact_root_controls_only_excluded", not (archive.CONTROLS & names))
        check("nested_original_controls_and_result_preserved", {"nested/FINAL_ARCHIVE_VERIFICATION.json", "nested/FINAL_SEAL_RECEIPT.json", "nested/FINISHER_RESULT.json"} <= names)
        check("raw_tensor_is_science_remote_only", "new_tensor.bin" in names)
        check("live_descriptor_exact_file_exclusion", "logs/finalizer.log" in exclusions and "logs" not in exclusions)
        archive.verify_rows(source, rows)
        checks.append("fresh55_all_sha_bytes")
        (source / "src.py").write_bytes(b"changed\n")
        rejects("fresh55_changed_source_rejected", lambda: archive.verify_rows(source, rows))
        (source / "src.py").write_bytes(b"science\n")
        (source / "link").symlink_to(source / "src.py")
        rejects("science_symlink_rejected", lambda: archive.catalog(source, {}))
        (source / "link").unlink()
        os.mkfifo(source / "fifo")
        rejects("science_nonregular_rejected", lambda: archive.catalog(source, {}))
        (source / "fifo").unlink()
        final, stage = folder / "final", folder / "partial_once"
        manifest = dict(task_id="CPU_ONLY_FIXTURE", execution_attempt="cpu", terminal="FAIL_CLOSED_STOP", formal_completed=0,
                        source_freeze_sha256=archive.sha(source / "SOURCE_FREEZE.json"), count=len(rows), bytes=sum(row["bytes"] for row in rows), files=rows)
        for name, program in (("preflight", archive.preflight_program(final, stage)), ("initialize", archive.initialize_program(final, stage, manifest)),
                              ("receiver", archive.receiver_program(stage)), ("publish", archive.publish_program(final, stage, {}, manifest))):
            compile(program, "<generated_" + name + ">", "exec")
            checks.append("generated_" + name + "_compiles")
        preflight = execute(archive.preflight_program(final, stage))
        check("preflight_mock_positive_negative_codepaths", preflight["free_bytes"] == 200 * 1024 ** 3)
        initial = execute(archive.initialize_program(final, stage, manifest))
        check("manifest_original_bytes_written", (stage / "SHA256_MANIFEST.json").read_bytes() == archive.canonical(manifest))
        received = execute(archive.receiver_program(stage), stream=tar_bytes(source, rows))
        check("uncompressed_receiver_full_dynamic_set", received["count"] == len(rows))
        namespace["inventory"](stage, names | {"SHA256_MANIFEST.json", "SHA256SUMS"})
        checks.append("strict_inventory_exact_set")
        extra = stage / "extra_empty"
        extra.mkdir()
        rejects("strict_inventory_extra_empty_directory", lambda: namespace["inventory"](stage, names | {"SHA256_MANIFEST.json", "SHA256SUMS"}))
        rejects("publish_refuses_extra_empty_directory", lambda: execute(archive.publish_program(final, stage, initial, manifest)))
        check("invalid_stage_not_published", not final.exists() and stage.exists())
        extra.rmdir()
        (stage / "extra.bin").write_bytes(b"x")
        rejects("strict_inventory_extra_file", lambda: namespace["inventory"](stage, names | {"SHA256_MANIFEST.json", "SHA256SUMS"}))
        (stage / "extra.bin").unlink()
        saved = (stage / "src.py").read_bytes()
        (stage / "src.py").unlink()
        rejects("strict_inventory_missing_file", lambda: namespace["inventory"](stage, names | {"SHA256_MANIFEST.json", "SHA256SUMS"}))
        (stage / "src.py").write_bytes(saved)
        (stage / "unexpected_link").symlink_to(stage / "src.py")
        rejects("strict_inventory_symlink", lambda: namespace["inventory"](stage, names | {"SHA256_MANIFEST.json", "SHA256SUMS"}))
        (stage / "unexpected_link").unlink()
        (stage / "src.py").write_bytes(b"changed\n")
        rejects("publish_fresh29_tamper_rejected", lambda: execute(archive.publish_program(final, stage, initial, manifest)))
        (stage / "src.py").write_bytes(saved)
        published = execute(archive.publish_program(final, stage, initial, manifest))
        check("mock_stage_published_once", final.exists() and not stage.exists())
        check("four_controls_exact_names", set(published["controls"]) == archive.CONTROLS)
        for name, record in published["controls"].items():
            check("exact_original_control_roundtrip_" + name, archive.decode_control(record) == (final / name).read_bytes())
        namespace["inventory"](final, names | archive.CONTROLS)
        check("final_dynamic_inventory_exact", len([path for path in final.rglob("*") if path.is_file()]) == len(rows) + 4)
        rejects("existing_final_cannot_republish", lambda: execute(archive.publish_program(final, stage, initial, manifest)))
        bad_stage = folder / "partial_bad_tar"
        execute(archive.initialize_program(folder / "unused_final", bad_stage, manifest))
        rejects("receiver_unknown_member_rejected", lambda: execute(archive.receiver_program(bad_stage), stream=tar_bytes(source, rows, extra="extra.bin")))
        check("bad_tar_preserved_not_final_published", bad_stage.exists() and not (folder / "unused_final").exists())
    result = dict(status="PASS_CPU_ARCHIVE_CODEC_FILTER_SERIALIZATION", count=len(checks), checks=checks,
                  archive_module_sha256=archive.sha(root / "src/resident_archive.py"),
                  scope="CPU isolated temporary files only; generated remote code exercised with mocked free-space and NOREPLACE; no SSH/network/GPU or actual29 atomic/capacity proof",
                  real29_renameat2_positive_negative_and_headroom_still_required=True, new_GPU_dispatches=0, new_AIME_samples=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(archive.canonical(result))
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(status=result["status"], count=len(checks))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
