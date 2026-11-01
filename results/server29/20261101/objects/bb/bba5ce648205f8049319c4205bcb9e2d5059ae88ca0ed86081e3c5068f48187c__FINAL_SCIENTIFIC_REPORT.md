# True STOP preserved

Status: PARTIAL_RESULTS_PRESERVED_AND_STOP. Actual fresh formal complete legal counts: {"W": 0, "R": 0, "RW": 0}. Qualification accounting: {"reused_qualification_count": 48, "fresh_qualification_legal_exit0_release": 0, "fresh_qualification_planned": 4, "combined_qualification_legal": 48, "qualification_planned": 52, "fresh_formal_legal_exit0_release": 0, "fresh_formal_planned": 90, "current_intervals": 1, "current_durable_units": 1, "original_reused_result_or_interval_files_created": 0, "old_failed_or_nonzero_result_credit": 0, "reuse_admission_sha256": "2ce745dce00c2a5237f0127e896b61a88d07b5f6f290ff6f7a8b9676dec9f8a1"}. Reused48 remain original BYTEVIEW results; new qualification maximum4 and formal maximum90 are separate. No full30 accuracy, no empty scientific CSV, no automatic rerun or prefix concatenation.

{
  "error": "RuntimeError('Unit worker exited unsuccessfully; inspect immutable log, no retry')",
  "status": "FAIL_CLOSED_STOP",
  "traceback": "Traceback (most recent call last):\n  File \"/dev/shm/qwen_wscale_rowbal_reuse48_archive29_20261025/scripts/gpu_controller.py\", line 85, in run\n    C.require(proc.returncode==0,'Unit worker exited unsuccessfully; inspect immutable log, no retry')\n  File \"/dev/shm/qwen_wscale_rowbal_reuse48_archive29_20261025/inherited/group32/src/inherited_device_guards.py\", line 7, in require\n    if not ok:raise RuntimeError(message)\nRuntimeError: Unit worker exited unsuccessfully; inspect immutable log, no retry\n",
  "unit": "GPU1_base_MATCHED4096"
}
