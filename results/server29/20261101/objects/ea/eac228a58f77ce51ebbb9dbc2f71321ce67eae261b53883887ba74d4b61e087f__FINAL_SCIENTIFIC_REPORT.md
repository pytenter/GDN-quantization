# True STOP preserved

Status: PARTIAL_RESULTS_PRESERVED_AND_STOP. Actual complete legal counts: {"W": 0, "R": 0, "RW": 0}. No full30 accuracy, no empty scientific CSV, no automatic rerun or prefix concatenation.

{
  "error": "RuntimeError('Independent output filesystem headroom below4GiB')",
  "status": "FAIL_CLOSED_STOP",
  "traceback": "Traceback (most recent call last):\n  File \"/tmp/QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1_BYTEVIEW_REPAIR_20261025/scripts/gpu_controller.py\", line 79, in run\n    lastpoll=now;C.storage_guard(R)\n  File \"/tmp/QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1_BYTEVIEW_REPAIR_20261025/src/execution_control.py\", line 107, in storage_guard\n    s=os.statvfs(root);require(s.f_bavail*s.f_frsize>=4*1024**3,'Independent output filesystem headroom below4GiB')\n  File \"/tmp/QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1_BYTEVIEW_REPAIR_20261025/inherited/group32/src/inherited_device_guards.py\", line 7, in require\n    if not ok:raise RuntimeError(message)\nRuntimeError: Independent output filesystem headroom below4GiB\n",
  "unit": "GPU1_base_MATCHED4096"
}
