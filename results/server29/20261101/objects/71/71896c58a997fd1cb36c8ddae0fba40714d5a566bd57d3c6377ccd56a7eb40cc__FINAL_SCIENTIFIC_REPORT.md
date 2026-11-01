# Group64真实STOP交付

状态：PARTIAL_RESULTS_PRESERVED_AND_STOP。实际合法完成数：{"G64-0": 0, "G64-1": 1, "G64-2": 0, "G64-3": 0}。未完成120，不输出完整准确率或伪造空科学CSV。保留首因、原日志、原失败和真实退出释放，无GPU重试/轨迹拼接/热改。

{
  "earliest_physical_cross_host_cause_order": "UNRESOLVED; retain each node actual first cause",
  "error": "RuntimeError('Node scientific first failure: server29_gpu6 {\\'error\\': \"RuntimeError(\\'Foreign process on assigned GPU; never kill it\\')\", \\'status\\': \\'FAIL_CLOSED_STOP\\', \\'traceback\\': \\'Traceback (most recent call last):\\\\n  File \"/data/zypan/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1/revisions/three_gpu_ledger_repair_v6/scripts/gpu_controller.py\", line 41, in run\\\\n    C.device(gpu,child[\\\\\\'pid\\\\\\']);gpu_first=now if gpu_first is None else gpu_first;gpu_last=now;last_gpu_owned=now\\\\n  File \"/data/zypan/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1/revisions/three_gpu_ledger_repair_v6/src/execution_control.py\", line 54, in device\\\\n    require(not [x for x in obs[\\\\\\'apps\\\\\\'] if x[\\\\\\'uuid\\\\\\']==UUIDS[gpu] and x[\\\\\\'pid\\\\\\']!=own_pid],\\\\\\'Foreign process on assigned GPU; never kill it\\\\\\')\\\\n  File \"/data/zypan/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1/revisions/three_gpu_ledger_repair_v6/src/inherited_device_guards.py\", line 7, in require\\\\n    if not ok:raise RuntimeError(message)\\\\nRuntimeError: Foreign process on assigned GPU; never kill it\\\\n\\', \\'unit\\': \\'aime26_29_G64-2_seed1\\'}')",
  "identity": {
    "cmdline": [
      "/data/ydai/miniconda3/envs/bitdecode/bin/python",
      "-u",
      "-B",
      "/data/zypan/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1/revisions/three_gpu_ledger_repair_v6/scripts/global_coordinator.py",
      "--root",
      "/data/zypan/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1/revisions/three_gpu_ledger_repair_v6"
    ],
    "pid": 49064,
    "starttick": 7144358136
  },
  "monotonic_on55": 71445502.86563991,
  "status": "GLOBAL_TRUE_STOP",
  "traceback": "Traceback (most recent call last):\n  File \"/data/zypan/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1/revisions/three_gpu_ledger_repair_v6/scripts/global_coordinator.py\", line 88, in <module>\n    if v.get('FIRST_FAILURE.json')or v.get('SESSION_RESULT.json',{}).get('status')=='FAIL_CLOSED_STOP':raise RuntimeError('Node scientific first failure: '+name+' '+repr(v.get('FIRST_FAILURE.json')))\nRuntimeError: Node scientific first failure: server29_gpu6 {'error': \"RuntimeError('Foreign process on assigned GPU; never kill it')\", 'status': 'FAIL_CLOSED_STOP', 'traceback': 'Traceback (most recent call last):\\n  File \"/data/zypan/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1/revisions/three_gpu_ledger_repair_v6/scripts/gpu_controller.py\", line 41, in run\\n    C.device(gpu,child[\\'pid\\']);gpu_first=now if gpu_first is None else gpu_first;gpu_last=now;last_gpu_owned=now\\n  File \"/data/zypan/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1/revisions/three_gpu_ledger_repair_v6/src/execution_control.py\", line 54, in device\\n    require(not [x for x in obs[\\'apps\\'] if x[\\'uuid\\']==UUIDS[gpu] and x[\\'pid\\']!=own_pid],\\'Foreign process on assigned GPU; never kill it\\')\\n  File \"/data/zypan/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1/revisions/three_gpu_ledger_repair_v6/src/inherited_device_guards.py\", line 7, in require\\n    if not ok:raise RuntimeError(message)\\nRuntimeError: Foreign process on assigned GPU; never kill it\\n', 'unit': 'aime26_29_G64-2_seed1'}\n"
}
