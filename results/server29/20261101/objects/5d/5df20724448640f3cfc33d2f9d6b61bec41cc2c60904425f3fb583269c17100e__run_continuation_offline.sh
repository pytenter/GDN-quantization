#!/bin/bash
set -u
export PYTHONDONTWRITEBYTECODE=1
cd /data/zypan/analyses/qwen_gdn_c6r_dual_m_continuation_20261017/experiments/qwen_gdn/quantizer/QWEN_GDN_C6R_DUAL_M_INTERACTION_REUSE_5DOC_V1
/data/ydai/miniconda3/envs/bitdecode/bin/python -u /data/zypan/analyses/qwen_gdn_c6r_dual_m_continuation_20261017/experiments/qwen_gdn/quantizer/QWEN_GDN_C6R_DUAL_M_INTERACTION_REUSE_5DOC_V1/control/controller.py > /data/zypan/analyses/qwen_gdn_c6r_dual_m_continuation_20261017/experiments/qwen_gdn/quantizer/QWEN_GDN_C6R_DUAL_M_INTERACTION_REUSE_5DOC_V1/logs/CONTROLLER_OUTER.log 2>&1
controller_status=$?
/data/ydai/miniconda3/envs/bitdecode/bin/python -u /data/zypan/analyses/qwen_gdn_c6r_dual_m_continuation_20261017/experiments/qwen_gdn/quantizer/QWEN_GDN_C6R_DUAL_M_INTERACTION_REUSE_5DOC_V1/control/finish_continuation.py > /data/zypan/analyses/qwen_gdn_c6r_dual_m_continuation_20261017/experiments/qwen_gdn/quantizer/QWEN_GDN_C6R_DUAL_M_INTERACTION_REUSE_5DOC_V1/logs/OFFLINE_FINISH.log 2>&1
finish_status=$?
/data/ydai/miniconda3/envs/bitdecode/bin/python -c 'import json,sys,time; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps(dict(controller_exit=int(sys.argv[2]),finisher_exit=int(sys.argv[3]),ended_unix=time.time(),no_GPU_retry=True))+"\n")' /data/zypan/analyses/qwen_gdn_c6r_dual_m_continuation_20261017/experiments/qwen_gdn/quantizer/QWEN_GDN_C6R_DUAL_M_INTERACTION_REUSE_5DOC_V1/OFFLINE_EXIT_STATUS.json "$controller_status" "$finish_status"
exit "$((controller_status != 0 || finish_status != 0))"
