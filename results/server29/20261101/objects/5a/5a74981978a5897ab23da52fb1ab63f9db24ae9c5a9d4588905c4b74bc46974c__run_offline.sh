#!/bin/bash
set -u
cd /data/zypan/worktrees/qwen-gdn-c6r-dual-m-interaction-reuse-5doc-v1
/data/ydai/miniconda3/envs/bitdecode/bin/python -u /data/zypan/analyses/qwen_gdn_c6r_dual_m_interaction_reuse_5doc_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_C6R_DUAL_M_INTERACTION_REUSE_5DOC_V1/control/coordinator.py
controller_status=$?
/data/ydai/miniconda3/envs/bitdecode/bin/python -u /data/zypan/analyses/qwen_gdn_c6r_dual_m_interaction_reuse_5doc_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_C6R_DUAL_M_INTERACTION_REUSE_5DOC_V1/analysis_scripts/offline_finish.py > /data/zypan/analyses/qwen_gdn_c6r_dual_m_interaction_reuse_5doc_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_C6R_DUAL_M_INTERACTION_REUSE_5DOC_V1/logs/OFFLINE_FINISH.log 2>&1
finish_status=$?
exit "$((controller_status != 0 || finish_status != 0))"
