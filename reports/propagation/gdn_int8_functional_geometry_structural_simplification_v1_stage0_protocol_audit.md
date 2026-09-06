# GDN_INT8_FUNCTIONAL_GEOMETRY_STRUCTURAL_SIMPLIFICATION_V1

## Stage 0 Protocol Audit

- TASK: `GDN_INT8_FUNCTIONAL_GEOMETRY_STRUCTURAL_SIMPLIFICATION_V1`
- timestamp: `2026-09-06 11:01:48 +0800`
- branch: `research-sync-2026-09-02`
- HEAD: `2842db12c873a98c23b23d4aad32dc552edd25f2`
- git_status_start: `?? experiments/propagation/run_int8_functional_geometry_structural_simplification.py
?? reports/propagation/gdn_int8_functional_geometry_structural_simplification_v1_stage0_protocol_audit.md
?? results/propagation/gdn_int8_functional_geometry_structural_simplification_v1_stage0_protocol_audit.json`
- source_cases: `/data/zypan/GDN-quantization/results/propagation/gdn_int8_functional_proxy_natural_config_generalization_v1_formal_raw_results.jsonl`
- matched_pairs_source: `/data/zypan/GDN-quantization/results/propagation/gdn_int8_functional_risk_beyond_reconstruction_controlled_v1_matched_pair_results.json`
- geometry_tensor_source: `/data/zypan/runs/gdn_int8_functional_orientation_low_rank_and_stability_audit_v1/geometry_tensors`
- geometry_tensor_count: `216`
- expected_geometry_tensor_count: `216`
- frozen_m5_formula: `M_G^2 = q^T E G_approx E^T q, G_t = A_t^T A_t, A_t = W_O D_g D_w J_RMS`
- isotropic_definition: `alpha I, alpha=trace(G)/d`
- diagonal_definition: `diag(diag(G)); no off-diagonal entries retained`
- head_block_definition: `blockdiag over true GDN [head,value] layout, H=32, V=128 contiguous value blocks`
- future_KL_use: `evaluation only; not used to fit iso/diag/block/static means`
- instrumentation: `post-hoc state/readout reconstruction only; no forward hook or model mutation`
- PROTOCOL_GATE: `PASS`
- BLOCK_SEMANTICS_GATE: `PASS`
- NO_FUTURE_INFORMATION_LEAKAGE_GATE: `PASS`
- GEOMETRY_APPROXIMATION_GATE: `PASS`
- M5_QUADRATIC_IDENTITY_GATE: `PASS`
- M5_QUADRATIC_IDENTITY_MAX_RELERR: `1.0590213118295197e-07`
- M5_QUADRATIC_IDENTITY_MEDIAN_RELERR: `2.753419336616738e-08`
- tensor_shape_checks_sample: `[{'unit_id': 'test/algebra/1332.json|64', 'layer': 0, 'state_shape': [1, 32, 128, 128], 'q_shape': [1, 32, 128], 'e_dim': 4096, 'G_shape': [4096, 4096]}, {'unit_id': 'test/algebra/1332.json|64', 'layer': 1, 'state_shape': [1, 32, 128, 128], 'q_shape': [1, 32, 128], 'e_dim': 4096, 'G_shape': [4096, 4096]}, {'unit_id': 'test/algebra/1332.json|64', 'layer': 2, 'state_shape': [1, 32, 128, 128], 'q_shape': [1, 32, 128], 'e_dim': 4096, 'G_shape': [4096, 4096]}, {'unit_id': 'test/algebra/1332.json|64', 'layer': 4, 'state_shape': [1, 32, 128, 128], 'q_shape': [1, 32, 128], 'e_dim': 4096, 'G_shape': [4096, 4096]}, {'unit_id': 'test/algebra/1332.json|64', 'layer': 5, 'state_shape': [1, 32, 128, 128], 'q_shape': [1, 32, 128], 'e_dim': 4096, 'G_shape': [4096, 4096]}, {'unit_id': 'test/algebra/1332.json|64', 'layer': 6, 'state_shape': [1, 32, 128, 128], 'q_shape': [1, 32, 128], 'e_dim': 4096, 'G_shape': [4096, 4096]}, {'unit_id': 'test/algebra/1332.json|64', 'layer': 8, 'state_shape': [1, 32, 128, 128], 'q_shape': [1, 32, 128], 'e_dim': 4096, 'G_shape': [4096, 4096]}, {'unit_id': 'test/algebra/1332.json|64', 'layer': 9, 'state_shape': [1, 32, 128, 128], 'q_shape': [1, 32, 128], 'e_dim': 4096, 'G_shape': [4096, 4096]}, {'unit_id': 'test/algebra/1332.json|64', 'layer': 10, 'state_shape': [1, 32, 128, 128], 'q_shape': [1, 32, 128], 'e_dim': 4096, 'G_shape': [4096, 4096]}, {'unit_id': 'test/algebra/1332.json|64', 'layer': 12, 'state_shape': [1, 32, 128, 128], 'q_shape': [1, 32, 128], 'e_dim': 4096, 'G_shape': [4096, 4096]}]`
