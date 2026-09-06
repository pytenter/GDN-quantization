# Recurrent Update Semantics

{
  "RECURRENT_UPDATE_SEMANTICS_GATE": "PASS",
  "branch_coefficients": "natural branch may change future hidden-dependent q/k/v/beta/gate; this audit treats it as full-network natural propagation",
  "readout": "query is l2-normalized, transposed to [B,H,T,K], scaled by K**-0.5, then contracted over K",
  "source": "gdn_native_core_output_replay_semantics_audit_v1 plus current Qwen3.5 cached decode path",
  "state_layout": "[B,H,K,V]",
  "state_update": "torch_recurrent_gated_delta_rule updates cached recurrent state during single-token decode"
}
