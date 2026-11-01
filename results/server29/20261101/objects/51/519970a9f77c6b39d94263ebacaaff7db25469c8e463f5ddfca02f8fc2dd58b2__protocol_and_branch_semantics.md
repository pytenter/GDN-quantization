# KDA_ROTATION_FULL_TRANSITION_FINAL_CAUSAL_CLOSURE_V1

## Protocol and branch semantics

TASK = "KDA_ROTATION_FULL_TRANSITION_FINAL_CAUSAL_CLOSURE_V1"
causal_unit = "actual live recurrent transition path"
NN = "Native quantized history, actual Native future"
NR = "Native quantized history, actual live Rotated future"
RN = "Rotated quantized history, actual Native future"
RR = "Rotated quantized history, actual live Rotated future"
switch_N_to_R = "rotate stored KDA state on V axis; preserve full cache positions/metadata; future probe rotates v and maps output back"
switch_R_to_N = "inverse-rotate stored KDA state on V axis; preserve full cache positions/metadata"
boundary_only_QR = "FORBIDDEN_AND_ABSENT"
historical_R_STATE_N = "persistent Native donor post-update PRE-QUANT state before current readout; Rotated receiver then INT8-quantized"
R_PREQUANT_N = "exact historical R_STATE_N semantics"
R_POSTQUANT_N = "true Native post-quant donor substituted at the same current-readout hook, then Rotated receiver quantization"
R_FP32_DRIVER = "not a distinct intervention: current live code already rotates v in float32 then casts at kernel boundary"
R_FP32_TRANSITION = "current branch drivers and state update in float32, unchanged Rotated INT8_R128 write-back"
