# KDA_ROTATION_PREFILL_KERNEL_EQUIVARIANCE_CAUSAL_V1

PACKED_PREFILL_KERNEL = "fla.ops.kda.chunk.chunk_kda / ChunkKDAFunction.apply"
PACKED_PREFILL_STATE_LAYOUT = "[B,HV,K,V] float32"
PACKED_PREFILL_INPUT_DTYPE = {"beta": "torch.float32", "g": "torch.float32", "k": "torch.bfloat16", "q": "torch.bfloat16", "v": "torch.bfloat16"}
PACKED_PREFILL_ACCUMULATION_DTYPE = "mixed Triton FP32 accumulation; final state float32"
PACKED_PREFILL_CHUNK_SIZE = 64
PACKED_PREFILL_SCAN_TYPE = "chunk scan with parallel intra-chunk solve"
PACKED_PREFILL_STATE_CARRY_SEMANTICS = "float32 final recurrent state [B,HV,K,V]"
PACKED_PREFILL_QUANTIZER_LOCATION = "outside kernel, once after packed prefill returns"
PACKED_PREFILL_VALUE_ROTATION_LOCATION = "v -> vR immediately before live KDA operator; state stored in rotated V coordinates"
PACKED_PREFILL_CAST_POINTS = "q/k/v loaded to FP32 inside Triton; output cast to v dtype; state emitted FP32"
DECODE_KERNEL = "fla.ops.kda.fused_recurrent.fused_recurrent_kda"
DECODE_STATE_LAYOUT = "[B,HV,K,V] float32"
DECODE_INPUT_DTYPE = "q/k/v model dtype; activated decay and beta float32"
DECODE_ACCUMULATION_DTYPE = "Triton FP32 recurrent accumulator"
DECODE_STATE_CARRY_SEMANTICS = "previous float32 state consumed directly and next float32 state returned"
DECODE_QUANTIZER_LOCATION = "outside kernel, after each token transition"
DECODE_VALUE_ROTATION_LOCATION = "v -> vR before kernel; output mapped by R.T; state remains rotated until boundary conversion"
DECODE_CAST_POINTS = "q/k/v loaded to FP32; output cast to v dtype; state remains FP32"
PACKED_VS_DECODE_DIFFERENCES = "chunk_kda uses 64-token chunk scan and parallel intra-chunk algebra; decode uses direct sequential fused recurrence with one carried FP32 state"
