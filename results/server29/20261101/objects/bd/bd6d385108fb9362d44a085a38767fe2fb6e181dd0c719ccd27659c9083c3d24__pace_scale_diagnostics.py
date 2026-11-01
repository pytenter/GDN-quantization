"""Failure-only PACE scale context; no model, tensor, or CUDA operations."""
from pathlib import Path


def scale_failure_context(exc, root, stage, unit, identity):
    numeric = getattr(exc, "diagnostics", None)
    if type(exc).__name__ != "FP16ScaleOverflowError" or not isinstance(numeric, dict):
        return None
    source = Path(root).resolve() / "src"
    context = dict(status="ACTUAL_UNREPRESENTABLE_STORED_FP16_SCALE", stage=stage,
                   unit=unit, identity=identity, numeric=numeric,
                   formal_completion_credit=0, no_parameter_change=True,
                   tensors_or_prompt_or_generated_text_recorded=False)
    heads = None
    tb = exc.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        path = Path(frame.f_code.co_filename).resolve()
        name = frame.f_code.co_name
        values = frame.f_locals
        if path == source / "pace_quantizer.py" and name == "encode":
            selected = values.get("heads")
            if isinstance(selected, tuple) and all(type(x) is int for x in selected):
                heads = selected
                context["state_selected_heads"] = list(selected)
        if path == source / "pace_buffer.py" and name == "_commit":
            context["role"] = "RECURRENT_STATE_WRITEBACK"
            reason = values.get("reason")
            if isinstance(reason, str):
                context["writeback_reason"] = reason
        if path == source / "pace_runtime.py" and name == "call":
            layer = values.get("layer")
            if type(layer) is int:
                context["layer"] = layer
            buffer = values.get("buffer")
            if buffer is not None:
                counters = {}
                for field in ("tokens", "prompt_length", "pending_tokens", "decode_period",
                              "prefill_period", "quantization_count", "writeback_count"):
                    value = getattr(buffer, field, None)
                    if type(value) is int:
                        counters[field] = value
                context["buffer_counters"] = counters
        tb = tb.tb_next
    context.setdefault("role", "AFFINE_GROUP_ENCODING")
    mapped = []
    if heads is not None:
        for index in numeric.get("first_bad_group_indices", []):
            if (isinstance(index, list) and len(index) == 4
                    and all(type(x) is int for x in index)
                    and 0 <= index[1] < len(heads)):
                mapped.append(dict(batch=index[0], head=heads[index[1]],
                                   value_column=index[2], key_group=index[3]))
    context["state_group_locations"] = mapped
    return context
