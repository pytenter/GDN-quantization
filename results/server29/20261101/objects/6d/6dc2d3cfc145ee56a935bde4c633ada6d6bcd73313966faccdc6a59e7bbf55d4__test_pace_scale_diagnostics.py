"""CPU-only failure-context checks; no Torch, CUDA, model, or answer imports."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pace_scale_diagnostics import scale_failure_context


class FP16ScaleOverflowError(ValueError):
    def __init__(self, diagnostic):
        super().__init__("affine group scale exceeds finite FP16 range")
        self.diagnostics = diagnostic


class BufferFixture:
    tokens, prompt_length, pending_tokens = 9217, 78, 1
    decode_period, prefill_period = 1, 256
    quantization_count, writeback_count = 9140, 9140


def run():
    identity = dict(pid=123, starttick=456, cmdline=["fixture"])
    assert scale_failure_context(ValueError("unrelated"), ROOT, "formal", "fixture", identity) is None
    numeric = dict(bits=4, first_bad_group_indices=[[0, 1, 17, 3]],
                   groups=[dict(index=[0, 1, 17, 3], raw_scale=70000.0,
                                effective_scale="+Inf")])
    namespace = dict(E=FP16ScaleOverflowError, diag=numeric, B=BufferFixture)
    exec(compile("def encode():\n heads=(2,9)\n raise E(diag)\n",
                 str(ROOT / "src/pace_quantizer.py"), "exec"), namespace)
    exec(compile("def _commit():\n reason='decode_period'\n return encode()\n",
                 str(ROOT / "src/pace_buffer.py"), "exec"), namespace)
    exec(compile("def call():\n layer=7\n buffer=B()\n return _commit()\n",
                 str(ROOT / "src/pace_runtime.py"), "exec"), namespace)
    try:
        namespace["call"]()
    except FP16ScaleOverflowError as error:
        result = scale_failure_context(error, ROOT, "formal", "fixture", identity)
    assert result["role"] == "RECURRENT_STATE_WRITEBACK"
    assert result["layer"] == 7 and result["buffer_counters"]["tokens"] == 9217
    assert result["state_group_locations"] == [dict(batch=0, head=9, value_column=17, key_group=3)]
    assert result["writeback_reason"] == "decode_period"
    assert result["formal_completion_credit"] == 0
    assert result["tensors_or_prompt_or_generated_text_recorded"] is False
    json.dumps(result, allow_nan=False)
    assert "generated_token_ids" not in result and "state" not in result
    return dict(status="PASS", cases=6, GPU_calls=0, model_loads=0, Torch_imported="torch" in sys.modules)


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
