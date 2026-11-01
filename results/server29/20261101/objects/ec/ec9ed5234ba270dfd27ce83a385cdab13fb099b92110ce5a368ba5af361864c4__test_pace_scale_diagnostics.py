"""CPU-only failure-context checks; no Torch, CUDA, model, or answer imports."""
import json
import importlib.util
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
    foreign = {}
    exec(compile("def fail():\n raise ValueError('extended scale is not positive finite FP32')\n",'/foreign/pace_quantizer.py','exec'),foreign)
    try:foreign['fail']()
    except ValueError as error:
        assert scale_failure_context(error,ROOT,'formal','fixture',identity)is None
    actual = run_actual_range_guards(identity) if importlib.util.find_spec('torch')is not None else dict(status='NOT_RUN_LOCAL_TORCH_UNAVAILABLE')
    return dict(status="PASS", cases=7, GPU_calls=0, model_loads=0, Torch_imported="torch" in sys.modules,
                actual_candidate_range_guards=actual)


def run_actual_range_guards(identity):
    import torch
    from pace_quantizer import PackedAffineState
    rows=[]
    for high,message in [(1e101,'INT8 scale exponent exceeds supported range'),
                         (1e42,'extended scale is not positive finite FP32'),
                         (6e38,'extended decoded FP32 carrier is nonfinite')]:
        state=torch.zeros((1,32,32,1),dtype=torch.float64)
        state[0,10,0,0]=high
        mask=[h<8 for h in range(32)]
        try:PackedAffineState.from_tensor(state,mask)
        except ValueError as error:
            assert str(error)==message
            result=scale_failure_context(error,ROOT,'formal','fixture',identity)
        else:raise AssertionError('Actual candidate range guard missing')
        assert result['numeric']['status']=='ACTUAL_EXPONENT_FORMAT_RANGE_GUARD_FAILURE'
        assert result['formal_completion_credit']==0
        assert len(result['numeric']['groups'])==1
        assert result['state_group_locations']==[dict(batch=0,head=10,value_column=0,key_group=0)]
        assert result['numeric']['diagnostic_group_limit']==8
        assert result['tensors_or_prompt_or_generated_text_recorded']is False
        json.dumps(result,allow_nan=False)
        rows.append(dict(error=message,context=result))
    return dict(status='PASS_ACTUAL_CPU_TORCH_RANGE_GUARDS',cases=3,rows=rows,actual_cuda=False,model_loads=0)


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
