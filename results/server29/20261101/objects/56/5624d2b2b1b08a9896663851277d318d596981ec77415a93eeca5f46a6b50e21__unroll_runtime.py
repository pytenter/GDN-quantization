"""Original generation loop with one explicit adapter-factory substitution.
Sampling, prompt forwarding, EOS, cap, preserve and memory/timing are inherited.
"""
import gc, hashlib, time
from pace_runtime import PACEEngine, PACEAdapter, TASK, NATIVE_SHA256, canonical_sha, file_sha, _require, _progress, _load_frozen_model
from lq_unroll_buffer import LQUnrollBuffer
TASK = 'PACE48_LEAP_UNROLL_DP8_DP16_V1'

class UnrollEngine(PACEEngine):
    def __init__(self, root, period, **kwargs):
        if period not in (8,16): raise ValueError("Only8/16")
        super().__init__(root,'R2',**kwargs)
        self.period=period
        self.condition='LQ'+str(period)
        self.buffer_factory=LQUnrollBuffer

    @classmethod
    def from_loader(cls,root,period,**kwargs):
        _progress(kwargs.get('progress'),'LOAD_MODEL')
        model,tokenizer,native=_load_frozen_model(root)
        obj=cls(root,period,model=model,tokenizer=tokenizer,native_module=native,**kwargs)
        _progress(kwargs.get('progress'),'MODEL_LOADED')
        return obj

    def generate(self, input_ids, *, problem_id, max_new_tokens=81920):
            import torch
            _require(isinstance(input_ids, list) and input_ids and all(type(x) is int for x in input_ids),
                     "Generation needs explicit frozen input ids")
            _require(type(max_new_tokens) is int and max_new_tokens > 0, "Invalid generation cap")
            self.f.setup_seed(1)
            generator = torch.Generator(device=self.device).manual_seed(1)
            initial_generator_sha = hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()
            warpers, stops = self.f.build_warpers(), self.f.stop_ids(self.tokenizer)
            vocab = self.model.get_output_embeddings().weight.shape[0]
            mask = torch.zeros((1, vocab), dtype=torch.bool, device=self.device)
            tokens, survivors, cache = [], [], None
            total, start = 0, time.monotonic()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            adapter = PACEAdapter(self.model, self.native, self.masks, self.period, len(input_ids), buffer_factory=self.buffer_factory)

            def step(token):
                nonlocal cache, total
                previous_attention_lengths = ([] if cache is None else
                    [layer.keys.shape[-2] for layer in cache.layers
                     if isinstance(getattr(layer, "keys", None), torch.Tensor)])
                out = self.model(input_ids=torch.tensor([[token]], dtype=torch.long, device=self.device),
                                 past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                total += 1
                _require(bool(torch.isfinite(out.logits).all()), "Nonfinite model logits")
                adapter.assert_cache(cache, total)
                actual = [layer.keys.shape[-2] for layer in cache.layers
                          if isinstance(getattr(layer, "keys", None), torch.Tensor)]
                if previous_attention_lengths:
                    _require(len(set(previous_attention_lengths)) == 1
                             and actual and all(x == previous_attention_lengths[0] + 1 for x in actual),
                             "Native full-attention cache did not advance exactly one token")
                if total % 128 == 0:
                    _progress(self.progress, "FORWARD_COMMITTED", input_forwards=total,
                              generated_tokens=len(tokens), problem_id=problem_id, condition=self.condition)
                return out

            with torch.inference_mode(), adapter:
                for token in input_ids:
                    out = step(token)
                prefill_seconds = time.monotonic() - start
                while len(tokens) < max_new_tokens:
                    current, count = self.f.sample_next(warpers, out.logits[:, -1, :], mask, generator)
                    mask.scatter_(1, current, True)
                    token = int(current.item())
                    tokens.append(token)
                    survivors.append(int(count))
                    if len(tokens) % 128 == 0 or token in stops or len(tokens) == max_new_tokens:
                        _progress(self.progress, "GENERATED_TOKENS_COMMITTED", input_forwards=total,
                                  generated_tokens=len(tokens), problem_id=problem_id, condition=self.condition)
                    if token in stops or len(tokens) == max_new_tokens:
                        break
                    out = step(token)
                _require(total == len(input_ids) + len(tokens) - 1, "Prompt/decode forward count differs")
                for buffer in adapter.buffers.values():
                    buffer.finalize(policy="preserve")
                audit = adapter.audit()
            elapsed = time.monotonic() - start
            gpu_memory = (dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
                          if self.device.type == "cuda" else {})
            result = dict(status="COMPLETE", task_id=TASK, condition=self.condition,
                          configuration="PACE48_LQUNROLL_DP" + str(self.period) + "_V1",
                          seed=1, problem_id=problem_id, input_ids_sha256=self.f.ids_sha256(input_ids),
                          input_tokens=len(input_ids), generated_token_ids=tokens,
                          generated_token_ids_sha256=canonical_sha(tokens), generated_token_count=len(tokens),
                          response=self.tokenizer.decode(tokens, skip_special_tokens=True),
                          finish_reason="eos" if tokens[-1] in stops else "length",
                          stop_token_ids=sorted(stops), wall_seconds=elapsed, prompt_seconds=prefill_seconds,
                          decode_and_sampling_seconds=elapsed-prefill_seconds, TPS=len(tokens)/elapsed,
                          actual_model_forward_calls=total, sampler_calls=len(tokens), nonfinite=False,
                          runtime_error=None, head_mask_sha256=self.head_mask_sha,
                          quantizer_audit=audit, memory=gpu_memory, **gpu_memory,
                          initial_generator_sha256=initial_generator_sha,
                          final_generator_sha256=hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest(),
                          sampling_audit=dict(top_k_parameter=20, min_survivors=min(survivors),
                                              max_survivors=max(survivors), original_warper_unchanged=True),
                          protocol_identity={name: file_sha(self.root / "protocol" / name)
                                             for name in ("sampler.py", "common.py", "aime26_scorer_v4.py", "aime26_frozen.jsonl")},
                          actual_native_sha256=NATIVE_SHA256,
                          deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                          implementation_identity="UNVERIFIED_AGAINST_OFFICIAL_CODE")
            del cache, out, adapter
            gc.collect()
            return result
