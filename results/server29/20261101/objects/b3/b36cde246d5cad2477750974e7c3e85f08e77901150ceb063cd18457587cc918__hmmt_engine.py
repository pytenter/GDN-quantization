"""Four fixed HMMT adapters over the inherited token-forward and sampler loop."""
import gc,hashlib,time
from pathlib import Path
import torch
import pace_runtime as R
from hmmt_adapters import adapter_for
from hmmt_sampling import sampler

class HMMTEngine:
    def __init__(self,root,config_id,model,tokenizer,native,progress):
        self.root=Path(root);self.config_id=config_id;self.model=model;self.tokenizer=tokenizer;self.native=native;self.progress=progress
        self.masks,self.mask_sha,_=R.load_head_mask(self.root/'calibration/head_mask.json')
        self.f=sampler(root,torch);self.device=torch.device('cuda:0')
    def generate(self,input_ids,*,problem_id,max_new_tokens=65536,diagnostic=False):
        assert input_ids and all(type(x) is int for x in input_ids)
        assert max_new_tokens==65536 or (diagnostic and 1<=max_new_tokens<=33)
        self.f.setup_seed(1)
        generator=torch.Generator(device=self.device).manual_seed(1)
        rngsha=lambda:hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()
        initial=rngsha();warpers=self.f.build_warpers();stops=self.f.stop_ids(self.tokenizer)
        mask=torch.zeros((1,self.model.get_output_embeddings().weight.shape[0]),dtype=torch.bool,device=self.device)
        tokens=[];survivors=[];cache=None;total=0;peak_kv=0
        adapter=adapter_for(self.config_id,self.model,self.native,self.masks,len(input_ids))
        def forward(token):
            nonlocal cache,total,peak_kv
            previous=[] if cache is None else [x.keys.shape[-2] for x in cache.layers if isinstance(getattr(x,'keys',None),torch.Tensor)]
            out=self.model(input_ids=torch.tensor([[token]],dtype=torch.long,device=self.device),past_key_values=cache,use_cache=True)
            cache=out.past_key_values;total+=1
            assert bool(torch.isfinite(out.logits).all())
            adapter.assert_cache(cache,total)
            attention=[x.keys.shape[-2] for x in cache.layers if isinstance(getattr(x,'keys',None),torch.Tensor)]
            assert len(attention)==8 and all(n==total for n in attention)
            if previous:assert len(set(previous))==1 and all(n==previous[0]+1 for n in attention)
            kv=sum(t.numel()*t.element_size() for x in cache.layers for t in [getattr(x,'keys',None),getattr(x,'values',None)] if isinstance(t,torch.Tensor))
            peak_kv=max(peak_kv,kv)
            if total%128==0:self.progress('FORWARD_COMMITTED',input_forwards=total,generated_tokens=len(tokens))
            return out
        with torch.inference_mode(),adapter:
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
            for token in input_ids:out=forward(token)
            torch.cuda.synchronize();prefill=time.monotonic()-start;decode_start=time.monotonic()
            while len(tokens)<max_new_tokens:
                current,count=self.f.sample_next(warpers,out.logits[:,-1,:],mask,generator)
                mask.scatter_(1,current,True);token=int(current.item());tokens.append(token);survivors.append(int(count))
                if len(tokens)%128==0 or token in stops or len(tokens)==max_new_tokens:
                    self.progress('GENERATED_TOKENS_COMMITTED',input_forwards=total,generated_tokens=len(tokens))
                if token in stops or len(tokens)==max_new_tokens:break
                out=forward(token)
            torch.cuda.synchronize();decode=time.monotonic()-decode_start;elapsed=time.monotonic()-start
            assert total==len(input_ids)+len(tokens)-1
            if hasattr(adapter,'buffers'):
                for b in adapter.buffers.values():
                    before=(b.tokens,b.pending_tokens,b.writeback_count);b.finalize('preserve')
                    assert before==(b.tokens,b.pending_tokens,b.writeback_count)
                    assert b.decode_tokens==len(tokens)-1 and b.pending_tokens==b.decode_tokens%b.decode_period
                    assert b.decode_dense_materializations==b.decode_tokens//b.decode_period
                    assert b.decode_direct_calls==b.decode_tokens and b.validation_dense_materializations==0
            audit=adapter.audit()
        result=dict(status='COMPLETE',config_id=self.config_id,seed=1,problem_id=problem_id,
            input_ids_sha256=self.f.ids_sha256(input_ids),input_tokens=len(input_ids),
            generated_token_ids=tokens,generated_token_ids_sha256=R.canonical_sha(tokens),generated_token_count=len(tokens),
            full_output=self.tokenizer.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False),
            response=self.tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False),
            finish_reason='eos' if tokens[-1] in stops else 'length',stop_token_ids=sorted(stops),
            actual_model_forward_calls=total,actual_decode_forwards=total-len(input_ids),sampler_calls=len(tokens),
            wall_seconds=elapsed,prompt_seconds=prefill,decode_and_sampling_seconds=decode,
            decode_forward_TPS=(len(tokens)-1)/decode,sampled_token_TPS=len(tokens)/decode,
            generation_TPS=len(tokens)/elapsed,head_mask_sha256=self.mask_sha,quantizer_audit=audit,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            full_attention_KV_peak_observed_payload_bytes=peak_kv,
            full_attention_KV_scope='Live payload observed after every forward; excludes transient concat and allocator overhead',
            initial_generator_sha256=initial,final_generator_sha256=rngsha(),
            sampling_audit=dict(top_k_parameter=20,min_survivors=min(survivors),max_survivors=max(survivors),
                                inherited_AST=True,presence_penalty=1.5,temperature=.6),
            actual_native_sha256=R.NATIVE_SHA256,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            nonfinite=False,runtime_error=None,diagnostic_only=diagnostic)
        del cache,out,adapter;gc.collect()
        return result
