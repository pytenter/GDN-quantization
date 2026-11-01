"""Post-forward token0 QDQ, with actual next native-helper input byte checks."""
from canonical_h128 import QwenKeyHadamardPatch,get_state
from int4_codec import encode,decode,require_consumed,tensor_bytes
import torch

class Int4PersistentQDQ:
    def __init__(self,model,method,layers,observe=True,callback=None):
        self.model=model;self.method=method;self.layers=tuple(layers)
        self.observe=observe;self.callback=callback;self.h=QwenKeyHadamardPatch(model)
        self.previous={};self.qdq_calls=0;self.consumption_calls=0;self.saved=[]
        self.consumption_layers=set();self.current_token=0

    def __enter__(self):
        import transformers.models.qwen3_5.modeling_qwen3_5 as Q
        self.h.__enter__()
        for name in ('torch_recurrent_gated_delta_rule','torch_chunk_gated_delta_rule'):
            original=getattr(Q,name);self.saved.append((Q,name,original))
            def wrap(fn=original,kind=name):
                def call(query,key,value,*args,**kwargs):
                    if query.shape[1]!=1:raise RuntimeError('Multi-token chunk violates per-token QDQ')
                    layer=int(self.h.current_layer)
                    state=kwargs.get('initial_state',args[2] if len(args)>2 else None)
                    if self.current_token>0 and self.observe:
                        if state is None or layer not in self.previous:
                            raise RuntimeError('Missing persistent state')
                        try:require_consumed(state,self.previous[layer])
                        except BaseException:
                            if self.callback:self.callback('NEXT_CONSUMPTION_FAILURE',self.current_token,layer,dict(actual=state,expected=decode(self.previous[layer])))
                            raise
                        self.consumption_calls+=1;self.consumption_layers.add(layer)
                        if self.callback:self.callback('KERNEL_INITIAL',self.current_token,layer,dict(actual=state))
                    return fn(query,key,value,*args,**kwargs)
                return call
            setattr(Q,name,wrap())
        return self

    def after_token(self,cache):
        if self.current_token!=self.qdq_calls:raise RuntimeError('QDQ sequence mismatch')
        for layer in self.layers:
            state=get_state(cache,layer)
            if state.dtype!=torch.float32:raise RuntimeError('Persistent carrier dtype changed')
            if self.callback:self.callback('PRE_QDQ',self.current_token,layer,dict(state=state))
            e=encode(state,self.method)
            d=decode(e);state.copy_(d)
            if self.observe:
                expected=e.cpu()
                require_consumed(state,expected)
                self.previous[layer]=expected  # Only integer codec fields/scales on CPU, no float recurrent shadow.
            if self.callback:
                values=dict(codes=e.codes,scales=e.scales,decoded=d,writeback=state)
                if e.zero_points is not None:values['zero_points']=e.zero_points
                if e.negative_zero_flags is not None:values['negative_zero_flags']=e.negative_zero_flags
                self.callback('POST_QDQ',self.current_token,layer,values)
        self.qdq_calls+=1;self.current_token+=1

    def __exit__(self,*args):
        for module,name,fn in self.saved:setattr(module,name,fn)
        self.h.__exit__(*args)

    def audit(self):
        return dict(method=self.method,tokens=self.qdq_calls,layer_count=len(self.layers),
                    qdq_layer_updates=self.qdq_calls*len(self.layers),token0_QDQ=self.qdq_calls>0,
                    next_consumption_checks=self.consumption_calls,consumption_layers=sorted(self.consumption_layers),
                    expected_checks=max(0,self.qdq_calls-1)*len(self.layers),
                    actual_values_and_bytes_compared=self.observe,unquantized_shadow=False,
                    recurrent_carrier='native_FP32_post_QDQ',packed_INT4_cache=False,logical_bits=4,codes_dtype="torch.uint8",integer_range=[0,15])
