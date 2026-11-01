"""Task-local native FP32 State and fixed resident Key-side H128 adapters.

C1/C3 instantiate the byte-identical old LQ buffer and PACEAdapter.
C2 inserts only a fixed H128 before the original one-time BF16 q/k norm.
The BF16 rotation cast is explicit and separately tested, never called exact.
"""
import math
import torch
import pace_runtime as R
from lq_unroll_buffer import LQUnrollBuffer

def hadamard128(device):
    h=torch.ones((1,1),dtype=torch.float32,device=device)
    while h.shape[0]<128:
        h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
    return h/math.sqrt(128)

def rotate_bf16(x,matrix):
    assert x.dtype==torch.bfloat16 and x.shape[-1]==128
    return (x.float()@matrix).to(torch.bfloat16)

class H128PACEAdapter(R.PACEAdapter):
    """Same original fast buffer; every q/k and all resident state use H basis."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.h128=None
        self.rotation_calls=0
    def _wrap(self,kind,original):
        parent=super()._wrap(kind,original)
        def call(q,k,v,*args,**kwargs):
            if self.h128 is None:self.h128=hadamard128(q.device)
            qh,kh=rotate_bf16(q,self.h128),rotate_bf16(k,self.h128)
            self.rotation_calls+=1
            # Original first-chunk compatibility check receives the same H inputs.
            # No endpoint state rotation and no inverse rotation at writeback.
            return parent(qh,kh,v,*args,**kwargs)
        return call
    def audit(self):
        value=super().audit()
        value.update(rotation='FIXED_H128_KEY_SIDE_RESIDENT',rotation_calls=self.rotation_calls,
                     rotation_matrix_nbytes=0 if self.h128 is None else self.h128.numel()*self.h128.element_size(),
                     qk_order='FP32 H matmul -> BF16 -> original BF16 norm (once); decode K4 before its original norm',
                     state_basis='H128',key_cache_basis='H128',value_rotated=False,
                     endpoint_double_rotation=False,per_boundary_inverse_rotation=False)
        return value

class NativeFPAdapter:
    """Observe original native State, never quantize, encode K4, or alter update."""
    def __init__(self,model,native,*,rotation=False):
        self.model=model;self.native=native;self.rotation=rotation
        self.layers=R.GDN_LAYERS;self.tracker=R.NativeLayerTracker(model,self.layers)
        self.calls={i:0 for i in self.layers};self.states={};self.saved=[];self.h128=None
    def __enter__(self):
        assert not getattr(self.native,'_pace48_active',False),'Nested adapter'
        self.native._pace48_active=True;self.tracker.__enter__()
        for kind,name in [('recurrent','torch_recurrent_gated_delta_rule'),('chunk','torch_chunk_gated_delta_rule')]:
            original=getattr(self.native,name);self.saved.append((name,original))
            setattr(self.native,name,self._wrap(kind,original))
        return self
    def _wrap(self,kind,original):
        def call(q,k,v,*args,**kwargs):
            layer=self.tracker.current_layer;assert layer in self.calls
            options=R._bind(kind,args,kwargs)
            assert options.get('output_final_state') is True
            previous=options.get('initial_state')
            if previous is not None:assert previous.dtype==torch.float32 and tuple(previous.shape[-3:])==(32,128,128)
            if self.rotation:
                if self.h128 is None:self.h128=hadamard128(q.device)
                q,k=rotate_bf16(q,self.h128),rotate_bf16(k,self.h128)
            output,state=original(q,k,v,*args,**kwargs)
            assert state.dtype==torch.float32 and tuple(state.shape[-3:])==(32,128,128)
            assert output.dtype==torch.bfloat16
            self.calls[layer]+=q.shape[1]
            # Native cache copies into a static-address allocation after return.
            # Never retain this transient computed State as a second dense copy.
            self.states[layer]=None
            return output,state
        return call
    def __exit__(self,*args):
        for name,original in reversed(self.saved):setattr(self.native,name,original)
        self.saved=[];self.tracker.__exit__();self.native._pace48_active=False
    def assert_cache(self,cache,total):
        assert set(self.states)==set(self.layers)
        for layer in self.layers:
            state=cache.layers[layer].recurrent_states[0]
            assert state.dtype==torch.float32 and state.shape==(1,32,128,128)
            assert self.calls[layer]==total
            self.states[layer]=state  # Alias the real cache allocation, no clone.
    def audit(self):
        inventory=[]
        for layer,state in self.states.items():
            assert bool(torch.isfinite(state).all())
            inventory.append(dict(layer=layer,shape=list(state.shape),dtype=str(state.dtype),nbytes=state.numel()*state.element_size()))
        return dict(implementation='PINNED_NATIVE_FP32_STATE',layers=list(self.layers),helper_calls=self.calls,
                    native_state_inventory=inventory,recurrent_state_nbytes=sum(x['nbytes'] for x in inventory),
                    state_quantization_calls=0,key_quantization_calls=0,buffer_payload_bytes=0,buffer_capacity_bytes=0,
                    rotation='H128_VALIDATION_ONLY' if self.rotation else None,rotation_matrix_nbytes=0 if self.h128 is None else self.h128.numel()*self.h128.element_size(),
                    dense_state_persistent=True,State_dtype='torch.float32')

def adapter_for(config_id,model,native,masks,prompt_length):
    if config_id=='C0':return NativeFPAdapter(model,native)
    period=16 if config_id=='C3' else 8
    cls=H128PACEAdapter if config_id=='C2' else R.PACEAdapter
    return cls(model,native,masks,period,prompt_length,buffer_factory=LQUnrollBuffer)
