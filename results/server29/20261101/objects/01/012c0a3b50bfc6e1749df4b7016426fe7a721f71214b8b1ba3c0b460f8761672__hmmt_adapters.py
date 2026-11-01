"""Task-local FP32 baseline and post-norm resident key-side H128.

C1/C3 retain byte-identical inherited adapters, buffers, codecs and kernel.
C2 stores base State in H basis; its unchanged physical K4 packets encode the
native preimage. At every use, original decode/BF16 norm precedes FP32 H.
No dense State inverse transform, additional K quantization or norm is used.
"""
import ast
import hashlib
import inspect
import math
import textwrap
import torch
import pace_runtime as R
from pace_buffer import PACEBuffer, native_l2norm
from lq_unroll_buffer import LQUnrollBuffer

_H128 = {}
_TRANSFORM_AUDIT = []

def hadamard128(device):
    device=torch.device(device)
    if device.type=='cuda' and device.index is None:
        device=torch.device('cuda',torch.cuda.current_device())
    label=str(device)
    if label not in _H128:
        h=torch.ones((1,1),dtype=torch.float32,device=device)
        while h.shape[0]<128:
            h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
        _H128[label]=h/math.sqrt(128)
    return _H128[label]

def rotate_fp32(x):
    assert x.dtype==torch.float32 and x.shape[-1]==128
    return x@hadamard128(x.device)

def postnorm_rotate(x):
    assert x.dtype==torch.bfloat16
    return rotate_fp32(native_l2norm(x).float())

def rotate_query_step(method, *, prefill_dispatch=False):
    """Copy inherited AST, inject H only after its one BF16 query norm.

    No inherited source is modified. This is also used by validation replay.
    The sole extra dispatch substitution avoids an unbound zero-arg super in
    the copied method and routes to the similarly transformed prefill method.
    """
    original=inspect.unwrap(method)
    source=textwrap.dedent(inspect.getsource(original))
    tree=ast.parse(source);fn=tree.body[0];fn.decorator_list=[]
    norm=0;dispatch=0
    class Insert(ast.NodeTransformer):
        def visit_Assign(self,node):
            nonlocal norm
            if (len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
                    and node.targets[0].id=='query' and isinstance(node.value,ast.Call)
                    and isinstance(node.value.func,ast.Attribute)
                    and isinstance(node.value.func.value,ast.Call)
                    and isinstance(node.value.func.value.func,ast.Name)
                    and node.value.func.value.func.id=='native_l2norm'):
                norm+=1
                addition=ast.parse('query = self._rotate(query)').body[0]
                return [node,ast.copy_location(addition,node)]
            return self.generic_visit(node)
        def visit_Call(self,node):
            nonlocal dispatch
            if (prefill_dispatch and isinstance(node.func,ast.Attribute) and node.func.attr=='step'
                    and isinstance(node.func.value,ast.Call)
                    and isinstance(node.func.value.func,ast.Name) and node.func.value.func.id=='super'):
                node.func=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='_rotated_prefill_step',ctx=ast.Load())
                dispatch+=1
            return self.generic_visit(node)
    tree=Insert().visit(tree);ast.fix_missing_locations(tree)
    assert norm==1 and dispatch==int(prefill_dispatch), 'Unexpected inherited AST; fail closed'
    namespace=dict(original.__globals__)
    exec(compile(tree,__file__+'::postnorm_'+original.__qualname__,'exec'),namespace)
    _TRANSFORM_AUDIT.append(dict(method=original.__qualname__,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        query_rotation_insertions=norm,prefill_dispatch_substitutions=dispatch,
        transformed_ast_sha256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()))
    return torch.no_grad()(namespace[fn.name])

class H128LQBuffer(LQUnrollBuffer):
    _rotate=staticmethod(rotate_fp32)
    _rotated_prefill_step=rotate_query_step(PACEBuffer.step)
    step=rotate_query_step(LQUnrollBuffer.step,prefill_dispatch=True)
    def _token_keys(self,token):
        return rotate_fp32(PACEBuffer._token_keys(self,token))
    def _batched_keys(self,records):
        # Parent validation calls self._token_keys; compare native keys before H.
        flag=self.verify_batched_keys;self.verify_batched_keys=False
        try: keys=LQUnrollBuffer._batched_keys(self,records)
        finally:self.verify_batched_keys=flag
        if flag:
            for i,record in enumerate(records):
                assert torch.equal(keys[i],PACEBuffer._token_keys(self,record))
                self.batched_key_exact_checks+=1
        return rotate_fp32(keys)
    def storage_report(self):
        row=super().storage_report()
        row.update(state_basis='H128_RESIDENT',physical_key_packet_basis='NATIVE_K4_PREIMAGE_UNCHANGED',
                   effective_key_query_basis='H128_AFTER_ORIGINAL_BF16_NORM_FP32',
                   per_boundary_inverse_rotation=False,endpoint_state_rotation=False,
                   rotation_matrix_scope='shared 128x128 FP32 per device, accounted once by adapter',
                   ast_transform_audit=_TRANSFORM_AUDIT)
        return row

class H128PACEAdapter(R.PACEAdapter):
    def audit(self):
        value=super().audit()
        value.update(rotation='FIXED_POSTNORM_FP32_H128_KEY_SIDE_RESIDENT',
            rotation_matrix_nbytes=128*128*4,
            qk_order='original native K4 decode -> BF16 norm -> FP32 H; query BF16 norm -> FP32 H',
            state_basis='H128',physical_key_cache_basis='NATIVE_K4_PREIMAGE',
            logical_decoded_key_basis='H128',value_rotated=False,
            endpoint_double_rotation=False,per_boundary_inverse_rotation=False)
        value['total_state_resident_tensor_nbytes']+=128*128*4
        return value

class NativeFPAdapter:
    """Observe actual native FP32 State cache; optional H is validation only."""
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
                assert options.get('use_qk_l2norm_in_kernel') is True
                self.h128=hadamard128(q.device)
                q,k=postnorm_rotate(q),postnorm_rotate(k)
                options['use_qk_l2norm_in_kernel']=False
                output,state=original(q,k,v,**options)
                output=output.to(torch.bfloat16)
            else:output,state=original(q,k,v,*args,**kwargs)
            assert state.dtype==torch.float32 and tuple(state.shape[-3:])==(32,128,128)
            assert output.dtype==torch.bfloat16
            self.calls[layer]+=q.shape[1]
            self.states[layer]=None  # cache copies into its static allocation later
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
            self.states[layer]=state  # alias real cache, never clone
    def audit(self):
        inventory=[]
        for layer,state in self.states.items():
            assert bool(torch.isfinite(state).all())
            inventory.append(dict(layer=layer,shape=list(state.shape),dtype=str(state.dtype),nbytes=state.numel()*state.element_size()))
        return dict(implementation='PINNED_NATIVE_FP32_STATE',layers=list(self.layers),helper_calls={str(k):v for k,v in self.calls.items()},
                    native_state_inventory=inventory,recurrent_state_nbytes=sum(x['nbytes'] for x in inventory),
                    state_quantization_calls=0,key_quantization_calls=0,buffer_payload_bytes=0,buffer_capacity_bytes=0,
                    rotation='H128_VALIDATION_ONLY' if self.rotation else None,rotation_matrix_nbytes=0 if self.h128 is None else self.h128.numel()*self.h128.element_size(),
                    dense_state_persistent=True,State_dtype='torch.float32')

def adapter_for(config_id,model,native,masks,prompt_length):
    if config_id=='C0':return NativeFPAdapter(model,native)
    assert config_id in ['C1','C2','C3']
    period=16 if config_id=='C3' else 8
    cls=H128PACEAdapter if config_id=='C2' else R.PACEAdapter
    factory=H128LQBuffer if config_id=='C2' else LQUnrollBuffer
    return cls(model,native,masks,period,prompt_length,buffer_factory=factory)
