"""Isolated Canonical V2 cache boundary, no KDA/model/kernel mathematics changes.

begin -> caller's existing full-model step/logits -> after_model_step QDQ ->
actual next capture.kernel initial_state bytes. CPU callback fixtures are NOT GPU
qualification. GPU execution uses only an injected explicitly authorized backend.
"""
from __future__ import annotations
import copy
import io
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from canonical_codec import VERSION,encode_torch,encode


def fingerprint(torch,value):
    host=value.detach().to(device='cpu',copy=True).contiguous()
    return dict(shape=list(host.shape),dtype=str(host.dtype),
                sha256=hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest())


def compare_stage_bits(actual,reference):
    if actual is None or reference is None or set(actual)!=set(reference):
        raise AssertionError('CANONICAL_STAGE_KEY_COVERAGE_MISMATCH')
    for key in reference:
        a,b=np.asarray(actual[key]),np.asarray(reference[key])
        if a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():
            mismatch=None
            if a.shape==b.shape:
                flat=np.flatnonzero(a.reshape(-1)!=b.reshape(-1))
                if flat.size:mismatch=int(flat[0])
            raise AssertionError(('CANONICAL_STAGE_BITS_MISMATCH',key,mismatch))
    return {'status':'PASS_EXACT','stage_keys':list(reference),'byte_tolerance':0}


class StageEvidenceWriter:
    """Durable complete snapshots, isolated execution evidence directory only.

    The receipt is fsynced after all arrays and context, before error propagation.
    An existing snapshot is never overwritten. The first failure pointer is O_EXCL.
    """
    def __init__(self,root,max_total_bytes=None,max_snapshot_bytes=None,failure_reserve_bytes=0):
        self.root=Path(root)
        self.max_total_bytes=max_total_bytes;self.max_snapshot_bytes=max_snapshot_bytes
        self.failure_reserve_bytes=failure_reserve_bytes
        for v in (max_total_bytes,max_snapshot_bytes):
            if v is not None and (type(v) is not int or v<=0):raise ValueError('POSITIVE_INTEGER_EVIDENCE_CAP_REQUIRED')
        if type(failure_reserve_bytes) is not int or failure_reserve_bytes<0:raise ValueError('INVALID_FAILURE_RESERVE')
        if max_total_bytes is not None and failure_reserve_bytes>=max_total_bytes:raise ValueError('FAILURE_RESERVE_MUST_FIT_TOTAL')
        if max_snapshot_bytes is not None and max_total_bytes is not None and max_snapshot_bytes>max_total_bytes:raise ValueError('SNAPSHOT_CAP_MUST_FIT_TOTAL')

    def bounded(self):
        return self.max_total_bytes is not None and self.max_snapshot_bytes is not None and self.failure_reserve_bytes>=self.max_snapshot_bytes

    def write(self,event,payload):
        failure=event=='canonical_v2_first_difference_evidence'
        name=('failure' if failure else 'snapshot')+'_p'+str(payload['position'])+'_l'+str(payload['layer'])
        directory=self.root/name
        refs=[];prepared=[]
        def save(relative,array):
            buffer=io.BytesIO();np.save(buffer,np.asarray(array),allow_pickle=False);raw=buffer.getvalue()
            prepared.append((relative,raw))
            refs.append(dict(relative_path=relative,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),shape=list(np.asarray(array).shape),dtype=str(np.asarray(array).dtype)))
        if payload.get('pre_qdq') is not None:save('preQDQ.npy',payload['pre_qdq'])
        for side in ('actual_stages','canonical_CPU_stages'):
            stages=payload.get(side)
            if stages is not None:
                for key,array in stages.items():save(side+'__'+key+'.npy',array)
        context={k:v for k,v in payload.items() if k not in ('pre_qdq','actual_stages','canonical_CPU_stages')}
        context['event']=event
        ctx_raw=(json.dumps(context,indent=2,allow_nan=False)+'\n').encode()
        prepared.append(('CONTEXT.json',ctx_raw))
        refs.append(dict(relative_path='CONTEXT.json',bytes=len(ctx_raw),sha256=hashlib.sha256(ctx_raw).hexdigest()))
        receipt=dict(status='DURABLE_FIRST_DIFFERENCE_EVIDENCE' if failure else 'DURABLE_CANONICAL_STAGE_SNAPSHOT',codec=VERSION,event=event,position=payload['position'],layer=payload['layer'],files=refs)
        path=directory/'SNAPSHOT_RECEIPT.json'
        receipt_raw=(json.dumps(receipt,indent=2)+'\n').encode();prepared.append(('SNAPSHOT_RECEIPT.json',receipt_raw))
        binding=dict(receipt=str(path),receipt_sha256=hashlib.sha256(receipt_raw).hexdigest())
        pointer_raw=(json.dumps(binding,indent=2)+'\n').encode()
        extra_pointer=len(pointer_raw) if failure and not (self.root/'FIRST_NUMERICAL_FAILURE.json').exists() else 0
        required=sum(len(raw) for _,raw in prepared)+extra_pointer
        current=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file()) if self.root.exists() else 0
        if self.max_snapshot_bytes is not None and required>self.max_snapshot_bytes:raise RuntimeError('EVIDENCE_SNAPSHOT_HARD_CAP')
        limit=None if self.max_total_bytes is None else self.max_total_bytes-(0 if failure else self.failure_reserve_bytes)
        if limit is not None and current+required>limit:raise RuntimeError('EVIDENCE_TOTAL_CAP_OR_FAILURE_RESERVE')
        # Admission precedes mkdir and all writes. Files include exact NPY headers,
        # context/receipt/pointer bytes, so successful snapshots cannot use reserve.
        directory.mkdir(parents=True,exist_ok=False)
        for relative,raw in prepared:
            with (directory/relative).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        fd=os.open(str(directory),os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        if failure:
            try:
                with (self.root/'FIRST_NUMERICAL_FAILURE.json').open('xb') as f:f.write(pointer_raw);f.flush();os.fsync(f.fileno())
            except FileExistsError:pass
        fd=os.open(str(self.root),os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return binding


class CanonicalV2RuntimeAdapter:
    def __init__(self,torch,capture,direction,layers,*,evidence_callback=None,
                 gpu_stage_authorized=False,cuda_backend=None,
                 frozen_h_fingerprints=None,audit_mode='CANARY_STRICT',probe_positions=(),evidence_root=None,evidence_limits=None):
        self.torch,self.capture,self.direction=torch,capture,direction
        self.layers=tuple(int(x) for x in layers)
        if not self.layers or len(set(self.layers))!=len(self.layers):raise ValueError('UNIQUE_FROZEN_KDA_LAYERS_REQUIRED')
        if audit_mode not in ('CANARY_STRICT','PILOT_LIGHT'):raise ValueError('EXPLICIT_FROZEN_AUDIT_MODE_REQUIRED')
        self.callback=evidence_callback;self.authorized=gpu_stage_authorized is True
        self.evidence_writer=None if evidence_root is None else StageEvidenceWriter(evidence_root,**(evidence_limits or {}))
        self.cuda_backend=cuda_backend;self.h_fingerprints=frozen_h_fingerprints
        self.audit_mode=audit_mode;self.probes=frozenset(int(x) for x in probe_positions)
        self.position=None;self.last=-1;self.called=set();self.previous={};self.valid_h=set()
        self.writes=self.consumed=self.stage_parities=0;self.parity_layers=set()
        self.peak_clone_bytes=0;self.observed_devices=set();self.actual_cuda_encode_count=0
        self.installed=False;self.original_kernel=None

    def _emit(self,event,payload):
        if event in ('canonical_v2_actual_stage_snapshot','canonical_v2_first_difference_evidence') and self.evidence_writer is not None:
            payload=dict(payload,durable_evidence=self.evidence_writer.write(event,payload))
        if self.callback is not None:self.callback(event,copy.deepcopy(payload))

    def _guard(self,value):
        if isinstance(value,self.torch.Tensor):
            if value.device.type=='cuda' and not self.authorized:raise PermissionError('GPU_STAGE_NOT_AUTHORIZED')
        elif isinstance(value,dict):
            for v in value.values():self._guard(v)
        elif isinstance(value,(list,tuple)):
            for v in value:self._guard(v)

    def install(self):
        if self.installed:raise RuntimeError('ADAPTER_ALREADY_INSTALLED')
        self.original_kernel=self.capture.kernel;self.capture.kernel=self.kernel;self.installed=True

    def begin(self,position):
        if not self.installed or self.position is not None or position!=self.last+1:
            raise RuntimeError('OWN_CONTIGUOUS_TOKEN0_HISTORY_REQUIRED')
        self.position=int(position);self.called=set()

    def kernel(self,**kwargs):
        if self.position is None:raise RuntimeError('KERNEL_OUTSIDE_EXPLICIT_TOKEN')
        self._guard(kwargs);layer=int(self.capture.probe.current_layer)
        if layer not in self.layers or layer in self.called:raise RuntimeError('ACTUAL_KERNEL_LAYER_COVERAGE_MISMATCH')
        initial=kwargs.get('initial_state');torch=self.torch
        if layer in self.previous:
            want=self.previous[layer]
            if initial is None or initial.dtype!=want.dtype or initial.shape!=want.shape:
                raise AssertionError('NEXT_KERNEL_CARRIER_IDENTITY_MISMATCH')
            if not torch.equal(initial.contiguous().view(torch.uint8),want.contiguous().view(torch.uint8)):
                raise AssertionError('POST_QDQ_BYTES_NOT_CONSUMED')
            self.consumed+=1
        elif initial is not None and bool(initial.count_nonzero()):
            raise AssertionError('NONZERO_DOCUMENT_START_NOT_ALLOWED')
        self.called.add(layer)
        # Real caller argument is untouched; audit clone is never a continuation.
        return self.original_kernel(**kwargs)

    def after_model_step(self,cache,rotations,position,probe=False):
        if position!=self.position or self.called!=set(self.layers):raise RuntimeError('MISSING_REAL_KERNEL_CALLS_BEFORE_WRITEBACK')
        if len(rotations)!=len(self.layers):raise ValueError('FROZEN_H_LAYER_COVERAGE_MISMATCH')
        torch=self.torch
        for index,layer in enumerate(self.layers):
            state=self.direction.cache_state(self.capture,cache,layer);self._guard(state)
            if state.dtype!=torch.float32 or state.ndim!=4 or tuple(state.shape[-2:])!=(128,128):
                raise ValueError('ORIGINAL_NATIVE_FP32_KDA_CARRIER_REQUIRED')
            self.observed_devices.add(str(state.device))
            if len(self.observed_devices)!=1:raise ValueError('ONE_ACTUAL_DEVICE_REQUIRED')
            h=rotations[index];self._guard(h)
            if h.device!=state.device or h.dtype!=torch.float32 or tuple(h.shape)!=(128,128):
                raise ValueError('FROZEN_H128_IDENTITY_OR_DEVICE_MISMATCH')
            if layer not in self.valid_h:
                if self.h_fingerprints is None or len(self.h_fingerprints)!=len(self.layers):raise ValueError('EXPLICIT_FROZEN_H_FINGERPRINTS_REQUIRED')
                if fingerprint(torch,h)!=self.h_fingerprints[index]:raise AssertionError('FROZEN_H_BYTES_MISMATCH')
                self.valid_h.add(layer)
            rotated=state @ h[None]
            verify=self.audit_mode=='CANARY_STRICT' or position in self.probes or layer not in self.parity_layers
            # Actual preQDQ host snapshot exists before encode or writeback. It is
            # evidence only and cannot be handed to the real recurrent kernel.
            host_input=rotated.detach().to(device='cpu',copy=True) if verify else None
            context=dict(native_shape=list(state.shape),native_stride=list(state.stride()),
                rotated_shape=list(rotated.shape),rotated_stride=list(rotated.stride()),
                rotated_storage_offset=int(rotated.storage_offset()),rotated_contiguous=rotated.is_contiguous(),
                dtype=str(rotated.dtype),device=str(rotated.device),
                h_fingerprint=self.h_fingerprints[index],
                native_before_writeback=fingerprint(torch,state) if verify or probe else None)
            encoded=None;reference=None
            try:
                if state.device.type=='cuda':
                    if self.cuda_backend is None or not self.authorized:raise PermissionError('AUTHORIZED_CANONICAL_CUDA_BACKEND_REQUIRED')
                    if self.evidence_writer is None or not self.evidence_writer.bounded():raise ValueError('GPU_REQUIRES_BOUNDED_DURABLE_EVIDENCE_WRITER')
                    encoded=self.cuda_backend.encode(rotated,capture_stages=verify);self.actual_cuda_encode_count+=1
                else:encoded=encode_torch(torch,rotated,capture_stages=verify)
                if verify:
                    reference=encode(host_input,capture_stages=True)
                    parity=compare_stage_bits(encoded.stages,reference.stages)
                    self.parity_layers.add(layer);self.stage_parities+=1
                    if probe or position==0 or position in self.probes:
                        self._emit('canonical_v2_actual_stage_snapshot',dict(position=position,layer=layer,
                            codec=VERSION,context=context,pre_qdq=host_input.numpy().copy(),actual_stages=encoded.stages,
                            canonical_CPU_stages=reference.stages,parity=parity,
                            scope='ACTUAL_CUDA_CODEC' if state.device.type=='cuda' else 'CPU_CALLBACK_FIXTURE'))
            except Exception as error:
                if host_input is None:
                    host_input=rotated.detach().to(device='cpu',copy=True)
                self._emit('canonical_v2_first_difference_evidence',dict(position=position,layer=layer,codec=VERSION,context=context,
                    pre_qdq=None if host_input is None else host_input.numpy().copy(),
                    pre_qdq_identity=None if host_input is None else fingerprint(torch,host_input),
                    actual_stages=None if encoded is None else encoded.stages,
                    canonical_CPU_stages=None if reference is None else reference.stages,
                    error=repr(error),writeback_performed=False,
                    actual_stage_boundary='UNKNOWN_IF_BACKEND_DID_NOT_RETURN_STAGES'))
                raise
            decoded=encoded.decode(torch)
            physical=decoded @ h[None].transpose(-1,-2)
            if not bool(torch.isfinite(physical).all()):
                self._emit('canonical_v2_first_difference_evidence',dict(position=position,layer=layer,codec=VERSION,context=context,
                    pre_qdq=None if host_input is None else host_input.numpy().copy(),actual_stages=encoded.stages,
                    canonical_CPU_stages=None if reference is None else reference.stages,
                    error='NONFINITE_POST_QDQ_NATIVE_WRITEBACK',writeback_performed=False))
                raise ArithmeticError('NONFINITE_POST_QDQ_NATIVE_WRITEBACK')
            with torch.inference_mode():state.copy_(physical)
            if not torch.equal(state.contiguous().view(torch.uint8),physical.contiguous().view(torch.uint8)):
                self._emit('canonical_v2_first_difference_evidence',dict(position=position,layer=layer,codec=VERSION,context=context,
                    pre_qdq=None if host_input is None else host_input.numpy().copy(),actual_stages=encoded.stages,
                    canonical_CPU_stages=None if reference is None else reference.stages,
                    error='WRITEBACK_BYTES_MISMATCH',writeback_performed=True))
                raise AssertionError('WRITEBACK_BYTES_MISMATCH')
            self.previous[layer]=state.detach().clone();self.writes+=1
            if probe:self._emit('canonical_v2_actual_writeback',dict(position=position,layer=layer,
                writeback=fingerprint(torch,state),payload=encoded.metadata_audit(),
                physical_packed_INT8_persistent=False,unquantized_shadow=False))
            del rotated,encoded,reference,decoded,physical,host_input
        self.peak_clone_bytes=max(self.peak_clone_bytes,sum(v.numel()*v.element_size() for v in self.previous.values()))
        self.last=position;self.position=None

    def integrity(self,endpoint):
        n=len(self.layers)
        if self.position is not None or self.last!=endpoint-1:raise AssertionError('UNFINISHED_CURSOR')
        if self.writes!=n*endpoint or self.consumed!=n*(endpoint-1):raise AssertionError('ALL_TOKEN_LAYER_WRITE_AND_CONSUMPTION_REQUIRED')
        if self.parity_layers!=set(self.layers):raise AssertionError('ALL_LAYER_CANONICAL_PARITY_REQUIRED')
        if self.audit_mode=='CANARY_STRICT' and self.stage_parities!=n*endpoint:raise AssertionError('CANARY_REQUIRES_EVERY_WRITE_STAGE_EXACT')
        actual_cuda=self.actual_cuda_encode_count==self.writes and self.writes>0 and self.authorized
        return dict(codec=VERSION,audit_mode=self.audit_mode,actual_token_count=endpoint,layer_count=n,
            every_token_layer_qdq_count=self.writes,next_kernel_byte_equal_post_qdq_count=self.consumed,
            all_consumption_bytes_verified=True,canonical_stage_parity_count=self.stage_parities,
            canonical_stage_parity_scope='ALL_WRITES' if self.audit_mode=='CANARY_STRICT' else 'FIRST_AND_FROZEN_PROBES',
            post_qdq_audit_clone_peak_bytes=self.peak_clone_bytes,audit_clone_used_as_continuation=False,
            unquantized_high_precision_shadow=False,physical_packed_INT8_persistent=False,
            observed_state_devices=sorted(self.observed_devices),actual_cuda_boundary_executed=actual_cuda,
            GPU_RUNTIME_STATUS='EXECUTED_ACTUAL_CUDA_BOUNDARY' if actual_cuda else 'NOT_RUN',
            execution_scope='ACTUAL_CUDA_BOUNDARY_NOT_OVERALL_GPU_QUALIFICATION' if actual_cuda else 'CPU_CALLBACK_FIXTURE')

    def close(self):
        if self.installed:self.capture.kernel=self.original_kernel;self.installed=False
        self.previous.clear()
