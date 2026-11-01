"""Bounded-memory CPU parser for category-correct Kineto Chrome activities.

No Torch imports. GPU user annotations are never classified as kernels or active
intervals. Each trace is parsed independently and its complete content SHA checked.
"""
from __future__ import annotations
import codecs
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics


class TraceEventStream:
    def __init__(self,path,expected_sha256,expected_bytes):
        self.path=Path(path);self.expected_sha256=expected_sha256;self.expected_bytes=expected_bytes;self.proof=None

    def __iter__(self):
        decoder=json.JSONDecoder();utf8=codecs.getincrementaldecoder('utf-8')();digest=hashlib.sha256()
        with self.path.open('rb') as handle:
            before=os.fstat(handle.fileno())
            if before.st_size!=self.expected_bytes:raise ValueError('Chrome trace bytes differ from recorded artifact')
            buffer='';pos=0;eof=False;read_bytes=0
            def feed():
                nonlocal buffer,eof,read_bytes
                data=handle.read(1024**2)
                if data:
                    digest.update(data);read_bytes+=len(data);buffer+=utf8.decode(data)
                else:
                    eof=True;buffer+=utf8.decode(b'',final=True)
            marker=re.compile(r'"traceEvents"\s*:\s*\[')
            found=None
            while found is None:
                if eof:raise ValueError('Chrome traceEvents array missing')
                feed();found=marker.search(buffer)
                if found is None and len(buffer)>16*1024**2:raise ValueError('Unbounded Chrome trace header')
            pos=found.end()
            while True:
                while True:
                    while pos<len(buffer) and (buffer[pos].isspace() or buffer[pos]==','):pos+=1
                    if pos<len(buffer):break
                    if eof:raise ValueError('Truncated traceEvents array')
                    buffer='';pos=0;feed()
                if buffer[pos]==']':
                    break
                if pos>1024**2:
                    buffer=buffer[pos:];pos=0
                while True:
                    try:
                        event,end=decoder.raw_decode(buffer,pos);pos=end;break
                    except json.JSONDecodeError:
                        if eof:raise
                        if len(buffer)-pos>16*1024**2:raise ValueError('Unbounded/malformed Chrome event')
                        if pos:
                            buffer=buffer[pos:];pos=0
                        feed()
                if not isinstance(event,dict):raise ValueError('Chrome trace event must be an object')
                yield event
            # Remaining bytes may contain metadata. Hash them too without retaining
            # the entire footer or making a second multi-gigabyte file copy.
            for data in iter(lambda:handle.read(4*1024**2),b''):
                digest.update(data);read_bytes+=len(data)
            after=os.fstat(handle.fileno())
            if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):raise ValueError('Chrome trace changed during CPU analysis')
            if read_bytes!=self.expected_bytes or digest.hexdigest()!=self.expected_sha256:raise ValueError('Complete Chrome trace SHA verification failed')
            self.proof=dict(path=str(self.path),bytes=read_bytes,sha256=digest.hexdigest(),stable_during_parse=True,parser='Stdlib incremental JSON event decoder, bounded text buffer, one trace at a time')


def distribution(values):
    x=sorted(values)
    if not x:raise ValueError('No actual duration observations')
    return dict(count=len(x),min=x[0],max=x[-1],mean=math.fsum(x)/len(x),p50=statistics.median(x),p95=x[max(0,math.ceil(.95*len(x))-1)])


def union_us(intervals):
    if not intervals:return 0.
    intervals.sort();total=0.;last_end=None
    for start,end in intervals:
        total+=(end-start) if last_end is None or start>last_end else max(0.,end-last_end)
        last_end=end if last_end is None else max(last_end,end)
    return total


def corrected_profile(path,expected_sha256,expected_bytes,owner_scope_names=()):
    stream=TraceEventStream(path,expected_sha256,expected_bytes)
    categories=collections.Counter();excluded_device_annotations=collections.Counter();grouped=collections.defaultdict(list)
    intervals=[];kernel_intervals=[];kernels=0;memory=0;devices=set();runtime=collections.defaultdict(list)
    runtime_correlations=set();kernel_correlations=collections.Counter();owners=collections.defaultdict(list)
    for event in stream:
        cat=event.get('cat','');categories[cat]+=1
        name=event.get('name','')
        if cat=='gpu_user_annotation':
            excluded_device_annotations[name]+=1
        if event.get('ph')!='X':continue
        duration=event.get('dur');start=event.get('ts')
        if not isinstance(duration,(int,float)) or not isinstance(start,(int,float)) or not math.isfinite(duration) or not math.isfinite(start) or duration<0:continue
        args=event.get('args') or {}
        if cat=='cuda_runtime':
            runtime[name].append(duration)
            correlation=args.get('correlation')
            if correlation is not None:runtime_correlations.add(str(correlation))
        if cat=='user_annotation' and name in owner_scope_names:
            owners[name].append(duration)
        if cat not in {'kernel','gpu_memcpy','gpu_memset'}:
            continue
        interval=(start,start+duration);intervals.append(interval);devices.add(str(event.get('pid')))
        kind='KERNEL_ACTIVITY' if cat=='kernel' else 'MEMORY_ACTIVITY'
        grouped[(cat,name,kind)].append(duration)
        if cat=='kernel':
            kernels+=1;kernel_intervals.append(interval)
            correlation=args.get('correlation')
            if correlation is not None:kernel_correlations[str(correlation)]+=1
        else:memory+=1
    if stream.proof is None:raise ValueError('Incomplete trace read')
    if not kernels:raise ValueError('No cat==kernel activities in actual Chrome trace')
    if len(devices)!=1:raise ValueError('Unexpected multiple CUDA device pids in declared oneGPU trace')
    activity_costs=[dict(chrome_category=cat,name=name,classification=kind,count=len(ds),summed_duration_us=math.fsum(ds),duration_summary_us=distribution(ds)) for (cat,name,kind),ds in sorted(grouped.items())]
    runtime_costs=[dict(name=name,count=len(ds),summed_cpu_duration_us=math.fsum(ds),duration_summary_us=distribution(ds)) for name,ds in sorted(runtime.items())]
    matched=sum(count for corr,count in kernel_correlations.items() if corr in runtime_correlations)
    kernel_with_corr=sum(kernel_correlations.values())
    return dict(status='CHROME_CATEGORY_CORRECTED',trace_identity=stream.proof,
        category_counts=dict(categories),classification='Only exact cat==kernel is a CUDAkernel; gpu_memcpy/gpu_memset are separate memory activities; gpu_user_annotation never contributes toactiveinterval orkernelcount.',
        CUDA_device_activity_count=kernels+memory,CUDA_kernel_activity_count_excluding_memcpy_memset=kernels,
        CUDA_memory_activity_count=memory,CUDA_device_pids=sorted(devices),
        CUDA_device_activity_costs_by_name=activity_costs,CUDA_active_interval_union_us=union_us(intervals),
        CUDA_kernel_only_interval_union_us=union_us(kernel_intervals),
        CUDA_activity_duration_sum_us=math.fsum(a['summed_duration_us'] for a in activity_costs),
        CUDA_kernel_duration_sum_us=math.fsum(a['summed_duration_us'] for a in activity_costs if a['classification']=='KERNEL_ACTIVITY'),
        excluded_GPU_user_annotations=dict(excluded_device_annotations),
        actual_cuda_runtime_API_events=runtime_costs,
        correlation_audit=dict(kernel_activities_with_correlation=kernel_with_corr,kernel_activities_correlated_to_runtime=matched,kernel_activities_without_matching_runtime=kernels-matched,
            scope='Chrome args.correlation checked across GPUkernel and CPUcuda_runtime events. Not assumedone-to-one because graphs canlaunchmultiplenodes. Missingmatches remainreported.'),
        CPU_owner_scopes_from_chrome={name:dict(count=len(ds),cpu_total_us=math.fsum(ds)) for name,ds in owners.items()},
        limitations='Union merges only truecaptureddeviceactivity intervals acrossstreams onthisoneGPU; excludesGPUannotations. Activitysum mayoverlap andisnotwall. HostwallminusunionisnotpureCPUdispatch. No LS/endtoendfraction inferred fromkernel-names alone.')
