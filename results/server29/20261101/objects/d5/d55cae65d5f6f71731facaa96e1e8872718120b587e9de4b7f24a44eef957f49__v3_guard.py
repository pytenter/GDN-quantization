"""UUID-scoped NVML compute ownership; raw fuser is diagnostic, never an allowlist."""
import json
import os
import subprocess
import time
from pathlib import Path


def proc_identity(pid):
    p=Path('/proc')/str(pid)
    stat=(p/'stat').read_text().rsplit(') ',1)[1].split()
    status=(p/'status').read_text().splitlines()
    return dict(pid=int(pid),start_ticks=int(stat[19]),uid=int(next(x for x in status if x.startswith('Uid:')).split()[1]),
        nspid=[int(x) for x in next(x for x in status if x.startswith('NSpid:')).split()[1:]],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def query(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=20)
    return dict(command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)


def evaluate(gpu,uuid,physical,compute,diagnostics,binding,current,require_context):
    errors=[]; classes=[]
    if physical['returncode'] or compute['returncode']: errors.append('NVML_QUERY_FAILED')
    for d in diagnostics:
        if d['returncode'] not in (0,1): errors.append('DIAGNOSTIC_QUERY_FAILED_NOT_SAFE')
    fields=[x.strip() for x in physical['stdout'].strip().split(',')]
    if len(fields)!=8 or fields[0]!=str(gpu) or fields[1]!=uuid: errors.append('PHYSICAL_UUID_INDEX_MISMATCH')
    target=[]
    for row in compute['stdout'].splitlines():
        if not row.strip(): continue
        parts=[x.strip() for x in row.split(',')]
        if len(parts)!=3 or not parts[1].isdigit(): errors.append('UNPARSEABLE_COMPUTE_PID'); continue
        u,pid,memory=parts; pid=int(pid)
        if u!=uuid: classes.append(dict(pid=pid,gpu_uuid=u,classification='OTHER_UUID_NOT_TARGET_COMPUTE')); continue
        target.append(pid)
        own=binding is not None and current==binding and pid==binding['pid']
        classes.append(dict(pid=pid,gpu_uuid=u,used_gpu_memory_mib=memory,
            classification='VERIFIED_THIS_GPU_WORKER' if own else 'UNAUTHORIZED_TARGET_COMPUTE'))
        if not own: errors.append('UNAUTHORIZED_TARGET_COMPUTE_PID')
    if binding is not None and current!=binding: errors.append('PID_STARTTIME_OR_NAMESPACE_IDENTITY_CHANGED')
    if require_context and (binding is None or binding['pid'] not in target): errors.append('OWN_TARGET_CONTEXT_NOT_OBSERVED')
    if binding is None and len(fields)==8:
        try:
            if float(fields[6])>64 or float(fields[7])!=0: errors.append('UNEXPLAINED_TARGET_MEMORY_OR_UTILIZATION')
        except ValueError: errors.append('UNPARSEABLE_PHYSICAL_OCCUPANCY')
    for d in diagnostics:
        for raw in d['stdout'].split():
            if not raw.isdigit(): errors.append('UNPARSEABLE_FUSER_PID'); continue
            pid=int(raw)
            classes.append(dict(pid=pid,device=d['command'][-1],classification=
                'THIS_WORKER_FD_DIAGNOSTIC' if binding is not None and pid==binding['pid'] else 'NON_SELF_FD_DIAGNOSTIC_NOT_COMPUTE_AUTHORIZATION'))
    return dict(exclusive=not errors,failure_conditions=sorted(set(errors)),target_compute_pids=target,pid_classifications=classes)


def device(gpu,cfg,pid=None,*,binding=None,evidence=None,require_context=False):
    uuid=cfg['gpu_uuids'][str(gpu)]; result=dict(epoch=time.time(),gpu=gpu,gpu_uuid=uuid,
        backend='nvidia-smi NVML compute-process query with exact physical UUID filter',
        scope='only verified current worker target context; no peer/all-GPU allowlist',binding=binding)
    try:
        physical=query(['nvidia-smi','-i',uuid,'--query-gpu=index,uuid,name,driver_version,pci.bus_id,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
        compute=query(['nvidia-smi','-i',uuid,'--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'])
        diagnostics=[query(['fuser',d]) for d in (f'/dev/nvidia{gpu}','/dev/nvidiactl','/dev/nvidia-uvm')]
        current=None if pid is None else proc_identity(pid)
        result.update(physical=physical,compute=compute,fuser=diagnostics,current_identity=current)
        result.update(evaluate(gpu,uuid,physical,compute,diagnostics,binding,current,require_context))
        if physical['returncode']==0:
            fields=[x.strip() for x in physical['stdout'].strip().split(',')]
            if len(fields)==8 and fields[3]!=cfg['runtime']['driver']:
                result['exclusive']=False; result['failure_conditions'].append('DRIVER_IDENTITY_MISMATCH')
    except Exception as exc:
        result.update(exclusive=False,failure_conditions=['QUERY_OR_IDENTITY_EXCEPTION'],query_exception=repr(exc))
    # Write before caller's assertion, including every raw query and classification.
    if evidence is not None:
        with Path(evidence).open('a',encoding='utf-8') as f:
            f.write(json.dumps(result,sort_keys=True,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
    return result
