"""Read-only nvidia-smi plus direct libNVML telemetry; no CUDA context."""
import ctypes as C
import json
import os
import pathlib
import subprocess
import time
import xml.etree.ElementTree as ET

def query(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return dict(command=args, returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
    except Exception as e:
        return dict(command=args, returncode=None, stdout=None, stderr=repr(e))

class Memory(C.Structure):
    _fields_ = [('total', C.c_ulonglong), ('free', C.c_ulonglong), ('used', C.c_ulonglong)]

class MemoryV2(C.Structure):
    _fields_ = [('version', C.c_uint), ('total', C.c_ulonglong), ('reserved', C.c_ulonglong), ('free', C.c_ulonglong), ('used', C.c_ulonglong)]

class Util(C.Structure):
    _fields_ = [('gpu', C.c_uint), ('memory', C.c_uint)]

def nvml(uuid):
    try:
        lib = C.CDLL('libnvidia-ml.so.1')
        assert lib.nvmlInit_v2() == 0
        h = C.c_void_p()
        assert lib.nvmlDeviceGetHandleByUUID(uuid.encode(), C.byref(h)) == 0
        out = {}
        mem, util = Memory(), Util()
        for key, fun, value in [('memory', 'nvmlDeviceGetMemoryInfo', mem), ('utilization', 'nvmlDeviceGetUtilizationRates', util)]:
            rc = getattr(lib, fun)(h, C.byref(value))
            out[key] = {n: getattr(value, n) for n, _ in value._fields_} if rc == 0 else dict(status='UNKNOWN', returncode=rc)
        for key, fun, extra in [('temperature', 'nvmlDeviceGetTemperature', [0]), ('power_mW', 'nvmlDeviceGetPowerUsage', []), ('pstate', 'nvmlDeviceGetPerformanceState', []), ('SM_clock_MHz', 'nvmlDeviceGetClockInfo', [1])]:
            value = C.c_uint()
            rc = getattr(lib, fun)(h, *extra, C.byref(value))
            out[key] = value.value if rc == 0 else dict(status='UNKNOWN', returncode=rc)
        m2 = MemoryV2(); m2.version = C.sizeof(MemoryV2) | (2 << 24)
        rc = lib.nvmlDeviceGetMemoryInfo_v2(h, C.byref(m2))
        out['memory_v2'] = {n: getattr(m2,n) for n,_ in m2._fields_} if rc == 0 else dict(status='UNKNOWN',returncode=rc)
        lib.nvmlShutdown()
        return dict(status='PASS', **out)
    except Exception as e:
        return dict(status='DRIVER_OR_NVML_ERROR', error=repr(e))

def inventory():
    r = query(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,utilization.memory,pstate,clocks.current.sm,power.draw,temperature.gpu,driver_version,pci.bus_id', '--format=csv,noheader,nounits'])
    if r['returncode'] != 0:
        return [], r
    rows = []
    for line in r['stdout'].splitlines():
        fields = [x.strip() for x in line.split(',')]
        if len(fields) != 13:
            return [], dict(r, parse_error=True)
        rows.append(dict(zip(['index','uuid','name','total_MiB','used_MiB','GPU_util','memory_util','pstate','SM_clock','power_W','temperature','driver','pci'], fields)))
    return rows, r

def sample(uuid):
    rows, physical = inventory()
    target = [r for r in rows if r['uuid'] == uuid]
    compute = query(['nvidia-smi','-i',uuid,'--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'])
    index = target[0]['index'] if len(target) == 1 else None
    fuser = query(['fuser', '/dev/nvidia' + index]) if index is not None else dict(returncode=None, stderr='UUID mapping UNKNOWN')
    return dict(epoch=time.time(), uuid=uuid, physical=physical, target=target, compute=compute, direct_NVML=nvml(uuid), fuser=fuser)

def observe(uuids, path, seconds=60, interval=5):
    records=[]; start=time.monotonic()
    with pathlib.Path(path).open('x') as f:
        for k in range(int(seconds/interval)+1):
            if k: time.sleep(max(0, start+k*interval-time.monotonic()))
            for uuid in uuids:
                row=sample(uuid); records.append(row); f.write(json.dumps(row,sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())
    return records

class Process(C.Structure):
    _fields_=[('pid',C.c_uint),('usedGpuMemory',C.c_ulonglong),('gpuInstanceId',C.c_uint),('computeInstanceId',C.c_uint)]

def nvml_processes(uuid):
    lib=C.CDLL('libnvidia-ml.so.1'); assert lib.nvmlInit_v2()==0
    try:
        h=C.c_void_p();assert lib.nvmlDeviceGetHandleByUUID(uuid.encode(),C.byref(h))==0
        out={}
        for key,fn in [('compute','nvmlDeviceGetComputeRunningProcesses_v3'),('graphics','nvmlDeviceGetGraphicsRunningProcesses_v3')]:
            count=C.c_uint(0); rc=getattr(lib,fn)(h,C.byref(count),None)
            if rc==0 and count.value==0: out[key]=dict(status='PASS',processes=[],api=fn); continue
            if rc!=7: out[key]=dict(status='UNKNOWN',returncode=rc,api=fn);continue
            values=(Process*(count.value+8))();count.value=len(values)
            rc=getattr(lib,fn)(h,C.byref(count),values)
            out[key]=dict(status='PASS',api=fn,processes=[{n:getattr(p,n) for n,_ in Process._fields_} for p in values[:count.value]]) if rc==0 else dict(status='UNKNOWN',returncode=rc,api=fn)
        return out
    finally:lib.nvmlShutdown()

def process_metadata(pid):
    try:
        d=pathlib.Path('/proc')/str(pid);st=d.stat();text=(d/'stat').read_text();fields=text[text.rfind(')')+2:].split()
        return dict(status='PASS',pid=pid,uid=st.st_uid,start_ticks=int(fields[19]),comm=(d/'comm').read_text().strip())
    except Exception as e:return dict(status='UNKNOWN',pid=pid,error=repr(e))

_original_sample=sample

def sample(uuid):
    s=_original_sample(uuid);s['direct_NVML_processes']=nvml_processes(uuid)
    x=query(['nvidia-smi','-i',uuid,'-q','-x']);s['SMI_XML']=x
    try:
        assert x['returncode']==0
        tree=ET.fromstring(x['stdout']);gpu=tree.find('gpu');assert gpu.findtext('uuid')==uuid
        ps=[dict(pid=p.findtext('pid'),type=p.findtext('type'),name=p.findtext('process_name'),used_memory=p.findtext('used_memory')) for p in gpu.findall('processes/process_info')]
        s['SMI_all_processes']=dict(status='PASS',processes=ps)
    except Exception as e:s['SMI_all_processes']=dict(status='UNKNOWN',error=repr(e))
    pids={int(p['pid']) for p in s['SMI_all_processes'].get('processes',[])}
    for group in s['direct_NVML_processes'].values():pids.update(p['pid'] for p in group.get('processes',[]))
    s['process_ownership']=[process_metadata(p) for p in sorted(pids)]
    return s


if __name__ == '__main__':
    import sys
    d=pathlib.Path(sys.argv[1]); rows, _=inventory()
    def safe(fn):
        try: return fn()
        except Exception as e: return dict(status='UNKNOWN',error=repr(e))
    env=dict(uid=os.getuid(), self_pid=os.getpid(), self_pid_namespace=os.readlink('/proc/self/ns/pid'), init_pid_namespace=safe(lambda:os.readlink('/proc/1/ns/pid')), self_status=pathlib.Path('/proc/self/status').read_text(), init_cgroup=safe(lambda:pathlib.Path('/proc/1/cgroup').read_text()), driver_logs=[query(['dmesg','--level=err,warn']),query(['journalctl','-k','-n','200','--no-pager'])])
    (d/'GPU_HOST_DIAGNOSTICS.json').write_text(json.dumps(env,indent=2))
    records=observe([r['uuid'] for r in rows],d/'GPU_DIAGNOSTIC_WINDOW.jsonl')
    print(json.dumps(dict(status='READ_ONLY_WINDOW_COMPLETE', samples=len(records), devices=len(rows), duration=60)))
