"""Real CPU imports/mapped-DSO SHA checks; mocked device observation, no CUDA/NVML API calls."""
import os,sys,json,copy,ctypes
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'));sys.path.insert(0,str(R/'runtime/transformers-qwen35/src'))
import execution_control as C
import torch
assert not torch.cuda.is_initialized()
torch.cuda._lazy_init=lambda *a,**kw:(_ for _ in()).throw(RuntimeError('CPU proof forbids CUDA init'))
import transformers
assert Path(transformers.__file__).resolve()==(R/'runtime/transformers-qwen35/src/transformers/__init__.py').resolve()
rows=C.bindings(R);nvml=[p for p in rows if Path(p).name.startswith('libnvidia-ml.so.')];assert len(nvml)==1
handle=ctypes.CDLL(nvml[0]) # Load exact inventory DSO only, invoke no NVML function.
original_device=C.device;C.device=lambda *a,**kw:dict(scope='EXPLICIT_CPU_DEVICE_MOCK_NO_GPU_API')
original_bindings=C.bindings;original_text=Path.read_text;original_sha=C.sha;checks=[]
positive=C.library_check(R,C.NODE['physical_gpu']);assert positive['status']=='PASS'and not positive['missing']and all(x['match']for x in positive['selected']);checks.append('Current host actual CPU mapped closure fresh SHA/bytes; no historical maps replay')
assert nvml[0]in{x['path']for x in positive['selected']}
cache={x['path']:x['sha256']for x in positive['selected']};C.sha=lambda p,*a,**kw:cache.get(str(Path(p).resolve()))or original_sha(p,*a,**kw)
for name,mut in [('missing NVML',lambda a:a.pop(nvml[0])),('wrong NVML SHA',lambda a:a[nvml[0]].update(sha256='0'*64)),('wrong NVML bytes',lambda a:a[nvml[0]].update(bytes=a[nvml[0]]['bytes']+1))]:
 bad=copy.deepcopy(rows);mut(bad);C.bindings=lambda root,bad=bad:bad;v=C.library_check(R,C.NODE['physical_gpu']);assert v['status']=='FAIL';checks.append('Reject '+name)
C.bindings=lambda root:rows
maps=original_text(Path('/proc/self/maps'));extra='\n1000-2000 r-xp 00000000 08:12 1 /usr/lib/x86_64-linux-gnu/libnvidia-unlisted.so.0\n'
Path.read_text=lambda self,*a,**kw:maps+extra if str(self)=='/proc/self/maps'else original_text(self,*a,**kw)
v=C.library_check(R,C.NODE['physical_gpu']);assert v['status']=='FAIL'and '/usr/lib/x86_64-linux-gnu/libnvidia-unlisted.so.0'in v['missing'];checks.append('Reject unknown NVIDIA DSO without directory exemption')
Path.read_text=original_text;C.bindings=original_bindings;C.device=original_device;C.sha=original_sha
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_CURRENT_HOST_NATIVE_CPU_MAPS_STRICT_NEGATIVES',node_id=C.NODE['id'],count=len(checks),checks=checks,fresh_disk_mapped_library_SHA_count=len(positive['selected']),actual_driver_API_calls=0,CUDA_NVML_identity='EXPLICIT_CPU_MOCK_ONLY',GPU_qualification_claim=False)))
