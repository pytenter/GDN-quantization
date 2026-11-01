"""Production mapped-library checker: replay old actual maps, with strict negatives.

CUDA/NVML identity is a captured-record CPU mock here, never GPU qualification.
"""
import os,sys,json,copy
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
R=Path(__file__).resolve().parents[1];P=R.parents[1];sys.path.insert(0,str(R/'src'));import execution_control as C
saved=C.read(P/'execution/GPU5_CODEC_MICRO/PREFLIGHT_LIBRARY.json');assert saved['missing']==['/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.178.04']
original_bindings=C.bindings;original_device=C.device;original_text=Path.read_text;original_sha=C.sha
Path.read_text=lambda self,*a,**kw:saved['full_maps']if str(self)=='/proc/self/maps'else original_text(self,*a,**kw)
C.device=lambda *a,**kw:dict(scope='CPU_MOCK_SAVED_UUID_RECORDS_NO_CUDA_OR_NVML_CALL',saved_identity=saved['identity'])
rows=original_bindings(R);checks=[]
positive=C.library_check(R,5);assert positive['status']=='PASS'and not positive['missing']and all(x['match']for x in positive['selected']);checks.append('Actual production checker accepts exact repaired closure with70 fresh disk SHA checks')
# After the positive real disk hashes, reuse hashes for in-memory negative inputs.
cache={x['path']:x['sha256']for x in positive['selected']};C.sha=lambda p,*a,**kw:cache.get(str(Path(p).resolve()))or original_sha(p,*a,**kw)
nvml='/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.178.04'
for name,mut in [('missing NVML',lambda a:a.pop(nvml)),('wrong NVML SHA',lambda a:a[nvml].update(sha256='0'*64)),('wrong NVML bytes',lambda a:a[nvml].update(bytes=a[nvml]['bytes']+1))]:
 bad=copy.deepcopy(rows);mut(bad);C.bindings=lambda root,bad=bad:bad;v=C.library_check(R,5);assert v['status']=='FAIL';checks.append('Reject '+name)
C.bindings=lambda root:rows
extra='\n1000-2000 r-xp 00000000 08:12 1 /usr/lib/x86_64-linux-gnu/libnvidia-unlisted.so.580.178.04\n'
Path.read_text=lambda self,*a,**kw:saved['full_maps']+extra if str(self)=='/proc/self/maps'else original_text(self,*a,**kw)
v=C.library_check(R,5);assert v['status']=='FAIL'and '/usr/lib/x86_64-linux-gnu/libnvidia-unlisted.so.580.178.04'in v['missing'];checks.append('Reject unlisted NVIDIA DSO; no driver-directory permission')
Path.read_text=original_text;C.bindings=original_bindings;C.device=original_device;C.sha=original_sha
assert 'torch'not in sys.modules
print(json.dumps(dict(status='PASS_ACTUAL_NATIVE_ADMISSION_CPU_REPLAY_AND_NEGATIVES',count=len(checks),checks=checks,fresh_disk_mapped_library_SHA_count=len(positive['selected']),actual_driver_API_calls=0,CUDA_NVML_identity='MOCKED_CAPTURED_RECORDS_ONLY',GPU_qualification_claim=False)))
