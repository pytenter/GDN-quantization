"""Explicit, SHA-pinned desktop graphics exception; never a compute exception.

The original default resource source is unchanged. This optional control policy
requires both graphics APIs, fixed process identities and bounded measured bytes.
Raw samples, device used/reserved/free, and every compute PID remain untouched.
"""
import ast
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import re

SCHEMA = 'ling_authorized_desktop_graphics_v1'
TASK = 'LING_KDA_ROTATION_DUAL_AXIS_STABLE_RECOVERY_V1'
PAIRS = {0:'GPU-283734e9-d350-ed3b-a9f2-e3ec1494e186',
         1:'GPU-936da32c-c539-627e-7f50-bd12fb7e9521'}
ALLOWANCE = 64 * 2**20


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _current_boot(): return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def _integer(value, message, minimum=0):
    assert type(value) is int and minimum <= value < 2**64-1, message
    return value


def _smi_bytes(value):
    assert isinstance(value,str), 'GRAPHICS_SMI_MEMORY_UNKNOWN'
    match=re.fullmatch(r'\s*([0-9]+)\s+MiB\s*',value)
    assert match is not None, 'GRAPHICS_SMI_MEMORY_UNKNOWN'
    return _integer(int(match.group(1))*2**20,'GRAPHICS_SMI_MEMORY_UNKNOWN')


class AuthorizedGraphicsPolicy:
    def __init__(self,path,expected_sha256,*,boot_id_provider=_current_boot):
        self.path=Path(path)
        assert self.path.is_absolute() and self.path.is_file(), 'GRAPHICS_AUTHORIZATION_PATH_REQUIRED'
        assert isinstance(expected_sha256,str) and len(expected_sha256)==64
        assert _sha(self.path)==expected_sha256, 'GRAPHICS_AUTHORIZATION_SHA_MISMATCH'
        self.sha256=expected_sha256
        self.config=json.loads(self.path.read_text())
        cfg=self.config
        assert cfg['schema']==SCHEMA and cfg['task_id']==TASK, 'GRAPHICS_AUTHORIZATION_SCOPE_MISMATCH'
        assert isinstance(cfg['authorization']['user_instruction'],str) and cfg['authorization']['user_instruction'].strip(), 'EXPLICIT_DESKTOP_AUTHORIZATION_REQUIRED'
        assert isinstance(cfg['boot_id'],str) and cfg['boot_id'], 'GRAPHICS_BOOT_PIN_REQUIRED'
        self.boot_id_provider=boot_id_provider
        assert self.boot_id_provider()==cfg['boot_id'], 'GRAPHICS_AUTHORIZATION_BOOT_CHANGED'
        devices=cfg['authorized_devices']
        assert isinstance(devices,dict) and set(devices)==set(PAIRS.values()), 'EXACT_AUTHORIZED_DEVICE_SET_REQUIRED'
        for index,uuid in PAIRS.items():
            device=devices[uuid]
            assert type(device['target_index']) is int and device['target_index']==index, 'GRAPHICS_DEVICE_INDEX_PIN_MISMATCH'
            processes=device['processes']
            assert isinstance(processes,list) and processes, 'GRAPHICS_PROCESS_BASELINE_REQUIRED'
            assert len({row['pid'] for row in processes})==len(processes), 'DUPLICATE_AUTHORIZED_GRAPHICS_PID'
            for row in processes:
                _integer(row['pid'],'GRAPHICS_PID_PIN_INVALID',1)
                _integer(row['uid'],'GRAPHICS_UID_PIN_INVALID')
                _integer(row['start_ticks'],'GRAPHICS_START_PIN_INVALID',1)
                assert isinstance(row['comm'],str) and row['comm'], 'GRAPHICS_COMM_PIN_INVALID'
                _integer(row['max_NVML_usedGpuMemory_bytes'],'GRAPHICS_NVML_MEMORY_CAP_INVALID')
                _integer(row['max_SMI_used_memory_bytes'],'GRAPHICS_SMI_MEMORY_CAP_INVALID')
            cap=_integer(device['max_attributed_graphics_bytes'],'GRAPHICS_AGGREGATE_CAP_INVALID')
            assert cap<=max(sum(row['max_NVML_usedGpuMemory_bytes'] for row in processes),sum(row['max_SMI_used_memory_bytes'] for row in processes)), 'GRAPHICS_AGGREGATE_CAP_EXCEEDS_PINNED_PROCESSES'

    def evidence(self,s,uuid):
        assert _sha(self.path)==self.sha256, 'GRAPHICS_AUTHORIZATION_CHANGED'
        assert self.boot_id_provider()==self.config['boot_id'], 'GRAPHICS_AUTHORIZATION_BOOT_CHANGED'
        assert uuid in self.config['authorized_devices'] and s.get('uuid')==uuid, 'GRAPHICS_DEVICE_UUID_CHANGED'
        for key in ('physical','compute','SMI_XML'):
            assert isinstance(s.get(key),dict) and s[key].get('returncode')==0, 'SAFETY_QUERY_UNKNOWN:'+key
        assert isinstance(s.get('direct_NVML'),dict) and s['direct_NVML'].get('status')=='PASS', 'NVML_UNAVAILABLE'
        device=self.config['authorized_devices'][uuid]
        assert len(s.get('target',[]))==1 and s['target'][0]['uuid']==uuid and int(s['target'][0]['index'])==device['target_index'], 'GRAPHICS_DEVICE_INDEX_CHANGED'
        groups=s.get('direct_NVML_processes');xml=s.get('SMI_all_processes')
        assert isinstance(groups,dict) and all(isinstance(groups.get(k),dict) and groups[k].get('status')=='PASS' and isinstance(groups[k].get('processes'),list) for k in ['compute','graphics']), 'PROCESS_VISIBILITY_UNKNOWN'
        assert isinstance(xml,dict) and xml.get('status')=='PASS' and isinstance(xml.get('processes'),list), 'PROCESS_VISIBILITY_UNKNOWN'
        nvml=groups['graphics']['processes']
        nvml_ids=[_integer(row.get('pid'),'GRAPHICS_PID_UNKNOWN',1) for row in nvml]
        assert len(nvml_ids)==len(set(nvml_ids)), 'DUPLICATE_NVML_GRAPHICS_PID'
        xml_graphics=[]
        for row in xml['processes']:
            assert row.get('type') in ('G','C','C+G','G+C'), 'SMI_PROCESS_TYPE_UNKNOWN'
            pid=row.get('pid')
            assert isinstance(pid,str) and pid.isdigit() and int(pid)>0, 'SMI_PROCESS_PID_UNKNOWN'
            if row['type']=='G':xml_graphics.append(row)
        xml_ids=[int(row['pid']) for row in xml_graphics]
        assert len(xml_ids)==len(set(xml_ids)), 'DUPLICATE_SMI_GRAPHICS_PID'
        expected={row['pid']:row for row in device['processes']}
        assert set(nvml_ids)==set(xml_ids)==set(expected), 'AUTHORIZED_GRAPHICS_API_SET_CHANGED'
        owners=s.get('process_ownership')
        assert isinstance(owners,list) and len({row.get('pid') for row in owners})==len(owners), 'GRAPHICS_OWNERSHIP_UNKNOWN_OR_DUPLICATE'
        owners={row['pid']:row for row in owners}
        graphics=[];memory=[]
        xml_by_pid={int(row['pid']):row for row in xml_graphics}
        for row in nvml:
            pid=row['pid'];pin=expected[pid];owner=owners.get(pid)
            assert isinstance(owner,dict) and owner.get('status')=='PASS', 'GRAPHICS_PROCESS_OWNERSHIP_UNKNOWN'
            assert all(owner.get(k)==pin[k] for k in ('pid','uid','start_ticks','comm')), 'AUTHORIZED_GRAPHICS_IDENTITY_CHANGED'
            nbytes=_integer(row.get('usedGpuMemory'),'GRAPHICS_NVML_MEMORY_UNKNOWN')
            sbytes=_smi_bytes(xml_by_pid[pid].get('used_memory'))
            assert nbytes<=pin['max_NVML_usedGpuMemory_bytes'] and sbytes<=pin['max_SMI_used_memory_bytes'], 'AUTHORIZED_GRAPHICS_PROCESS_MEMORY_CAP_EXCEEDED'
            graphics.append(owner)
            memory.append(dict(pid=pid,NVML_usedGpuMemory_bytes=nbytes,SMI_used_memory_bytes=sbytes))
        nsum=sum(row['NVML_usedGpuMemory_bytes'] for row in memory)
        ssum=sum(row['SMI_used_memory_bytes'] for row in memory)
        attributed=min(nsum,ssum,device['max_attributed_graphics_bytes'])
        m=s.get('direct_NVML',{}).get('memory_v2')
        assert isinstance(m,dict) and all(type(m.get(k)) is int and m[k]>=0 for k in ('used','free','reserved','total')), 'NVML_MEMORY_UNKNOWN'
        assert m['total']==m['used']+m['free']+m['reserved'], 'MEMORY_ARITHMETIC_FAILURE'
        assert attributed<=m['total'], 'GRAPHICS_ATTRIBUTION_EXCEEDS_DEVICE_MEMORY'
        residual=m['used']-attributed
        assert residual>=-ALLOWANCE, 'GRAPHICS_ATTRIBUTION_MATERIALLY_EXCEEDS_USED_UNKNOWN'
        return dict(graphics=sorted(graphics,key=lambda row:row['pid']),
                    graphics_process_evidence=sorted(memory,key=lambda row:row['pid']),
                    authorized_graphics_attributed_bytes=attributed,
                    signed_unattributed_residual_bytes=residual,
                    unattributed_allocated_bytes=max(0,residual),
                    allocated_bytes=m['used'],free_bytes=m['free'],reserved_bytes=m['reserved'],
                    authorized_graphics_baseline_path=str(self.path),
                    authorized_graphics_baseline_sha256=self.sha256,
                    attribution_scope='Actual minimum of current two-API graphics sums and pinned aggregate ceiling; reserved is never subtracted.')

    def compute_union(self,s,uuid,owner):
        # The desktop authorization is never removed from any compute list.
        groups=s.get('direct_NVML_processes',{});xml=s.get('SMI_all_processes',{})
        assert groups.get('compute',{}).get('status')=='PASS' and xml.get('status')=='PASS', 'PROCESS_VISIBILITY_UNKNOWN'
        union={_integer(row.get('pid'),'COMPUTE_PID_UNKNOWN',1) for row in groups['compute']['processes']}
        union.update(int(row['pid']) for row in xml['processes'] if row.get('type')!='G')
        query=s.get('compute')
        assert isinstance(query,dict) and query.get('returncode')==0 and isinstance(query.get('stdout'),str), 'EARLY_SMI_COMPUTE_QUERY_UNKNOWN'
        early=[]
        for fields in csv.reader(io.StringIO(query['stdout'])):
            if not fields or all(not x.strip() for x in fields):continue
            assert len(fields)==3, 'EARLY_SMI_COMPUTE_CSV_UNKNOWN'
            actual,pid,memory=(value.strip() for value in fields)
            assert actual==uuid and pid.isdigit() and int(pid)>0, 'EARLY_SMI_COMPUTE_IDENTITY_UNKNOWN'
            early.append(dict(uuid=actual,pid=int(pid),used_gpu_memory=memory))
        captured=s.get('early_SMI_compute_processes')
        assert isinstance(captured,dict) and captured.get('status')=='PASS' and captured.get('processes')==early, 'EARLY_SMI_COMPUTE_CAPTURE_UNKNOWN_OR_CONFLICT'
        union.update(row['pid'] for row in early)
        allowed=set() if owner is None else {owner['pid']}
        assert not (union-allowed), ('UNAUTHORIZED_EXTERNAL_COMPUTE_PROCESS_ALL_THREE_SOURCES',sorted(union-allowed))
        assert not (union & {row['pid'] for row in self.config['authorized_devices'][uuid]['processes']}), 'AUTHORIZED_DESKTOP_PID_IS_COMPUTE_FORBIDDEN'
        return union


def install(namespace,module,source_tree,path,expected_sha256):
    """Install only the named, explicitly authorized control exception.

    Default original functions stay byte-for-byte on disk. In the authorized
    copy, exactly one material-memory test uses actual unattributed graphics
    residual rather than total used memory. No raw field or compute list changes.
    """
    policy=AuthorizedGraphicsPolicy(path,expected_sha256)
    node=copy.deepcopy(next(node for node in source_tree.body if isinstance(node,ast.FunctionDef) and node.name=='assess'))
    matches=0
    for item in ast.walk(node):
        if isinstance(item,ast.Compare) and ast.dump(item)==ast.dump(ast.parse("m['used']>64*2**20",mode='eval').body):
            item.left=ast.Call(func=ast.Name(id='_authorized_unattributed_bytes',ctx=ast.Load()),args=[ast.Name(id='s',ctx=ast.Load()),ast.Name(id='uuid',ctx=ast.Load())],keywords=[])
            matches+=1
    assert matches==1, 'ORIGINAL_MATERIAL_MEMORY_PREDICATE_CHANGED'
    node.name='_authorized_original_assess'
    namespace['_authorized_unattributed_bytes']=lambda s,uuid:policy.evidence(s,uuid)['unattributed_allocated_bytes']
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path)+':authorized_resource_control','exec'),namespace)
    bound_assess=namespace['_authorized_original_assess']

    def graphics_identity(snapshot):return policy.evidence(snapshot,snapshot['uuid'])['graphics']

    def assess(snapshot,uuid,owner=None,baseline=None):
        details=policy.evidence(snapshot,uuid)
        policy.compute_union(snapshot,uuid,owner)
        value=bound_assess(snapshot,uuid,owner,baseline)
        value.update(details)
        return value

    def assess_window(rows,uuid):
        assert len(rows)==3, 'THREE_SAMPLES_REQUIRED'
        values=[assess(snapshot,uuid) for snapshot in rows]
        baseline=values[0]['graphics']
        assert all(value['graphics']==baseline for value in values), 'GRAPHICS_IDENTITY_NOT_STABLE'
        memory=[value['unattributed_allocated_bytes'] for value in values]
        assert max(memory)-min(memory)<=ALLOWANCE, 'UNATTRIBUTED_MEMORY_GROWTH'
        assert all(value<=ALLOWANCE for value in memory), 'UNATTRIBUTED_DEVICE_ALLOCATION'
        util=[max(value['SMI_util'],value['NVML_util']) for value in values]
        assert sum(value>2 for value in util)<2 and util[-1]<=2, 'SUSTAINED_OR_CURRENT_MATERIAL_GPU_OCCUPANCY'
        return dict(status='PASS',UUID=uuid,sample_count=3,graphics_baseline=baseline,
                    min_free_bytes=min(value['free_bytes'] for value in values),
                    measured_peak_budget_bytes=namespace['PEAK'],required_extra_headroom_bytes=namespace['HEADROOM'],
                    low_util_difference_warnings=[i for i,value in enumerate(values) if value['SMI_util']!=value['NVML_util']],
                    authorized_graphics_baseline_path=str(policy.path),authorized_graphics_baseline_sha256=policy.sha256,
                    allocation_gate='UNATTRIBUTED_ALLOCATED_LE_64MIB_AFTER_EXACT_AUTHORIZED_GRAPHICS_ATTRIBUTION',values=values)

    # Compiled originals reference this exact namespace; lifecycle imports the
    # module. Both consumers must receive the same functions.
    namespace.update(graphics_identity=graphics_identity,assess=assess,assess_window=assess_window)
    module.__dict__.update(graphics_identity=graphics_identity,assess=assess,assess_window=assess_window)
    module.authorized_graphics_policy=policy
    return policy
