"""New CPU publication recovery; old failed audit claim/staging remain immutable."""
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

TASK=Path(__file__).resolve().parents[1]
ROOT=TASK/'releases/ASSET_FIX_V2'
PRIOR=TASK/'final_review/FINAL_CPU_REVIEW_20261029'
OUT=TASK/'final_review/FINAL_CPU_REVIEW_PUBLISH_RECOVERY_20261029'
TARGET='/data/zypan/analyses/pace48_lqunroll_dp32_throughput_v1/releases/ASSET_FIX_V2_FINAL_CPU_REVIEW_PUBLISH_RECOVERY_20261029'
sys.path.insert(0,str(ROOT/'operations'))
from transport import run

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def put(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def write(p,b):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'_local/READER.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'CLAIM.json').exists(),'Recovery already claimed: no automatic repeat'
    assert 'undefined symbol: renameat2' in load(PRIOR/'FIRST_FAILURE.json')['error']
    assert load(PRIOR/'LOCAL_REVIEW.json')['status'].startswith('PASS_')
    assert load(PRIOR/'REMOTE_FRESH_REVIEW.json')['status'].startswith('PASS_')
    canary=TASK/'_local/EXTRA_PUBLICATION_SYSCALL_PREFLIGHT_20261029.json'
    assert load(canary)['status']=='PASS_X86_64_RENAMEAT2_SYSCALL316_REAL_NOREPLACE_AND_EEXIST'
    oldplan=load(PRIOR/'PLAN.json')
    for x in oldplan['input_files']:
        p=Path(x['local']);assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
    copies=[('review/'+name,PRIOR/name) for name in oldplan['outputs']]
    copies += [('evidence/EXTRA_REVIEW_FIRST_FAILURE.json',PRIOR/'FIRST_FAILURE.json'),
               ('evidence/SYSCALL_PREFLIGHT.json',canary),
               ('final_extra_publish_recovery_cpu.py',Path(__file__))]
    names=[name for name,_ in copies]+['PLAN.json','CLAIM.json','PUBLICATION_RECOVERY.md','SOURCE_RECHECK.json']
    parts=[names[i:i+22] for i in range(0,len(names),22)]
    plan=dict(scope='Distinct CPU-only immutable publication recovery after missing glibc symbol',
              original_failed_claim_sha256=sha(PRIOR/'CLAIM.json'),
              original_first_failure_sha256=sha(PRIOR/'FIRST_FAILURE.json'),
              original_independent_review_sha256=sha(PRIOR/'REMOTE_FRESH_REVIEW.json'),
              primary_manifest_sha256=sha(ROOT/'MANIFEST.sha256'),
              inputs=[dict(name=n,path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for n,p in copies],
              outputs=names,parts=parts,target=TARGET,maximum_part_files_including_metadata=24,
              maximum_part_bytes=32*1024*1024,old_staging_preserved=True,
              science_and_review_artifact_bytes_unchanged=True,
              renameat2='Actual x86_64 syscall316 with RENAME_NOREPLACE; separately passed real EEXIST canary',
              GPU_or_benchmark_or_score_restarts=0)
    verify_code=r'''
import hashlib,json,stat,os
from pathlib import Path
for x in INPUTS:
 if 'remote' not in x:continue
 p=Path(x['remote']);s=p.stat()
 assert stat.S_ISREG(s.st_mode) and s.st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],str(p)
print(json.dumps(dict(status='PASS_FRESH55_ALL_PRIOR_FIXED_INPUT_SHA_UNCHANGED_BEFORE_NEW_PUBLICATION',files=len([x for x in INPUTS if 'remote' in x]))))
'''
    verify_code='INPUTS='+repr(oldplan['input_files'])+'\n'+verify_code
    compile(verify_code,'<new-publication-source-recheck>','exec')
    if '--validate-plan-only' in sys.argv:
        print(json.dumps(dict(status='PASS_NEW_CPU_RECOVERY_PLAN',parts=[len(x) for x in parts],claim_created=False)))
        return
    put(OUT/'PLAN.json',plan)
    put(OUT/'CLAIM.json',dict(pid=os.getpid(),time=time.time(),plan_sha256=sha(OUT/'PLAN.json')))
    try:
        for name,p in copies:write(OUT/name,p.read_bytes())
        note='''# 额外审核补充包的独立CPU发布恢复

最终科学状态仍为 COMPLETE_NO_CLEAR_SPEEDUP。完整比较见 review/FINAL_COMPARISON.md。

原独立额外审核的780项输入、18组测量、两fresh CUDA资格/真实退出、V1失败与复用、源/模型17/实际regularmaps等CPU审查均已PASS。其第一次发布在55旧glibc中找不到renameat2导出符号时失败，发生于已SHA/fsync的任务自有incoming暂存阶段；没有发布最终part，没有改源、主seal或任何实验，原claim/FIRST_FAILURE和incoming完整保留。

本流程是另一个固定计划、唯一claim、独立sibling的CPU发布恢复。先实际执行x86_64 Linux renameat2 syscall316 NOREPLACE及重复目标EEXIST canary，确认原数据不覆盖；随后复制完全相同的审核报告/证据，连同该CPU失败、canary和恢复程序一并按每part最多22个payload+2个metadata、32MiB限制发布。没有重试旧claim或删除旧暂存，更没有GPU/benchmark/finisher/评分重跑。

原额外审核本地预检曾误把case.length看成总长，已有LOCAL_REVIEW注明；根据原记录其为decode length加独立prompt，修正只发生在未claim的CPU reviewer。另一次工具字符串包装解析失败同样未执行任何工具/远程语句。所有科学工件保持原字节。量化门禁不放宽，未测occupancy/SMactive等仍未测，绝对降速根因仍UNRESOLVED。

行号校正：原比较报告末尾scripts/throughput_dp32.py:115实际是prefill计时收束；长decode循环入口在:116，实际固定token forward和host计时在:118。解释和实测数据不变，原报告字节保留。
'''
        write(OUT/'PUBLICATION_RECOVERY.md',note.encode())
        put(OUT/'SOURCE_RECHECK.json',run(verify_code,timeout=None))
        receipts=[]
        for index,files in enumerate(parts,1):
            part=f'part{index:02d}';local=OUT/part;local.mkdir()
            rows=[]
            for name in files:
                b=(OUT/name).read_bytes();write(local/name,b)
                rows.append(dict(path=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
            put(local/'PACKAGE.json',dict(files=rows,part=part,source_review=plan['original_independent_review_sha256'],primary_manifest=plan['primary_manifest_sha256']))
            write(local/'MANIFEST.sha256',(''.join(x['sha256']+'  '+x['path']+'\n' for x in rows)+sha(local/'PACKAGE.json')+'  PACKAGE.json\n').encode())
            rows += [dict(path=n,bytes=(local/n).stat().st_size,sha256=sha(local/n)) for n in ['PACKAGE.json','MANIFEST.sha256']]
            assert len(rows)<=24 and sum(x['bytes'] for x in rows)<=32*1024*1024
            payload=[dict(x,data=base64.b64encode((local/x['path']).read_bytes()).decode()) for x in rows]
            code=r'''
import base64,ctypes,errno,hashlib,json,os,platform
from pathlib import Path
assert platform.machine()=='x86_64'
root=Path(TARGET);root.mkdir(parents=True,exist_ok=True)
dest=root/PART;stage=root/('.incoming_'+PART)
assert not dest.exists()
stage.mkdir(exist_ok=False)
assert len(PAYLOAD)<=24 and sum(x['bytes'] for x in PAYLOAD)<=32*1024*1024
for x in PAYLOAD:
 rel=Path(x['path']);assert not rel.is_absolute() and '..' not in rel.parts
 b=base64.b64decode(x['data']);assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256']
 p=stage/rel;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
for p in sorted([stage]+[p for p in stage.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):
 fd=os.open(str(p),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
lib=ctypes.CDLL(None,use_errno=True);fn=lib.syscall;fn.restype=ctypes.c_long
ret=fn(ctypes.c_long(316),ctypes.c_int(-100),ctypes.c_char_p(os.fsencode(stage)),ctypes.c_int(-100),ctypes.c_char_p(os.fsencode(dest)),ctypes.c_uint(1))
if ret!=0:
 err=ctypes.get_errno();raise OSError(err,os.strerror(err))
fd=os.open(str(root),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
for x in PAYLOAD:
 p=dest/x['path'];assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
print(json.dumps(dict(status='PASS_REAL_SYSCALL316_NOREPLACE_FSYNC_POST_SHA',name=PART,root=str(dest),files=len(PAYLOAD),bytes=sum(x['bytes'] for x in PAYLOAD),manifest_sha256=hashlib.sha256((dest/'MANIFEST.sha256').read_bytes()).hexdigest())))
'''
            code='TARGET='+repr(TARGET)+'\nPART='+repr(part)+'\nPAYLOAD='+repr(payload)+'\n'+code
            compile(code,'<new-independent-noreplace-syscall-publisher>','exec')
            receipt=run(code,timeout=None)
            put(OUT/(part+'_RECEIPT.json'),receipt);receipts.append(receipt)
        # A separate read-only remote interpreter checks every final manifest again.
        code=r'''
import hashlib,json,stat
from pathlib import Path
for row in RECEIPTS:
 r=Path(row['root']);assert hashlib.sha256((r/'MANIFEST.sha256').read_bytes()).hexdigest()==row['manifest_sha256']
 for line in (r/'MANIFEST.sha256').read_text().splitlines():
  h,n=line.split('  ',1);p=r/n;assert stat.S_ISREG(p.stat().st_mode) and hashlib.sha256(p.read_bytes()).hexdigest()==h,str(p)
print(json.dumps(dict(status='PASS_INDEPENDENT55_POST_PUBLICATION_ALL_PARTS_SHA',parts=RECEIPTS)))
'''
        code='RECEIPTS='+repr(receipts)+'\n'+code
        compile(code,'<fresh-final-read-only-verifier>','exec')
        put(OUT/'INDEPENDENT_POST_PUBLICATION_REVIEW.json',run(code,timeout=None))
        for part in receipts:
            for line in (OUT/part['name']/'MANIFEST.sha256').read_text().splitlines():
                h,n=line.split('  ',1);assert sha(OUT/part['name']/n)==h
        for x in oldplan['input_files']:
            p=Path(x['local']);assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
        assert sha(ROOT/'MANIFEST.sha256')==plan['primary_manifest_sha256']
        assert sha(PRIOR/'CLAIM.json')==plan['original_failed_claim_sha256']
        assert sha(PRIOR/'FIRST_FAILURE.json')==plan['original_first_failure_sha256']
        result=dict(status='PASS_ALL_DP32_FINAL_EXTRA_REVIEW_AND_NEW_CPU_PUBLICATION_RECOVERY_55_MAC',
                    scientific_status='COMPLETE_NO_CLEAR_SPEEDUP',extra_review_pending=False,
                    independent_review_sha256=plan['original_independent_review_sha256'],
                    primary_manifest_sha256=plan['primary_manifest_sha256'],parts=receipts,
                    original_failed_claim_and_staging_preserved=True,all_original_fixed_input_local_SHA_rechecked=True,
                    new_GPU_AIME_accuracy_or_finisher_runs=0)
        put(OUT/'RESULT.json',result)
        print(json.dumps(result),flush=True)
    except BaseException as e:
        put(OUT/'FIRST_FAILURE.json',dict(error=repr(e),no_automatic_repeat=True))
        raise
if __name__=='__main__':main()
