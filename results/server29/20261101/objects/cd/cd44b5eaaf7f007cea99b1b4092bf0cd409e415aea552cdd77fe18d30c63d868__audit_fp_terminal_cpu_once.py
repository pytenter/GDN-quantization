"""Independent stdlib/NumPy CPU audit; no source imports, model or GPU APIs.

Read-only nvidia-smi observes ownership. New external audit/exit receipts are
exclusive; original frozen parent failure, source and science never change.
"""
import hashlib,json,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
R=Path('/tmp/QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1')
L=tuple(i for i in range(32) if i%4!=3)
UUID='GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3'

def need(ok,msg):
    if not ok:raise RuntimeError(msg)
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def identity(pid):
    p=Path('/proc')/str(pid)
    try:
        raw=(p/'stat').read_text().rsplit(')',1)[1].split();argv=(p/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        return dict(pid=pid,starttick=int(raw[19]),cmdline=[x.decode() for x in argv])
    except (OSError,ValueError):return None
def observe_nvml():
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    need(p.returncode==0,'read-only NVML observation failed')
    return [dict(pid=int(f[0]),uuid=f[1],memory_MiB=f[2]) for f in [[v.strip() for v in line.split(',')] for line in p.stdout.splitlines() if line.strip()]]
def new(p,value):
    p=Path(p);need(not p.exists(),'external receipt already exists:'+str(p));p.parent.mkdir(parents=True,exist_ok=True)
    b=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()+b'\n';t=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    os.link(t,p);t.unlink()
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return hashlib.sha256(b).hexdigest()


def main():
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','independent observer requires empty CUDA visibility')
    need(not (R/'calibration/FP_COLLECTION_EXIT_RELEASE.json').exists(),'exit receipt already exists; never duplicate')
    claim=R/'calibration/INDEPENDENT_FP_TERMINAL_AUDIT_CLAIM.json'
    new(claim,dict(observer=identity(os.getpid()),scope='single independent CPU audit of existing completedFPpayload; noGPUretry'))
    try:
        before=time.monotonic();phase=read(R/'CALIBRATION_SOURCE_FREEZE.json');source_rows=[]
        for row in phase['files']:
            rel=Path(row['path']);p=R/rel
            need(not rel.is_absolute() and '..' not in rel.parts and not p.is_symlink() and p.is_file(),'unsafe phaseA source')
            need(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'phaseA source SHA/bytes differs:'+str(rel));source_rows.append(row)
        phase_sha=sha(R/'CALIBRATION_SOURCE_FREEZE.json')
        failure=read(R/'calibration/PARENT_FAILURE.json');fail_sha=sha(R/'calibration/PARENT_FAILURE.json')
        start=read(R/'calibration/execution/STARTED.json');dispatch=read(R/'calibration/PARENT_DISPATCH.json');parent=read(R/'calibration/PARENT_STARTED.json')
        manifest=read(R/'calibration/FP_COLLECTION_MANIFEST.json');manifest_sha=sha(R/'calibration/FP_COLLECTION_MANIFEST.json');permit=read(R/'CALIBRATION_PERMIT.json');ident=manifest['identity']
        need(ident==start['identity']==dispatch['identity']==failure['identity'],'worker exact identities disagree')
        need(parent['identity']==dispatch['parent_identity'] and identity(parent['identity']['pid'])!=parent['identity'],'original exact CPUparent still live')
        need(identity(ident['pid'])!=ident,'original exact worker still live')
        need(failure['error']=="RuntimeError('Owned collector identity changed while Popen active')",'different firstcause requires another independent scope')
        cleanup=failure['cleanup'];need(cleanup['actual_exit']==0 and cleanup['release']=='PASS' and cleanup['observations'],'original actual Popen wait0/release missing')
        need(all(not r['exact_identity_alive'] and not r['owned_NVML'] for r in cleanup['observations']),'original actual release observations disagree')
        need(manifest['status']=='PASS_ACTUAL_BOUNDED_NONAIME_FP_COLLECTION' and manifest['actual_FP_tokens']==1280 and manifest['H128_prepared_calls']==30720,'actual FP bounded coverage differs')
        need(set(map(int,manifest['actual_helper_calls']))==set(L) and set(manifest['actual_helper_calls'].values())=={1280},'actual24layer helper coverage differs')
        need(manifest['source_freeze_sha256']==phase_sha==permit['source_freeze_sha256']==parent['source_freeze_sha256'],'phaseA source identity differs')
        need(manifest['permit_sha256']==sha(R/'CALIBRATION_PERMIT.json')==start['permit_sha256'],'permit SHA differs')
        need(manifest['candidate_plan_sha256']==sha(R/'CALIBRATION_CANDIDATE_PLAN.json'),'candidate plan differs')
        need(manifest['AIME_used'] is False and manifest['quantized'] is False and manifest['new_AIME_samples']==0,'collection scope differs')
        need(manifest['gpu']==1 and manifest['gpu_uuid']==UUID,'scientific worker physical GPU differs')
        driver=read(R/'calibration/execution/DRIVER_IDENTITY.json');need(driver['status']=='PASS' and driver['actual_uuid']==UUID and driver['CUDA_VISIBLE_DEVICES']==UUID,'actual driver UUID differs')
        native_rows=[]
        for filename in ['PREFLIGHT_LIBRARY.json','POSTFLIGHT_LIBRARY.json']:
            native=read(R/'calibration/execution'/filename)
            need(native['status']=='PASS' and not native['missing'] and native['selected'],'actual native pre/post closure failed')
            own=[v for v in native['identity']['apps'] if v['pid']==ident['pid']]
            need(own and all(v['uuid']==UUID for v in own),'actual mapped pre/post own UUID evidence differs')
            for row in native['selected']:
                p=Path(row['path']);need(row['match'] is True and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'current native library bytes differ')
            native_rows.append(dict(file=filename,sha256=sha(R/'calibration/execution'/filename),selected_files=len(native['selected']),actual_own_UUID='PASS'))
        models=read(R/'analysis/MODEL_TOKENIZER_FRESH_SHA.json')['files'];need(len(models)==17,'exact17model/tokenizer files missing');model_rows=[]
        for row in models:
            p=Path(row['path']);s=p.stat();need((s.st_size,s.st_ino,s.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'model/tokenizer stat differs')
            h=sha(p);need(h==row['sha256'],'current model/tokenizer SHA differs')
            model_rows.append(dict(path=str(p),bytes=s.st_size,sha256=h))
        plan=read(R/'CALIBRATION_CANDIDATE_PLAN.json');docs={d['document_id']:d for d in plan['new_collection_proposal']['documents']}
        expectedA={(d,t,l) for d in docs for t in [31,95,159,223] for l in L};expectedD={(d,t,l) for d in docs for t in list(range(32,40))+list(range(96,104))+list(range(160,168))+list(range(224,232)) for l in L}
        gotA=set();gotD=set();kinds={};rows=[];raw=Path(manifest['raw_root']);need(str(raw)=='/dev/shm/qwen_sym8_g32_wscale_rowbal_calibration_v1' and raw.is_dir() and not raw.is_symlink(),'raw root differs')
        fingerprints={(v['document_id'],v['token']):v['state_sha256'] for v in manifest['anchor_state_fingerprints']}
        gamma_exact=0;paired_qk_checks=0
        for row in manifest['files']:
            rel=Path(row['relative']);p=raw/rel;kind=row['kind'];kinds[kind]=kinds.get(kind,0)+1
            need(not rel.is_absolute() and '..' not in rel.parts and str(p)==row['path'] and p.is_file() and not p.is_symlink(),'unsafe raw scientific path')
            need(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'actual scientific raw SHA/bytes differs')
            rows.append(dict(path=str(p),bytes=row['bytes'],sha256=row['sha256'],kind=kind))
            if kind in ('FP_STATE_AND_NEXT_TRANSPORT','ACTUAL_FP_GDN_DRIVE'):
                key=(row['document_id'],row['token'],row['layer']);need(row['document_id'] in docs and row['role']==docs[row['document_id']]['role'],'document/split identity differs')
                with np.load(p,allow_pickle=False) as z:
                    values={k:z[k] for k in z.files};need(all(v.dtype==np.float32 and np.isfinite(v).all() for v in values.values()),'actual FPraw dtype/finite differs')
                    if kind=='FP_STATE_AND_NEXT_TRANSPORT':
                        need(key in expectedA and key not in gotA and row['next_token']==row['token']+1,'anchor roster/alignment differs');gotA.add(key)
                        need(values['state_h'].shape==(32,128,128) and values['transported_next'].shape==(32,128),'actualState/transport shape differs')
                        need(hashlib.sha256(values['state_h'].tobytes()).hexdigest()==fingerprints[(key[0],key[1])][str(key[2])],'rawState fingerprint differs from original live FPcapture')
                        q=values['q_next_readout_H'].astype(np.float64);k=values['k_next_H'].astype(np.float64);alpha=values['alpha_next'].astype(np.float64);beta=values['beta_next'].astype(np.float64)
                        reference=(alpha[:,None]*(q-beta[:,None]*k*np.sum(k*q,axis=-1,keepdims=True))).astype(np.float32)
                        need(np.array_equal(reference,values['transported_next']),'independent original-coordinate transported-query algebra differs');gamma_exact+=1
                        need(np.array_equal(values['q_next_H_pre_scale'][0::2],values['q_next_H_pre_scale'][1::2]) and np.array_equal(values['k_next_H'][0::2],values['k_next_H'][1::2]),'original actual QK16-to32 sharing differs');paired_qk_checks+=1
                    else:
                        need(key in expectedD and key not in gotD,'drive roster differs');gotD.add(key)
                        need(all(values[n].shape==(32,128) for n in ('q_H_pre_scale','q_readout_H','key_H','value')) and all(values[n].shape==(32,) for n in ('g','alpha','beta')),'actual drive shape differs')
                        need(np.all(values['g']<=0) and np.all((values['alpha']>=0)&(values['alpha']<=1)) and np.all((values['beta']>=0)&(values['beta']<=1)),'actual native gate/beta domain differs')
        need(gotA==expectedA and gotD==expectedD,'actual scientific raw is incomplete')
        need(kinds=={'FP_STATE_AND_NEXT_TRANSPORT':480,'ACTUAL_FP_GDN_DRIVE':3840,'RAW_CAL_FP64_SUMS':1,'FROZEN_SHARED_TRANSPORTED_OMEGA':1},'exact raw scientific roster differs')
        omega=read(R/'TRANSPORTED_QUERY_OMEGA.json');w=np.asarray(omega['omega_normalized_fp32'],np.float32)
        need(omega['collection_manifest_sha256']==manifest_sha and omega['source_freeze_sha256']==phase_sha and omega['AIME_used'] is False and omega['VAL_used_for_omega'] is False and omega['layer_ids']==list(L) and w.shape==(24,32,128),'omega provenance/identity differs')
        need(np.isfinite(w).all() and np.all((w>=1/16)&(w<=16)),'actualomega finite/frozenclamp differs')
        need(set(manifest['omega']['counts'].values())=={1020} and set(map(int,manifest['omega']['counts']))==set(L),'omega actual CAL transition coverage differs')
        with np.load(raw/'OMEGA_FP32.npz',allow_pickle=False) as z:
            need(all(np.array_equal(w[i],z[str(l)]) for i,l in enumerate(L)),'omega JSON/raw FP32 bytes differ')
        obs=observe_nvml();need(not any(v['pid']==ident['pid'] for v in obs),'original workerPID remains in currentNVML')
        need(sha(R/'calibration/PARENT_FAILURE.json')==fail_sha and sha(R/'calibration/FP_COLLECTION_MANIFEST.json')==manifest_sha,'original immutable receipts changed during independent audit')
        audit=dict(status='PASS_INDEPENDENT_CPU_COMPLETE_FP_ACTUAL_EXIT_RELEASE_AND_RAW',task_id=R.name,observer=identity(os.getpid()),worker_identity=ident,original_parent_identity=parent['identity'],parent_failure_preserved=True,parent_failure_sha256=fail_sha,parent_failure_reclassified=False,original_parent_failure= failure['error'],transient_parent_observation_cause='UNRESOLVED; terminalexit race is an inference only',actual_Popen_wait_returncode=0,actual_release='PASS',original_wait_release_evidence=cleanup,current_NVML=obs,source_freeze_sha256=phase_sha,phaseA_sources_count=len(source_rows),phaseA_sources=source_rows,model17_current_SHA_bytes_stat='PASS',model_files=model_rows,native_pre_post=native_rows,collection_manifest_sha256=manifest_sha,raw_files=rows,raw_count=len(rows),raw_bytes=sum(v['bytes'] for v in rows),raw_kinds=kinds,raw_State_fingerprint_checks=480,independent_transport_algebra_exact_checks=gamma_exact,original_QK_head_sharing_checks=paired_qk_checks,actual_FP_tokens=1280,actual_helper_calls=30720,CAL_transitions_per_layer_head=1020,omega_current_JSON_raw_exact='PASS',AIME_used=False,model_loads=0,GPU_calls=0,GPU_retries=0,readonly_NVML_observations=1,wall_seconds=time.monotonic()-before)
        audit_path=R/'calibration/INDEPENDENT_FP_EXIT_RAW_AUDIT.json';audit_sha=new(audit_path,audit)
        receipt=dict(status='PASS_ACTUAL_EXIT0_AND_RELEASE',exit_returncode=0,release='PASS',worker_identity=ident,collection_manifest_sha256=manifest_sha,evidence='original failedCPUparent exactPopen.cleanup.actual_exit0+actualownedNVMLrelease; independentCPUsource/model/native/allraw/currentidentity verification',independent_audit_sha256=audit_sha,original_parent_failure_sha256=fail_sha,parent_failure_preserved=True,parent_failure_reclassified=False,original_parent_observation_cause='UNRESOLVED',GPU_calls=0,GPU_retries=0)
        receipt_path=R/'calibration/FP_COLLECTION_EXIT_RELEASE.json';receipt_sha=new(receipt_path,receipt)
        return dict(status=audit['status'],audit_sha256=audit_sha,audit_bytes=audit_path.stat().st_size,exit_receipt_sha256=receipt_sha,exit_receipt_text=receipt_path.read_text(),audit_summary={k:v for k,v in audit.items() if k not in ('raw_files','phaseA_sources','model_files')})
    except BaseException as error:
        new(R/'calibration/INDEPENDENT_FP_TERMINAL_AUDIT_FAILURE.json',dict(status='INDEPENDENT_CPU_FP_AUDIT_FAILED_NO_RETRY',error=repr(error),traceback=traceback.format_exc(),observer=identity(os.getpid()),GPU_calls=0,GPU_retries=0,preserve_originals=True));raise

if __name__=='__main__':print(json.dumps(main(),sort_keys=True))
