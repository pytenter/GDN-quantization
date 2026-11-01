"""New GPU3-only FP reference preparation. Original numerical capture/replay unchanged."""
from pathlib import Path
import hashlib,importlib.util,json,os,shutil,subprocess,sys,time,traceback
sys.dont_write_bytecode=True
W=Path('/data/zypan/worktrees/ling-observability-gpu3-expansion-v1')
A=W/'hardware_expansion_gpu3_v1'
R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3')
Q=R/'provenance_capsule_v3/scientific_reproducibility_v4_whole_runtime_replay'
HD=Path('/data/zypan/experiment_runs/ling_historical_runtime_replay_v1/checkout/experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1')
DEST=Path('/data/zypan/experiment_runs/ling_observability_gpu3_references_v1')

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def emit(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def git(*args,cwd=W):return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()
def load(p,name):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def main():
    assert not (A/'QUALIFICATION_STOP.json').exists()
    qual=read(A/'FRESH_PROCESS_NUMERIC_DETERMINISM_GATE.json');assert qual['gate']=='PASS' and qual['repeat_count']==3
    for row in qual['files']:assert sha(row['path'])==row['sha256']
    for run in ['fresh_process_01','fresh_process_02','fresh_process_03']:
        receipt=read(A/'qualification'/run/'QUALIFICATION_COMPLETE.json')
        assert receipt['gate']=='PASS' and receipt['hardware']['physical_gpu']==3
    inputs=read(A/'REFERENCE_PREPARATION_INPUT_COMMIT.json');commit=inputs['commit']
    assert git('rev-parse','HEAD')==commit
    assert not git('status','--porcelain=v1','--untracked-files=no')
    for row in inputs['inputs']:assert sha(row['path'])==row['sha256']
    assert git('rev-parse','HEAD',cwd=R)=='e428b5deb0579b926b2ad098a8699b3f4f2233f0'
    assert sha(HD/'scripts/bridge.py')=='e4baf3f7e3ec19f014434f8c28f1ecc77e87651487c547f9cd107c4f3eeafedd'
    for row in read(R/'provenance_capsule_v3/scientific_asset_manifest.json')['files']:
        assert sha(row['path'])==row['sha256'],('SCIENTIFIC_ASSET_MISMATCH',row['path'])
    for row in read(Q/'HISTORICAL_EXECUTED_PACKAGE_IDENTITY.json')['files']:
        assert sha(row['path'])==row['sha256'],('RUNTIME_IDENTITY_MISMATCH',row['path'])
    qualification=load(A/'gpu3_expansion_qualification.py','frozen_gpu3_assignment_hardware')
    b=load(HD/'scripts/bridge.py','unchanged_historical_reference_capture')
    co=b.imp(b.OLD_N/'scripts/coherence.py','gpu3_reference_original_coherence')
    co.GPU=3;co.UUID=qualification.UUID
    t,c,cap,core,stats=co.modules();torch=core.torch
    c.D=A/'reference_preparation';c.R=W;c.TASK=b.TASK;cap.D=c.D;cap.R=W;cap.TASK=b.TASK
    hw=qualification.hardware(torch)
    docs=read(R/'experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1/document_manifest.json')['documents']
    assignment=read(A/'HARDWARE_ASSIGNMENT_AMENDMENT.json')['proposed_document_assignment']
    selected=[d for d in docs if assignment[d['document_id']]==3]
    assert [d['document_id'] for d in selected]==['pg19-test-5396','pg19-test-26493']
    reserve=20*1024**3;required=len(selected)*13325495104
    assert shutil.disk_usage(W).free>required+reserve,'STORAGE_CAPACITY_FAIL'
    assert not DEST.exists();DEST.mkdir(parents=True)
    emit(A/'reference_preparation/STARTED.json',dict(time_unix=time.time(),hardware=hw,input_commit=commit,
        selected_documents=[d['document_id'] for d in selected],reference_preparation_only=True,FORMAL_MEASUREMENTS=0,
        FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO',GPU6_REFERENCE_REUSED=False))
    completed=[]
    for doc in selected:
        qualification.hardware(torch)
        source=t.mapped(doc['token_file']['path']);assert sha(source)==doc['token_file']['sha256']
        ids=read(source);assert len(ids)==32769
        assert hashlib.sha256(b''.join(int(v).to_bytes(4,'little') for v in ids)).hexdigest()==doc['token_ids_sha256']
        path=DEST/doc['document_id'];model=cap.Capture()
        try:trace=model.document(ids[:32768],path,block=512,keep_step_hashes=False)
        finally:model.close()
        assert trace['tokens']==32768 and trace['layer_ids']==b.LAYERS
        assert trace['token_sha256']==hashlib.sha256(b''.join(int(v).to_bytes(4,'little') for v in ids[:32768])).hexdigest()
        state=torch.zeros((18,16,128,128),device='cuda',dtype=torch.float32)
        expected_start=0
        with torch.inference_mode():
            for chunk in trace['chunks']:
                assert chunk['start_zero_based']==expected_start
                art=chunk['artifact'];assert sha(art['path'])==art['sha256']
                x=torch.load(art['path'],map_location='cuda',weights_only=False)
                for j in range(x['q'].shape[1]):
                    rec={k:x[k][:,j:j+1].contiguous() for k in ['q','k','v','beta','log_decay']}
                    state=core.update_pre_q(state,rec)
                assert torch.equal(state,x['fp_after']),('GPU3_FP_CAPTURE_REPLAY_BITWISE_FAIL',doc['document_id'],chunk['end_one_based'])
                expected_start=chunk['end_one_based'];del x
                print('GPU3_REFERENCE_REPLAY_CHUNK_PASS',doc['document_id'],expected_start,flush=True)
        assert expected_start==32768
        del state;torch.cuda.empty_cache()
        trace_file=path/'trace_freeze.json'
        record=dict(document=doc['document_id'],physical_gpu=3,gpu_uuid=qualification.UUID,
            input_commit=commit,token_ids_sha256=doc['token_ids_sha256'],trace_path=str(trace_file),trace_sha256=sha(trace_file),
            chunks=trace['chunks'],FP_CAPTURE_REPLAY_BITWISE_GATE='PASS',FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO',
            reference_preparation_only=True,FORMAL_MEASUREMENTS=0)
        emit(A/'reference_preparation'/f"{doc['document_id']}_FP_REFERENCE.json",record)
        completed.append(record)
        print('GPU3_SAME_GPU_DOCUMENT_REFERENCE_COMPLETE',doc['document_id'],flush=True)
    emit(A/'reference_preparation/FP_REFERENCE_MANIFEST.json',dict(gate='PASS',hardware=hw,references=completed,
        FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO',FORMAL_MEASUREMENTS=0,GPU3_FORMAL_AUTHORIZED='NO',
        next_required='scientific premeasurement freeze; minimal postcommit identity/parity; safe assignment handoff'))
    print('GPU3_FP_REFERENCES_COMPLETE_NOT_FORMAL_AUTHORIZATION',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        stop=A/'reference_preparation/STOP.json'
        if not stop.exists():emit(stop,dict(error=repr(e),traceback=traceback.format_exc(),FORMAL_MEASUREMENTS=0,
            GPU3_FORMAL_AUTHORIZED='NO',FINAL_ACTION='FAIL_AND_STOP'))
        raise
