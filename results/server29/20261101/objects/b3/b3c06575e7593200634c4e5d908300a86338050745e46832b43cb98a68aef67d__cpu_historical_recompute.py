#!/usr/bin/env python3
"""Read-only historical metadata recomputation. No torch, scoring, or GPU imports."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

TASK = 'QWEN_GDN_HISTORICAL_THROUGHPUT_RECONCILIATION_V1'
HISTORY = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen')
ORIGINAL = Path('/data/zypan/worktrees/aime26-sglang-rotation-v1')
METHODS = ('QWEN_FP_STATE', 'QWEN_INT8_C128_KEY_HADAMARD')

def sha(raw): return hashlib.sha256(raw).hexdigest()
def read(p): return json.loads(p.read_bytes())
def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
def stats(rows):
    if not rows: return {'N': 0, 'status': 'NO_HISTORICAL_SAMPLES'}
    rates = [r['recomputed_TPS'] for r in rows]
    nt = sum(r['output_tokens'] for r in rows)
    tm = math.fsum(r['latency_s'] for r in rows)
    return {'N':len(rows),'status':'ACTUAL_METADATA_RECOMPUTED','sum_output_tokens':nt,
            'sum_latency_s':tm,'mean_TPS':statistics.mean(rates),'median_TPS':statistics.median(rates),
            'weighted_TPS':nt/tm,'min_TPS':min(rates),'max_TPS':max(rates),
            'input_tokens_min':min(r['input_tokens'] for r in rows),'input_tokens_max':max(r['input_tokens'] for r in rows),
            'output_tokens_min':min(r['output_tokens'] for r in rows),'output_tokens_max':max(r['output_tokens'] for r in rows),
            'EOS_count':sum(r['eos_seen'] for r in rows),'cap_count':sum(r['truncated'] for r in rows)}
def corr(rows, key):
    x=[r[key] for r in rows];y=[r['recomputed_TPS'] for r in rows]
    mx=statistics.mean(x);my=statistics.mean(y)
    denominator=math.sqrt(math.fsum((a-mx)**2 for a in x)*math.fsum((a-my)**2 for a in y))
    return math.fsum((a-mx)*(b-my) for a,b in zip(x,y))/denominator if denominator else None
def write_csv(path, rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f, fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root
    prov_path=HISTORY/'results/aime26/81920/qwen_gdn/PROVENANCE.json'
    prov_raw=prov_path.read_bytes();prov=json.loads(prov_raw)
    records=[];sources=[]
    for entry in prov['source_files']:
        if entry['condition'] not in METHODS:continue
        p=Path(entry['original_path']);before=p.stat();raw=p.read_bytes();after=p.stat()
        assert sha(raw)==entry['sha256'] and len(raw)==entry['size'],str(p)
        assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
        lines=[line for line in raw.splitlines() if line.strip()]
        assert len(lines)==15
        sources.append({'path':str(p),'raw_sha256':sha(raw),'bytes':len(raw),'rows':len(lines),'historical_manifest_match':True})
        for line_number,line in enumerate(lines,1):
            j=json.loads(line);n=j['output_tokens'];t=j['latency_s']
            assert j['method']==entry['condition'] and n==len(j['generated_token_ids'])==j['generated_token_count']
            assert not j['runtime_error'] and j['nonfinite'] is False and t>0
            assert j['finish_reason'] in ['eos','length']
            assert j['eos_seen']==(j['finish_reason']=='eos') and j['truncated']==(j['finish_reason']=='length')
            assert not j['truncated'] or n==81920
            assert j['tokens_per_s']==n/t
            keys=['problem_id','seed','method','input_tokens','output_tokens','latency_s','tokens_per_s','finish_reason',
                  'eos_seen','truncated','runtime_error','nonfinite','gpu','worker_id','git_commit','runtime_version',
                  'input_ids_hash','num_workers','max_new_tokens','state_quantization','rotation','quantizer_audit','hadamard_audit']
            row={k:j.get(k) for k in keys};row.update(recomputed_TPS=n/t,raw_source=str(p),raw_line=line_number,raw_record_bytes_SHA256=sha(line))
            records.append(row)
    assert len(records)==120
    result={'task_id':TASK,'status':'PASS_RAW_HISTORICAL_METADATA_RECOMPUTATION','raw_sources':sources,
            'provenance':{'path':str(prov_path),'sha256':sha(prov_raw)},'methods':{},
            'definition':'Historical requested13.38/9.12 are sumoutputtokens/sumlatency across60batch1requests on4separateGPUworkers; not4GPUaggregate throughput.',
            'timing_scope':'Original generate_one includes context-manager setup,prefill,decode,sampler,QDQ,finite/scalarchecks; excludes load,tokenization/inputtransfer,final textdecode,scoring and JSONI/O.',
            'historical_GPU_UUID':'UNKNOWN_NOT_RECORDED_IN_RAW_BASELINE_ROWS','answers_or_correctness_rescored':False}
    length_rows=[];gpu_rows=[]
    bins=[('0_to_2048',0,2048),('2048_to_8192',2048,8192),('8192_to_32768',8192,32768),('32768_to_81920',32768,81920)]
    for method in METHODS:
        rr=[r for r in records if r['method']==method];assert len(rr)==60
        assert len({(r['problem_id'],r['seed']) for r in rr})==60
        result['methods'][method]=dict(stats(rr),stored_per_request_TPS_exact=True,
            output_length_TPS_Pearson_descriptive=corr(rr,'output_tokens'),prefill_length_TPS_Pearson_descriptive=corr(rr,'input_tokens'),
            runtime_versions=list({json.dumps(r['runtime_version'],sort_keys=True) for r in rr}),
            git_commits=sorted({r['git_commit'] for r in rr}),GPU_indices=sorted({r['gpu'] for r in rr}),
            displayed_historical_TPS=13.38 if method==METHODS[0] else 9.12)
        for label,lo,hi in bins:
            gg=[r for r in rr if lo<r['output_tokens']<=hi];length_rows.append(dict(method=method,length_group=label,interval='(lower,upper]',**stats(gg)))
        for gpu in sorted({r['gpu'] for r in rr}):
            gpu_rows.append(dict(method=method,recorded_physical_GPU_index=gpu,historical_GPU_UUID='UNKNOWN',**stats([r for r in rr if r['gpu']==gpu])))
    root.mkdir(parents=True,exist_ok=True)
    save(root/'HISTORICAL_TPS_RECOMPUTATION.json',result)
    save(root/'HISTORICAL_PER_GENERATION_METADATA.json',{'task_id':TASK,'status':'ACTUAL_METADATA_ONLY_NO_ANSWERS_OR_RESCORING','records':records})
    write_csv(root/'HISTORICAL_TPS_BY_LENGTH.csv',length_rows);write_csv(root/'HISTORICAL_TPS_BY_GPU.csv',gpu_rows)
    print(json.dumps({'status':result['status'],'methods':result['methods'],'raw_sources':len(sources)},ensure_ascii=False))

if __name__=='__main__': main()
