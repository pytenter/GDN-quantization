#!/usr/bin/env python3
"""Metadata/output adapter only; unmodified canonical generator does all computation."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1')
RUNNER=OLD/'scripts/run_qwen_c5_c6_formal.py'
EXPECTED_RUNNER='77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69'
ARMS=('C5R','C6R','C7')


def generator():
    spec=importlib.util.spec_from_file_location('unchanged_canonical3090_generator',RUNNER)
    g=importlib.util.module_from_spec(spec);sys.modules[spec.name]=g;spec.loader.exec_module(g)
    if g.file_sha256(RUNNER)!=EXPECTED_RUNNER:raise RuntimeError('frozen canonical runner drift')
    # New method labels only; operators, sampling, normalization, QDQ remain byte-identical source.
    g.METHODS={a.lower():'QWEN_INT8_C128_KEY_'+a for a in ARMS}
    return g


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--worker',type=int,choices=range(4),required=True)
    args=p.parse_args();g=generator();physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical!=str(args.worker):raise RuntimeError('fixed physical GPU assignment differs')
    config=ROOT/'protocol/canonical_endpoint.json';cfg=json.loads(config.read_text())
    if g.file_sha256(g.DATASET)!=cfg['dataset_sha256']:raise RuntimeError('canonical dataset changed')
    if not (ROOT/'deployment/canonical_3090_smoke.json').exists():raise RuntimeError('deployment smoke absent')
    if json.loads((ROOT/'deployment/canonical_3090_smoke.json').read_text())['status']!='PASS':raise RuntimeError('deployment smoke not PASS')
    manifest=ROOT/f'deployment/{args.arm.lower()}_manifest.json';identity=json.loads(manifest.read_text())
    rotation=ROOT/'deployment'/args.arm/'frozen_rotation_fp32.pt'
    import canonical3090_pipeline as P
    P.check_freeze()
    dataset={r['problem_id']:r for r in g.load_frozen_dataset(g.DATASET)}
    offset=ARMS.index(args.arm)*60
    jobs=[(dataset[j['problem_id']],j['seed'],offset+j['pair_index']) for j in cfg['jobs'] if j['gpu']==args.worker]
    if len(jobs)!=15:raise RuntimeError('Canonical60 shard not 15')
    path=ROOT/f'aime26/{args.arm.lower()}/shards/worker{args.worker}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise RuntimeError('existing endpoint shard: no automatic semantic rerun')
    model,tokenizer=g.load_model_and_tokenizer();device=next(model.parameters()).device
    bank,deployment=g.load_frozen_rotation(rotation,manifest,device,args.arm)
    cfgsha=g.file_sha256(config)
    with path.open('x') as f:
        for item,seed,index in jobs:
            if shutil.disk_usage('/data').free<15*1024**3:raise RuntimeError('below inherited 15GiB; do not start new sample')
            print(json.dumps({'event':'formal_unit_start','condition':args.arm,'problem_id':item['problem_id'],'seed':seed,'GPU':physical}),flush=True)
            r=g.generate_one(model,tokenizer,item,seed,args.arm.lower(),physical,args.worker,index,bank,deployment,cfgsha)
            if not g.is_valid_record(r,args.arm.lower()):
                with path.with_suffix('.attempts.jsonl').open('a') as a:a.write(json.dumps(r,ensure_ascii=False)+'\n');a.flush();os.fsync(a.fileno())
                raise RuntimeError('canonical generation failed; no retry')
            f.write(json.dumps(r,sort_keys=True,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    gate=g.validate_records(rows,args.arm.lower(),15,[(r['problem_id'],s) for r,s,_ in jobs])
    path.with_suffix('.summary.json').write_text(json.dumps(gate,indent=2)+'\n')
    if gate['gate']!='PASS':raise RuntimeError('canonical shard identity gate failed')


if __name__=='__main__':main()
