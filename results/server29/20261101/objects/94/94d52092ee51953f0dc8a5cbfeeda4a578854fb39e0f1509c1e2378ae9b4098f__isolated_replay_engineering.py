"""Fail-closed engineering qualification using unchanged production functions.

No full model, cumsum, learned parameters, free generation or scientific units.
"""
import argparse
import ast
import contextlib
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).parent))
from capture_original_drive import AMEND,ROOT,FORMAL,POLICY,read,save,sha,tensor_sha,load_module

GATES=AMEND/'engineering/gates'
RULES=AMEND/'protocol/remaining_engineering_gate_policy.json'
LAYERS=[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30]
CONDITIONS=['Native','H','C6R','C7']

def require(phase):
    assert read(GATES/(phase+'.json'))['gate']=='PASS',phase+' prerequisite not PASS'

def source_identity():
    p=read(POLICY)
    receipt=read(AMEND/'capture_freeze_receipt.json')
    assert sha(RULES)==receipt['remaining_engineering_policy_sha256'],'frozen engineering policy changed'
    for path,expected in p['source_sha256'].items():assert sha(path)==expected,'production source changed'
    return p

def phase_factorization():
    assert read(AMEND/'engineering/capture_runtime_variability.json')['CAPTURE_EMPIRICAL_REPRODUCIBILITY_GATE']=='PASS'
    p=source_identity()
    arch=Path(next(k for k in p['source_sha256'] if k.endswith('modeling_qwen3_5.py')))
    text=arch.read_text();tree=ast.parse(text)
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='torch_recurrent_gated_delta_rule')
    fragment=ast.get_source_segment(text,helper)
    assert 'cumsum' not in fragment and 'kv_mem' in fragment and 'delta' in fragment
    args=[a.arg for a in helper.args.args]
    assert args[:7]==['query','key','value','g','beta','initial_state','output_final_state']
    producer={n:('in_proj_qkv -> original causal_conv1d/update -> reshape -> repeat_interleave' if n in ['query','key','value'] else
                'in_proj_a/A_log/dt_bias -> original float softplus' if n=='g' else 'in_proj_b -> original sigmoid') for n in p['tensors']}
    rows=[]
    for n in p['tensors']:
        rows.append(dict(name=n,shape=p['required_qkv_shape'] if n in ['query','key','value'] else p['required_g_beta_shape'],
            dtype=p['expected_dtype'][n],producer=producer[n],consumer='original recurrent helper',
            depends_on_replayed_recurrent_state=False,depends_on_capture_FP_history_transitively=True,
            depends_on_blocked_cumsum_transitively_in_capture=True,frozen_external=True,recomputed_in_replay=False))
    for n in ['decayed_recurrent_state','kv_mem','delta','next_recurrent_state','quantizer_scale','quantizer_codes','QDQ_writeback_state']:
        rows.append(dict(name=n,dtype='original production dtype',producer='unchanged recurrent helper / formal quantizer',
            consumer='state transition',depends_on_replayed_recurrent_state=True,depends_on_blocked_cumsum_in_replay=False,
            frozen_external=False,recomputed_in_replay=True))
    save(GATES/'drive_dependency_graph.json',dict(source_path=str(arch),source_sha256=sha(arch),nodes=rows,
        scope='Inputs are external to replayed state, NOT independent of canonical acquisition FP model history',
        no_state_dependent_intermediate_frozen=True))
    save(GATES/'drive_schema.json',dict(tensors=rows[:5],token_indexing='1-based sequential,1..32768',layer_order=LAYERS,
        pre_normalization_inputs=True,initial_state='zero; original helper initial_state=None at first token',
        original_dtype=True,all_qkv_repeated32_heads_retained=True,chunked_shards=True,shard_tokens=64,
        scientific_sparse_FP_reference_checkpoints=[2048,8192,32768],full_model_hidden_trajectory=False,
        mutable_state_never_stored_as_drive=True,canonical_capture='FIRST_SUCCESSFUL_VALID_CAPTURE'))
    path=GATES/'drive_factorization_report.md';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        f.write('# Frozen-drive factorization\n\nExternal q/k/v/g/beta are acquired BEFORE production normalization and the recurrent helper. '
            'They depend on canonical FP acquisition history transitively, including original prefill cumsum, but never on any replayed anchor state. '
            'Decay multiplication, erase/project kv_mem, beta-weighted delta/write, condition-specific scale/code/dequant and state writeback are recomputed in the unchanged helper/quantizer. '
            'No state-dependent intermediate is frozen. State readout query is retained even though it is not used by the state-update equation. '
            'Source AST and hashes are authoritative; no recurrence formula is copied or rewritten.\n')
    save(GATES/'factorization.json',dict(gate='PASS',DRIVE_FACTORIZATION_GATE='PASS',
        source_sha256=sha(arch),schema_sha256=sha(GATES/'drive_schema.json'),dependency_graph_sha256=sha(GATES/'drive_dependency_graph.json')))

def phase_schema():
    import torch
    require('factorization');p=source_identity()
    path=AMEND/'engineering/capture_01';m=read(path/'drive_manifest.json');rows=[]
    assert m['layer_ids']==LAYERS and m['token_count']==256 and m['all_complete_finite']
    next_token=1
    for s in m['shards']:
        assert s['first_token']==next_token and sha(path/s['relative_path'])==s['file_sha256']
        v=torch.load(path/s['relative_path'],map_location='cpu',weights_only=True)
        count=s['last_token']-s['first_token']+1
        assert set(v['layers'])==set(map(str,LAYERS))
        for layer in LAYERS:
            assert set(v['layers'][str(layer)])==set(p['tensors'])
            for name in p['tensors']:
                t=v['layers'][str(layer)][name]
                expected=[1,count,32,128] if name in ['query','key','value'] else [1,count,32]
                assert list(t.shape)==expected and str(t.dtype)==p['expected_dtype'][name]
                assert bool(torch.isfinite(t).all())
                identity=next(r for r in s['tensors'] if r['layer']==layer and r['name']==name)
                assert tensor_sha(t)==identity['sha256']
                rows.append(dict(first_token=s['first_token'],last_token=s['last_token'],layer=layer,name=name,shape=expected,dtype=str(t.dtype),finite=True))
        next_token=s['last_token']+1
    assert next_token==257 and len(rows)==4*24*5
    save(GATES/'schema.json',dict(gate='PASS',CAPTURE_SCHEMA_GATE='PASS',rows=rows,drive_manifest_sha256=sha(path/'drive_manifest.json')))

def compile_original(path,names,namespace):
    source=path.read_text();nodes=[]
    for name in names:
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
        node.decorator_list=[] # reuse unchanged fallback BODY, not Hub dispatch.
        nodes.append(node)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace

class Replay:
    def __init__(self):
        import torch
        self.torch=torch;self.policy=source_identity();self.rules=read(RULES)
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.F=load_module(FORMAL,'isolated_original_formal')
        self.F.setup_seed(self.policy['capture_seed'])
        arch=Path(next(k for k in self.policy['source_sha256'] if k.endswith('modeling_qwen3_5.py')))
        ns=compile_original(arch,['l2norm','torch_recurrent_gated_delta_rule'],{'torch':torch})
        self.helper=ns['torch_recurrent_gated_delta_rule']
        self.layers=LAYERS;self.banks={};self.patches={'H':self.F.QwenKeyHadamardPatch(None)}
        anchors=read(ROOT/'protocol/frozen_anchor_conditions.json')
        for condition,alias in [('C6R','C5'),('C7','C6')]:
            row=next(r for r in anchors['anchors'] if r['condition']==condition)
            assert sha(row['export_path'])==row['export_file_sha256']
            payload=torch.load(row['export_path'],map_location='cpu',weights_only=True)
            bank=self.F.FrozenRotationBank(payload['layer_ids'],payload['matrices']).to('cuda:0')
            self.banks[alias]=bank
            self.patches[condition]=self.F.QwenFrozenRotationPatch(None,bank,{'export_file_sha256':row['export_file_sha256']})
        physical_path=Path(next(k for k in self.policy['source_sha256'] if k.endswith('sparse_probe.py')))
        self.physical_fn=compile_original(physical_path,['physical_state'],{'torch':torch})['physical_state']
        self.H=self.F.hadamard(128,device='cuda:0');self.aliases={'H':'H','C6R':'C5','C7':'C6'}
        self.calls={'blocked_cumsum':0,'full_model_forward':0,'QDQ_path':0,'QDQ_reference':0}

    @contextlib.contextmanager
    def guard(self):
        t=self.torch;old_fun=t.cumsum;old_method=t.Tensor.cumsum;old_call=t.nn.Module.__call__
        def blocked(*a,**k):
            self.calls['blocked_cumsum']+=1;raise RuntimeError('Replay attempted blocked cumsum; STOP')
        def blocked_model(*a,**k):
            self.calls['full_model_forward']+=1;raise RuntimeError('Replay attempted module/full-model forward; STOP')
        t.cumsum=blocked;t.Tensor.cumsum=blocked;t.nn.Module.__call__=blocked_model
        try:
            with t.inference_mode():yield
        finally:t.cumsum=old_fun;t.Tensor.cumsum=old_method;t.nn.Module.__call__=old_call

    def physical(self,c,l,s):
        if c=='Native':return s.detach().float()
        return self.physical_fn(self.F,self.aliases[c],l,s,self.banks)

    def inverse(self,c,l,s):
        if c=='Native':return s
        y=self.torch.einsum('ij,...jv->...iv',self.H,s)
        if c!='H':y=self.torch.einsum('ij,...jv->...iv',self.banks[self.aliases[c]].layer(l).matrix().T,y)
        return y

    def step(self,c,l,s,r):
        q=r['query'];k=r['key'];norm=True
        if c!='Native':
            patch=self.patches[c];patch.current_layer=l
            q=patch._transform(q,True);k=patch._transform(k,True);norm=False
        _,state=self.helper(q,k,r['value'],g=r['g'],beta=r['beta'],initial_state=s,output_final_state=True,use_qk_l2norm_in_kernel=norm)
        return state

    def qdq(self,states,role):
        audit=dict(calls=0,layers=set(),scale_shapes={},code_min=0,code_max=0)
        self.F.quantize_c128(SimpleNamespace(recurrent_states=states),audit)
        assert audit['calls']==1 and sorted(audit['layers'])==self.layers
        self.calls['QDQ_'+role]+=1

    def drive(self,length=256,capture_path=None):
        path=capture_path or AMEND/'engineering/capture_01'
        m=read(path/'drive_manifest.json')
        assert length<=m['token_count']
        for s in m['shards']:
            if s['first_token']>length:break
            assert sha(path/s['relative_path'])==s['file_sha256']
            blob=self.torch.load(path/s['relative_path'],map_location='cpu',weights_only=True)
            values={l:{name:t.to('cuda:0') for name,t in blob['layers'][str(l)].items()} for l in self.layers}
            for offset in range(s['last_token']-s['first_token']+1):
                token=s['first_token']+offset
                if token>length:break
                yield token,{l:{name:t[:,offset:offset+1] for name,t in values[l].items()} for l in self.layers}

def phase_fp():
    require('schema');R=Replay();path=AMEND/'engineering/capture_01'
    reference=read(path/'fp_state_hashes.json');rows=[]
    with R.guard():
        states={l:None for l in R.layers}
        for token,records in R.drive():
            for l in R.layers:
                states[l]=R.step('Native',l,states[l],records[l])
                actual=tensor_sha(states[l]);expected=reference[token-1]['per_layer_sha256'][str(l)]
                row=dict(token=token,layer=l,expected_sha256=expected,actual_sha256=actual,exact=actual==expected)
                rows.append(row)
                if not row['exact']:
                    save(GATES/'fp.json',dict(gate='FAIL',FP_REPLAY_PARITY_GATE='FAIL',first_failure=row,rows=rows,
                        numerical_policy='Frozen parent exact FP trajectory check; no fallback/relaxation',calls=R.calls))
                    raise RuntimeError('FP replay trajectory hash mismatch')
            if token in [32,256]:print('FP_REPLAY_LENGTH_'+str(token)+'_PASS',flush=True)
    save(GATES/'fp.json',dict(gate='PASS',FP_REPLAY_PARITY_GATE='PASS',lengths=[32,256],
        rows=rows,checked_states=24*256,all_tokens_exact=True,calls=R.calls,
        REPLAY_BLOCKED_CUMSUM_CALL_COUNT=0,REPLAY_FULL_MODEL_FORWARD_COUNT=0,
        drive_manifest_sha256=sha(path/'drive_manifest.json')))

def phase_invariance():
    require('fp');R=Replay();t=R.torch;maximum={c:0. for c in CONDITIONS};worst={};count=0
    with R.guard():
        fp={l:None for l in R.layers};states={c:{l:None for l in R.layers} for c in CONDITIONS}
        for token,records in R.drive():
            for l in R.layers:
                fp[l]=R.step('Native',l,fp[l],records[l])
                for c in CONDITIONS:
                    states[c][l]=R.step(c,l,states[c][l],records[l])
                    native=R.physical(c,l,states[c][l]);scale=max(1.,float(native.abs().max()),float(fp[l].abs().max()))
                    difference=float((native-fp[l]).abs().max());bound=8*R.rules['NO_QDQ_ROTATION_INVARIANCE_GATE']['gamma128']*scale
                    row=dict(token=token,layer=l,condition=c,max_abs_difference=difference,bound=bound,scale=scale,
                        PASS=bool(t.isfinite(native).all()) and difference<=bound)
                    count+=1
                    if difference>maximum[c]:maximum[c]=difference;worst[c]=row
                    if not row['PASS']:
                        save(GATES/'invariance.json',dict(gate='FAIL',NO_QDQ_ROTATION_INVARIANCE_GATE='FAIL',first_failure=row,
                            comparisons_completed=count,maximum_by_condition=maximum,calls=R.calls))
                        raise RuntimeError('Unquantized rotation numerical invariance failed')
            if token%64==0:print('INVARIANCE_TOKEN='+str(token),flush=True)
    save(GATES/'invariance.json',dict(gate='PASS',NO_QDQ_ROTATION_INVARIANCE_GATE='PASS',conditions=CONDITIONS,
        comparisons_completed=count,expected_comparisons=256*24*4,maximum_by_condition=maximum,worst_rows=worst,
        calls=R.calls,quantizer_calls=0,criterion='Prospectively frozen inherited8*gamma128 operand-scale bound; no exact arithmetic claim'))

def phase_decomposition():
    require('invariance');R=Replay();t=R.torch;u64=2**-53;n=32*128*128;gamma=n*u64/(1-n*u64)
    observations={c:[] for c in CONDITIONS};summary={c:{'vector_max':0.,'energy_max':0.} for c in CONDITIONS}
    with R.guard():
        fp={l:None for l in R.layers};states={c:{l:None for l in R.layers} for c in CONDITIONS}
        for token,records in R.drive():
            fp_old=fp;fp={l:R.step('Native',l,fp_old[l],records[l]) for l in R.layers}
            for c in CONDITIONS:
                pre={};reference={};hist={}
                for l in R.layers:
                    native_old=t.zeros_like(fp[l]) if states[c][l] is None else R.physical(c,l,states[c][l])
                    hist[l]=R.step('Native',l,native_old,records[l])
                    pre[l]=R.step(c,l,states[c][l],records[l])
                    old_ref=t.zeros_like(fp[l]) if fp_old[l] is None else fp_old[l]
                    reference[l]=R.step(c,l,R.inverse(c,l,old_ref),records[l])
                R.qdq(pre,'path');R.qdq(reference,'reference')
                normalized=[];raw=[]
                for l in R.layers:
                    sq=R.physical(c,l,pre[l]).double();f=fp[l].double();history=hist[l].double();clean=R.physical(c,l,reference[l]).double()
                    E=sq-f;H=history-f;IP=sq-history;IR=clean-f;interaction=IP-IR
                    vector=float((E-H-IP).abs().max());scale=max(1.,float(sq.abs().max()),float(f.abs().max()),float(history.abs().max()))
                    energies=t.stack([(E*E).sum(),(H*H).sum(),(IP*IP).sum(),(IR*IR).sum(),2*(H*IP).sum(),(interaction*interaction).sum()])
                    numeric=energies.cpu().tolist();energy=abs(numeric[0]-numeric[1]-numeric[2]-numeric[4])
                    bound=16*gamma*max(1.,numeric[0],numeric[1],numeric[2],abs(numeric[4]))
                    finite=all(math.isfinite(x) for x in numeric)
                    if not finite or vector>8*u64*scale or energy>bound:
                        row=dict(token=token,layer=l,condition=c,vector_residual=vector,vector_bound=8*u64*scale,
                            energy_residual=energy,energy_bound=bound,raw_energies=numeric,finite=finite)
                        save(GATES/'decomposition.json',dict(gate='FAIL',COUNTERFACTUAL_DECOMPOSITION_GATE='FAIL',first_failure=row,calls=R.calls))
                        raise RuntimeError('Counterfactual vector/energy closure failed')
                    denom=float((f*f).sum())+R.rules['epsilon']
                    raw.append(numeric);normalized.append([x/denom for x in numeric])
                    summary[c]['vector_max']=max(summary[c]['vector_max'],vector)
                    summary[c]['energy_max']=max(summary[c]['energy_max'],energy)
                observations[c].append(dict(token=token,normalized=normalized,raw_energies=raw))
                states[c]=pre
            if token%32==0:print('DECOMPOSITION_TOKEN='+str(token),flush=True)
    save(GATES/'engineering_energy_observations.json',dict(engineering_only=True,scientific_window_applied=False,
        no_scientific_contrasts=True,observations=observations,metric_names=R.rules['scientific_metric_names'],
        drive_manifest_sha256=sha(AMEND/'engineering/capture_01/drive_manifest.json')))
    save(GATES/'decomposition.json',dict(gate='PASS',COUNTERFACTUAL_DECOMPOSITION_GATE='PASS',conditions=CONDITIONS,
        comparisons=256*24*4,summary=summary,calls=R.calls,FP64_accumulator=True,scientific_measurement_count=0,
        observations_sha256=sha(GATES/'engineering_energy_observations.json')))

def phase_isolation():
    require('decomposition');rows=[]
    for phase in ['fp','invariance','decomposition']:
        v=read(GATES/(phase+'.json'));calls=v['calls']
        assert calls['blocked_cumsum']==0 and calls['full_model_forward']==0
        rows.append(dict(phase=phase,source_file_sha256=sha(GATES/(phase+'.json')),calls=calls))
    save(GATES/'isolation.json',dict(gate='PASS',REPLAY_ISOLATION_GATE='PASS',rows=rows,
        REPLAY_BLOCKED_CUMSUM_CALL_COUNT=0,REPLAY_FULL_MODEL_FORWARD_COUNT=0,
        runtime_guard='torch.cumsum, Tensor.cumsum and nn.Module.__call__ fail closed; guards restored after replay',
        recurrence='Unchanged production fallback function AST; no whole Transformer imported/instantiated by replay'))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['factorization','schema','fp','invariance','decomposition','isolation'],required=True)
    args=parser.parse_args();GATES.mkdir(parents=True,exist_ok=True)
    try:
        frozen=read(AMEND/'short_qualification_freeze.json')
        assert sha(Path(__file__))==frozen['replay_engineering_script_sha256'],'frozen qualification script drift'
        assert sha(RULES)==frozen['remaining_engineering_policy_sha256'],'frozen numerical policy drift'
        globals()['phase_'+args.phase]()
    except Exception as e:
        p=GATES/(args.phase+'.error.json')
        if not p.exists():save(p,dict(status='FAIL_AND_STOP',phase=args.phase,exception=type(e).__name__,message=str(e),
            traceback=traceback.format_exc(),retry_authorized=False,scientific_discovery_units=0))
        raise

if __name__=='__main__':main()
