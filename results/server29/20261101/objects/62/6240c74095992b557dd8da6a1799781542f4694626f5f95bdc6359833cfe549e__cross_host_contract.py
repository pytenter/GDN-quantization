"""CPU-only cross-host identities, global barrier and payload admission. No dispatch APIs."""
import hashlib,json
from pathlib import Path
import execution_control as C

CONDITIONS={'G64-0','G64-1','G64-2','G64-3'}
def validate_global_layout(root):
    root=Path(root);p=C.read(root/'CROSS_HOST_PLAN.json');nodes={n['id']:n for n in p['nodes']}
    C.require(set(nodes)=={'server55_gpu0','server29_gpu6'},'Only the two latest human-authorized nodes')
    expected={'server55_gpu0':('202.38.247.55',0,'GPU-a78176ed-f97f-c54f-6201-5a01b570f422'),'server29_gpu6':('202.38.247.29',6,'GPU-de53f243-b271-521c-d875-d96611ed8574')}
    for name,n in nodes.items():C.require((n['host'],n['physical_gpu'],n['uuid'])==expected[name],'Authorized physical host/UUID differs')
    jobs=C.read(root/'GLOBAL_FORMAL_JOB_MANIFEST.json')['jobs'];assignment=C.read(root/'QUESTION_GPU_ASSIGNMENT.json')['assignments'];records={x['problem_id']:x for x in C.read(root/'AIME26_DATASET_FREEZE.json')['records']}
    C.require(len(jobs)==120 and len({j['id']for j in jobs})==120 and len(assignment)==30 and len({a['problem_id']for a in assignment})==30,'Global120/30 cardinality differs')
    C.require({j['problem_id']for j in jobs}==set(records),'Global question roster differs')
    for a in assignment:
        n=nodes[a['node_id']];rows=[j for j in jobs if j['problem_id']==a['problem_id']]
        C.require(len(rows)==4 and [j['condition']for j in rows]==a['conditions']and set(a['conditions'])==CONDITIONS,'Same-question four-condition rotation differs')
        for j in rows:
            C.require(j['node_id']==n['id']and j['compute_host']==n['host']and j['gpu']==n['physical_gpu']and j['gpu_uuid']==n['uuid']and j['seed']==1 and j['max_new_tokens']==81920 and j['input_ids_sha256']==records[j['problem_id']]['input_ids_sha256']and j['input_tokens']==records[j['problem_id']]['input_tokens'],'Global job protocol/host/UUID differs')
    for name in nodes:C.require(sum(j['node_id']==name for j in jobs)==60 and sum(a['node_id']==name for a in assignment)==15,'Each node must15question bundles/60samples')
    q=p['qualification_units'];C.require(len(q)==38 and len({u['id']for u in q})==38,'Global qualification count differs')
    for name,n in nodes.items():
        rows=[u for u in q if u['node_id']==name];C.require(len(rows)==19 and all(u['gpu']==n['physical_gpu']and u['compute_host']==n['host']for u in rows),'Node qualification binding differs')
    return dict(status='PASS_GLOBAL38_120_FIXED_TWO_HOST_LAYOUT',nodes=nodes,qualification=38,formal=120)

def require_global_formal_admission(root,admission):
    layout=validate_global_layout(root);p=C.read(root/'CROSS_HOST_PLAN.json');expected=set(layout['nodes'])
    C.require(admission['status']=='PASS_BOTH_HOST19_TRUE_QUALIFICATIONS_ALLOW_FORMAL120_ONCE'and admission['global_plan_sha256']==C.sha(root/'CROSS_HOST_PLAN.json')and admission['global_jobs_sha256']==C.sha(root/'GLOBAL_FORMAL_JOB_MANIFEST.json')and admission['human_grant_sha256']==C.sha(root/'HUMAN_GPU_AUTHORIZATION.json'),'Global formal admission binding differs')
    C.require(set(admission['nodes'])==expected,'Both node qualifications are required')
    for name,x in admission['nodes'].items():
        n=layout['nodes'][name]
        C.require(x['node_id']==name and x['host']==n['host']and x['gpu_uuid']==n['uuid']and x['qualification_count']==19 and x['all19_actual_exit0_release_PASS']is True and x['own_worker_processes_and_UUID_released']is True,'Node qualification/actual-release proof missing')
        for key in ['source_sha256','permit_sha256','qualification_sha256','model_bytes_identity_sha256']:
            C.require(type(x[key])is str and len(x[key])==64 and all(c in'0123456789abcdef'for c in x[key]),'Malformed node identity digest')
    mine=admission['nodes'][C.NODE['id']]
    C.require(mine['source_sha256']==C.sha(root/'SOURCE_FREEZE.json')and mine['permit_sha256']==C.sha(root/'GPU_PERMIT.json')and mine['qualification_sha256']==C.sha(root/'GROUP64_GPU_CODEC_QUALIFICATION.json'),'Own true qualification/source/permit changed')
    C.require(len({x['model_bytes_identity_sha256']for x in admission['nodes'].values()})==1,'Cross-host model/tokenizer bytes differ')
    return True

def model_bytes_identity(root):
    xs=C.read(root/'analysis/MODEL_TOKENIZER_FRESH_SHA.json')['files'];rows=sorted((Path(x['path']).name,x['bytes'],x['sha256'])for x in xs)
    C.require(len(rows)==17,'Model/tokenizer17 roster differs')
    return C.objsha(rows)

def inspect_qualification(root):
    root=Path(root);ready=C.read(root/'LOCAL_QUALIFICATION_READY.json');qualified=C.read(root/'GROUP64_GPU_CODEC_QUALIFICATION.json');plan=C.read(root/'GPU_EXECUTION_PLAN.json')['qualification_units'];ledger=C.read(root/'RESOURCE_LEDGER.json')['intervals'];by={x['unit']:x for x in ledger};identities=[];checks=[]
    C.require(len(plan)==19 and set(qualified['units'])=={u['id']for u in plan}and ready['count']==19,'Node19 qualification incomplete')
    for u in plan:
        d=root/'execution'/u['id'];w=C.read(d/'RESULT.json');saved=w['payload_sha256'];v=dict(w);v.pop('payload_sha256');i=by[u['id']]
        C.require(C.objsha(v)==saved and w['status']=='PASS'and w['unit']==u['id']and w['source_freeze_sha256']==C.sha(root/'SOURCE_FREEZE.json')and w['permit_sha256']==C.sha(root/'GPU_PERMIT.json'),'Actual qualification payload identity differs')
        C.require(w['identity']==i['identity']==C.read(d/'STARTED.json')['identity']and i['exit_returncode']==0 and i['release']=='PASS'and w['gpu']==C.NODE['physical_gpu']and w['gpu_uuid']==C.NODE['uuid'],'Actual qualification host/identity/exit/release differs')
        C.require(C.read(d/'DRIVER_IDENTITY.json')['status']=='PASS'and C.read(d/'DRIVER_IDENTITY.json')['actual_uuid']==C.NODE['uuid']and C.read(d/'PREFLIGHT_LIBRARY.json')['status']=='PASS','Driver/native admission missing')
        if u['kind']!='gpu_codec_micro':C.require(C.read(d/'POSTFLIGHT_LIBRARY.json')['status']=='PASS','Postflight native admission missing')
        r=w['result']
        if u['kind']=='gpu_codec_micro':C.require(r['count']==74 and len(r['checks'])==74,'Real micro field count differs')
        elif u['kind']=='group64_persistent_fresh':
            C.require(r['tokens']==512 and [m['tokens']for m in r['milestones']]==[32,128,512]and r['audit']['qdq_layer_updates']==24*512 and r['audit']['next_consumption_checks']==24*511 and len(r['writebacks_by_layer'])==24 and all(v==512 for v in r['writebacks_by_layer'].values()),'Real persistent512 coverage differs')
            C.require([x['token']for x in C.read(d/'TRACE.json')['checkpoints']]==[0,31,127,511],'Full actual checkpoint roster differs')
        elif u['kind']=='nonAIME_free_generation_fresh':
            C.require(r['status']=='COMPLETE'and len(r['generated_token_ids'])==r['generated_token_count']and 0<r['generated_token_count']<=64 and r['quantizer_audit']['full_actual_values_and_bytes_each_next']is True,'Actual free64 sampler/QDQ result invalid')
            if u['repeat']==1:
                old=C.read(root/'execution'/u['reference_unit']/'RESULT.json')['result']
                for k in ['generated_token_ids','input_ids_sha256','initial_generator_sha256','final_generator_sha256','finish_reason']:C.require(old[k]==r[k],'Fresh free64 sequence differs')
        elif u['kind']=='nonAIME_throughput':
            C.require(set(r['conditions'])==CONDITIONS and all(x['warmup_tokens']==128 and x['measured_tokens']==256 and x['TPS']>0 and x['observer']is False for x in r['conditions'].values()),'Real throughput roster differs')
        elif u['kind']=='real_model_memory_pressure':C.require(len(r['cases'])==4 and all(x['total_context']==82293 and x['actual_short_QDQ_tokens']==16 for x in r['cases'])and r['numerical_81K_proof']is False,'Actual context pressure scope differs')
        identities.append(w['identity']);checks.append(dict(unit=u['id'],payload_sha256=saved,actual_exit=0,release='PASS'))
    C.require(all(not C.same_process(i)for i in identities),'Qualified own worker still running')
    obs=C.query_gpu();C.require(not any(x['pid']in{i['pid']for i in identities}for x in obs['apps']),'Qualified own GPU not released')
    C.require(ready['source_freeze_sha256']==C.sha(root/'SOURCE_FREEZE.json')and ready['permit_sha256']==C.sha(root/'GPU_PERMIT.json')and ready['qualification_sha256']==C.sha(root/'GROUP64_GPU_CODEC_QUALIFICATION.json'),'Qualification-ready receipt changed')
    return dict(node_id=C.NODE['id'],host=C.NODE['host'],gpu_uuid=C.NODE['uuid'],source_sha256=C.sha(root/'SOURCE_FREEZE.json'),permit_sha256=C.sha(root/'GPU_PERMIT.json'),qualification_sha256=C.sha(root/'GROUP64_GPU_CODEC_QUALIFICATION.json'),model_bytes_identity_sha256=model_bytes_identity(root),qualification_count=19,all19_actual_exit0_release_PASS=True,own_worker_processes_and_UUID_released=True,checks=checks)

def validate_formal_record(wrapper,job,root,node):
    """Original formal validator expressions with explicit source-node UUID binding."""
    row=wrapper['result'];C.require(wrapper['status']=='PASS'and wrapper['unit']==job['id'],'Worker result/unit not PASS')
    copy=dict(wrapper);saved=copy.pop('payload_sha256');C.require(C.objsha(copy)==saved,'Worker payload SHA differs')
    C.require(wrapper['source_freeze_sha256']==C.sha(root/'SOURCE_FREEZE.json')and wrapper['permit_sha256']==C.sha(root/'GPU_PERMIT.json'),'Worker source/permit differs')
    C.require(row['status']=='COMPLETE'and not row['nonfinite']and row['runtime_error']is None,'Technical/incomplete generation')
    C.require(row['condition']==job['condition']and row['problem_id']==job['problem_id']and row['seed']==1 and row['gpu']==job['gpu']==node['physical_gpu']and row['gpu_uuid']==job['gpu_uuid']==node['uuid']and job['node_id']==node['id'],'Configuration/question/node/GPU binding differs')
    C.require(row['input_ids_sha256']==job['input_ids_sha256']and row['input_tokens']==job['input_tokens'],'Frozen actual prompt differs')
    ts=row['generated_token_ids'];n=len(ts);C.require(0<n<=81920 and all(type(t)is int and 0<=t<248320 for t in ts)and n==row['generated_token_count']and C.objsha(ts)==row['generated_token_ids_sha256'],'Token payload/range/hash differs')
    eos=ts[-1]in row['stop_token_ids'];C.require(not any(t in row['stop_token_ids']for t in ts[:-1])and row['finish_reason']==('eos'if eos else'length')and(eos or n==81920),'Illegal EOS/length termination')
    total=row['input_tokens']+n-1;q=row['quantizer_audit'];C.require(q['group_size']==64 and q['group_axis']=='Key'and q['condition']==job['condition'],'Group64 codec identity differs')
    C.require(row['actual_model_forward_calls']==total and q['tokens']==total and q['qdq_layer_updates']==24*total and row['prompt_layer_QDQ']==24*row['input_tokens']and row['decode_layer_QDQ']==24*(n-1),'Prompt/token0/decode QDQ count differs')
    C.require(len(row['writebacks_by_layer'])==24 and all(v==total for v in row['writebacks_by_layer'].values())and row['sampler_calls']==n,'Layer/sampling coverage incomplete')
    expected=24*sum(t>0 and(t in range(1,128)or t%2048==0)for t in range(total));C.require(q['next_consumption_checks']==expected,'Sparse next-consumption roster differs')
    C.require(row['sampler_source_sha256']==C.sha(root/'protocol/sampler.py')and row['scorer_source_sha256']==C.sha(root/'protocol/aime26_scorer_v4.py')and row['dataset_freeze_sha256']==C.sha(root/'AIME26_DATASET_FREEZE.json'),'Sampler/scorer/input identity differs')
    return row
