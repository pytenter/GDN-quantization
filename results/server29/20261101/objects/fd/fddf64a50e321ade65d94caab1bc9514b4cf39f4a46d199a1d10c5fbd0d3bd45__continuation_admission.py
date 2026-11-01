"""Strict provenance for human-authorized resource continuation; no dispatch API."""
from pathlib import Path
import execution_control as C

OLD_ROOT=Path('/data/zypan/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1/revisions/cross_host_v4')
OLD_ARCHIVE=Path('/data/zypan/qwen_group64_four_codec_aime26_archive/revisions/QWEN_GDN_GROUP64_FOUR_CODEC_AIME26_SEED1_V1/cross_host_v4')
V6_ROOT=Path('/data/zypan/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1/revisions/three_gpu_ledger_repair_v6')
V6_ARCHIVE=Path('/data/zypan/qwen_group64_four_codec_aime26_archive/revisions/QWEN_GDN_GROUP64_FOUR_CODEC_AIME26_SEED1_V1/three_gpu_ledger_repair_v6')
AUTH_SHA='ef8e6fcc185c6df73d87be381c2a2dda4c94e9e1ac43268c10b7a63895514931'
RESUME_SHA='d2915c96d6bd1ca088d47c6f0a8650ce1c3c3c7735dd0ee9eb3a328640a3a0a5'
EXPECTED={
 'server55_gpu0':(OLD_ROOT,'202.38.247.55',0,'GPU-a78176ed-f97f-c54f-6201-5a01b570f422','77696a7200aa3454265a07a0c8acf252dee4de414072663af1e7d2be7ce3d49e','dd207195328fd967154b7db9ed3c86c25c59bd3cdedb62783d02e0d311b73eff'),
 'server29_gpu6':(OLD_ROOT,'202.38.247.29',6,'GPU-de53f243-b271-521c-d875-d96611ed8574','c755babd1730dd0f042d35e7d55857d0434257d2cf193ab0fa4efab8052e0e17','1844ad896668d15e5351625739f348015d609cf6797a64fa20a227d643646a79'),
 'server29_gpu5':(V6_ROOT,'202.38.247.29',5,'GPU-07d9c381-888d-1111-a944-56abcc583573','84da8bc7eb7b81ed391989fc6cc4c5e6df2ccb169082b236cce4dee9cd68e215','f4bb4f1f518626529963285aecc4d4a38796dde334c2c365014886322f042954')}

def binding(root):
 p=C.read(Path(root)/'REUSE_ADMISSION_STATIC.json')
 C.require(p['human_authorization_sha256']==AUTH_SHA and p['resume_authorization_sha256']==RESUME_SHA,'Exact expansion/resume authorization differs')
 C.require(set(p['origins'])==set(EXPECTED),'Exactly three authorized qualification origins required')
 for name,(path,host,gpu,uuid,source,permit)in EXPECTED.items():
  x=p['origins'][name];n=x['node']
  C.require(x['root']==str(path)and(n['id'],n['host'],n['physical_gpu'],n['uuid'])==(name,host,gpu,uuid)and x['source_sha256']==source and x['permit_sha256']==permit,'Original root/node/UUID/source/permit binding differs')
 C.require(len(p['qualification'])==57 and all(sum(x['origin_node']==n for x in p['qualification'].values())==19 for n in EXPECTED),'Exactly57 original per-device qualifications required')
 C.require(set(p['formal'])=={'aime26_29_G64-1_seed1'},'Exactly one original complete formal payload must be retained')
 return p

def origin_root(root,origin_node):
 p=binding(root);n=p['origins'][origin_node];path=Path(n['root'])
 if n['node']['host']==C.NODE['host']:source=path
 else:
  C.require(C.NODE['host']=='202.38.247.29'and origin_node=='server55_gpu0'and path==OLD_ROOT,'Unknown cross-host provenance source')
  source=OLD_ROOT/'peer55'
 C.require(C.sha(source/'SOURCE_FREEZE.json')==n['source_sha256']and C.sha(source/'GPU_PERMIT.json')==n['permit_sha256'],'Original source/permit bytes changed')
 return source

def qualification_source(root,unit_id):
 root=Path(root);p=binding(root)
 C.require(unit_id in {u['id']for u in C.read(root/'GPU_EXECUTION_PLAN.json')['qualification_units']},'Unknown qualification ID')
 C.require(unit_id in p['qualification'],'No fresh qualification is authorized in this continuation')
 x=p['qualification'][unit_id];source=origin_root(root,x['origin_node']);d=source/'execution'/unit_id
 C.require(C.sha(d/'RESULT.json')==x['result_sha256']and C.sha(d/'RESOURCE_INTERVAL.json')==x['interval_sha256'],'Reused qualification result/interval bytes changed')
 if x.get('trace_sha256')is not None:C.require(C.sha(d/'TRACE.json')==x['trace_sha256'],'Original qualification trace changed')
 return source

def host_reused_formal(root):
 p=binding(root)
 return {k:v for k,v in p['formal'].items()if v['job']['compute_host']==C.NODE['host']}

def formal_source(root,job):
 root=Path(root);p=binding(root);x=p['formal'].get(job['id'])
 if x is not None:
  C.require(job==x['job']and x['origin_node']=='server29_gpu6','Reused formal job/UUID changed')
  source=origin_root(root,x['origin_node']);d=source/'formal'/job['id']
  C.require(C.sha(d/'RESULT.json')==x['result_sha256']and C.sha(d/'RESOURCE_INTERVAL.json')==x['interval_sha256'],'Original complete formal bytes changed')
  return source
 return root if job['compute_host']==C.NODE['host']else root/'peer55'

def admit_local_reuse(root):
 root=Path(root);p=binding(root)
 C.require(C.sha(root/'HUMAN_RESUME_AUTHORIZATION.json')==RESUME_SHA and C.sha(root/'HUMAN_THREE_GPU_CONTINUATION_AUTHORIZATION.json')==AUTH_SHA,'Human authorization raw bytes changed')
 local={n['id']for n in C.HOST_NODES};sources={};ids=[];qualification=[]
 for name in local:
  source=origin_root(root,name);sources[name]=source;C.verify_sources(source);C.model_guard(source)
  if name=='server29_gpu5':
   C.require(C.read(source/'SESSION_RESULT.json')['status']=='FAIL_CLOSED_STOP','The original v6 terminal must remain a real STOP')
   C.require(C.sha(source/'GLOBAL_FORMAL_ADMISSION.json')==p['v6_global_admission_sha256'],'Original v6 real57 admission changed')
  else:
   stop=C.read(source/'GLOBAL_STOP_REQUEST.json')
   C.require(stop['status']=='USER_REQUESTED_CONTROLLED_STOP_FOR_THREE_GPU_CONTINUATION'and stop['human_authorization_sha256']==AUTH_SHA,'Original authorized v4 scheduling STOP changed')
  C.require(not C.same_process(C.read(source/'GPU_RUN_STARTED.json')['identity']),'Original controller still alive')
 if C.NODE['host']=='202.38.247.29':
  C.require(C.sha(OLD_ARCHIVE/'FINAL_SEAL_RECEIPT.json')==p['archive_seal_sha256']and C.sha(V6_ARCHIVE/'FINAL_SEAL_RECEIPT.json')==p['v6_archive_seal_sha256'],'Original v4/v6 archival seal changed')
 plan=C.read(root/'GPU_EXECUTION_PLAN.json')['qualification_units']
 for unit in plan:
  x=p['qualification'][unit['id']]
  C.require(unit['node_id']in local and x['origin_node']==unit['node_id']and unit==x['unit'],'Qualification origin/assignment modified or moved')
  source=qualification_source(root,unit['id']);d=source/'execution'/unit['id'];w=C.read(d/'RESULT.json');i=C.read(d/'RESOURCE_INTERVAL.json');v=dict(w);saved=v.pop('payload_sha256')
  C.require(C.objsha(v)==saved==x['payload_sha256']and w['status']=='PASS'and w['unit']==unit['id']and w['source_freeze_sha256']==C.sha(source/'SOURCE_FREEZE.json')and w['permit_sha256']==C.sha(source/'GPU_PERMIT.json'),'Original qualification payload identity changed')
  C.require(i['identity']==w['identity']==C.read(d/'STARTED.json')['identity']and i['exit_returncode']==0 and i['release']=='PASS'and w['gpu']==unit['gpu']and w['gpu_uuid']==C.UUIDS[unit['gpu']],'Original qualification actual exit/release/UUID differs')
  C.require(not(root/'execution'/unit['id']/'DISPATCH.json').exists(),'Reused qualification was redispatched')
  ids.append(i['identity']);qualification.append(unit['id'])
 C.require(len(qualification)==len(plan)==19*len(C.UUIDS),'All local original qualifications must be reused')
 jobs={j['id']:j for j in C.read(root/'FORMAL_JOB_MANIFEST.json')['jobs']};formal=[]
 from cross_host_contract import validate_formal_record
 nodes={n['id']:n for n in C.read(root/'CROSS_HOST_PLAN.json')['nodes']}
 for unit,x in host_reused_formal(root).items():
  C.require(unit in jobs and jobs[unit]==x['job'],'Original formal assignment changed')
  source=formal_source(root,jobs[unit]);d=source/'formal'/unit;w=C.read(d/'RESULT.json');validate_formal_record(w,jobs[unit],source,nodes[jobs[unit]['node_id']]);i=C.read(d/'RESOURCE_INTERVAL.json')
  C.require(i['identity']==w['identity']and i['exit_returncode']==0 and i['release']=='PASS','Only complete actual exited/released formal results are reusable')
  C.require(not(root/'formal'/unit/'DISPATCH.json').exists(),'Original formal was redispatched')
  ids.append(i['identity']);formal.append(unit)
 C.require(all(not C.same_process(i)for i in ids),'Original admitted process still alive')
 obs=C.query_gpu();C.require(not any(a['pid']in{i['pid']for i in ids}for a in obs['apps']),'Original admitted process still holds a GPU')
 return dict(status='PASS_STRICT_V4_38_V6_GPU5_19_AND_ONE_ORIGINAL_FORMAL_REUSE',qualification_ids=qualification,formal_ids=formal,original_roots={k:str(v)for k,v in sources.items()},original_identities=ids,new_fresh_qualification_count=0,no_prefix_splicing=True)
