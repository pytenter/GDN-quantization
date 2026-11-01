import pathlib,json,hashlib,sys,ast,os,struct
os.environ['CUDA_VISIBLE_DEVICES']=''
r=pathlib.Path('/data/zypan/analyses/qwen_gdn_aime26_stepquant6_vs_h_dualm_2seed_v1')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();canon=lambda x:hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest();ints=lambda x:hashlib.sha256(struct.pack('<'+str(len(x))+'q',*x)).hexdigest()
c=json.loads((r/'CALIBRATION_INPUTS.json').read_bytes().decode());p=r/c['segments_path'];assert sha(p)==c['segments_sha256'] and c['AIME_used'] is False
x=json.loads(p.open().readline());assert x['segment_index']==0 and len(x['input_ids'])==2048 and ints(x['input_ids'])==x['token_ids_sha256']
sys.path.insert(0,'/data/zypan/transformers-qwen35/src');from transformers import AutoTokenizer
import torch
assert not torch.cuda.is_initialized();tok=AutoTokenizer.from_pretrained('/data/zypan/modelscope_models/Qwen3.5-9B',trust_remote_code=True,local_files_only=True)
prefix_ids=x['input_ids'][:256];passage=tok.decode(prefix_ids)
instruction='Continue the following encyclopedia passage with a detailed, coherent discussion of at least 600 words. Preserve the language and topic of the passage.\n\n'
problem=instruction+passage
common=r/'protocol_recovery/snapshots/34a320d20a48cda8d0a2a194914f3fa04f8c32bf8c6c533f5f9916fc5d6f9df7_aime26_common.py';runner=r/'protocol_recovery/snapshots/9ff56d00b6b2dbaf9be3c3186bd30be8d3b9cc9d9b580a680316bcc8c01247dd_run_qwen_aime26_formal.py'
ns=dict(torch=torch,hashlib=hashlib,json=json)
for file,names in [(common,{'raw_messages','ids_sha256'}),(runner,{'render_input'})]:
 nodes=[n for n in ast.parse(file.read_text()).body if getattr(n,'name',None) in names];assert len(nodes)==len(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(file),'exec'),ns)
messages,ids=ns['render_input'](tok,problem)
o=dict(task_id='QWEN_GDN_AIME26_STEPQUANT6_VS_H_DUALM_2SEED_V1',status='FROZEN_NON_AIME_CPU_INPUT_IDENTITY_ONLY',canonical_index=-1,problem_idx=-1,problem_id='NON_AIME_WIKITEXT_CANARY_0',problem=problem,answer=None,seed=1,non_AIME=True,method_order=['STEPQUANT6','H_DUALM'],input_ids_hash=ns['ids_sha256'](ids[0].tolist()),input_tokens=int(ids.shape[-1]),raw_messages=messages,source=dict(calibration_inputs_path='CALIBRATION_INPUTS.json',calibration_inputs_SHA256=sha(r/'CALIBRATION_INPUTS.json'),dataset=c['dataset'],revision=c['revision'],parquet_sha256=c['parquet_sha256'],segments_path=c['segments_path'],segments_SHA256=c['segments_sha256'],segment_index=0,full_window_start=x['start'],full_window_tokens=2048,full_window_token_SHA256=x['token_ids_sha256'],full_window_SHA_definition='little-endian signed-int64 bytes',full_window_token_canonicalJSON_SHA256=canon(x['input_ids']),prefix_tokens=256,prefix_token_SHA256=ints(prefix_ids),prefix_token_SHA_definition='little-endian signed-int64 bytes',prefix_token_canonicalJSON_SHA256=canon(prefix_ids),prefix_decode_skip_special_tokens=False,prefix_decoded_utf8_SHA256=hashlib.sha256(passage.encode()).hexdigest()),problem_utf8_SHA256=hashlib.sha256(problem.encode()).hexdigest(),selection_rule='Fixed first official calibration window; first256tokens; exact predeclared instruction; no model output selection or replacement',CPU_tokenizer_only=True,model_forward_count=0,CUDA_initialized=torch.cuda.is_initialized(),max_new_tokens=128,EOS_policy='Respect same original stop IDs; earlierEOS is real qualificationSTOP; no forced continuation or reprompt')
q=r/'CANARY_INPUT.json'
with q.open('x') as f:json.dump(o,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'sha256':sha(q),'text':q.read_text()}))
