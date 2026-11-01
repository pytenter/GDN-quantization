"""CPU-only input/descriptor/lineage audit of SHA-verified partial baseline data.

No raw model/tensor is downloaded or evaluated. Descriptor consistency and
recorded hash lineage are distinct from fresh-run numerical equivalence.
"""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import math
import re
import struct

here=Path(__file__).resolve().parent
root=here.parents[1]
evidence=root/'delivery/remote_evidence'
fixture=json.loads((here/'FIXTURE_PREFIX256_IDENTITY.json').read_text())
ids=fixture['prefix256']
assert fixture['native_byteorder']=='little','Input SHA reconstruction requires explicit byte order'
catalog_path=root/'independent_delivery/REMOTE_SMALL_EVIDENCE_CATALOG.json'
catalog={x['path']:x for x in json.loads(catalog_path.read_text())['files']}
contract_path=evidence/'gpu3/analysis/STRICT_BASELINE_CONTRACT.json'
contract=json.loads(contract_path.read_text())
layers=contract['math_and_state']['GDN_layers']
attention=contract['math_and_state']['attention_layers']
sizes={'torch.float32':4,'torch.bfloat16':2,'torch.int64':8}
sha=lambda b:hashlib.sha256(b).hexdigest()
errors=[]
counts=collections.Counter()
cards=[]
def check(value,label,context):
    counts[label]+=1
    if not value:errors.append(dict(check=label,**context))

for card in range(4):
    key='gpu'+str(card)
    paths=sorted((evidence/key/'execution/B0_FP_Q0/numeric').glob('token_*.json'))
    previous={}
    rows=[]
    for ordinal,p in enumerate(paths):
        raw=p.read_bytes();rel=str(p.relative_to(evidence));row=catalog[rel]
        context=dict(card=key,path=rel,token=ordinal)
        check(len(raw)==row['bytes']and sha(raw)==row['sha256'],'download_bytes_SHA',context)
        j=json.loads(raw);t=j['token'];meta=j['metadata'];items={x['name']:x for x in j['tensors']}
        check(t==ordinal==meta['token'],'contiguous_token_identity',context)
        check(meta['input_id']==ids[t],'bound_input_id',context)
        check(meta['next_input_id']==(ids[t+1]if t+1<256 else None),'bound_next_input_id',context)
        check(meta['cache_cursor']==t+1,'cache_cursor',context)
        check(meta['codec']=='FP'and meta['GDN_layers']==layers,'codec_GDN_layer_identity',context)
        check(j['role']=='FULL_REFERENCE'and j['actual_bytes_compared']is False and j['byte_equal']is True,'reference_only_disclosure',context)
        check(items['input_ids']['sha256']==sha(struct.pack('<q',ids[t])),'input_tensor_descriptor_matches_ID_bytes',context)
        check(len(items)==len(j['tensors']),'unique_tensor_names',context)
        expect={'input_ids':('torch.int64',[1,1]),'logits':('torch.bfloat16',[1,1,248320])}
        for layer in layers:
            for name in ('q','k','v','readout'):expect[f'GDN/{layer}/{name}']=('torch.bfloat16',[1,1,32,128])
            expect[f'GDN/{layer}/kwargs/beta']=('torch.bfloat16',[1,1,32])
            expect[f'GDN/{layer}/kwargs/g']=('torch.float32',[1,1,32])
            expect[f'GDN/{layer}/updated_state']=('torch.float32',[1,32,128,128])
            if t>0:expect[f'GDN/{layer}/kwargs/initial_state']=('torch.float32',[1,32,128,128])
            expect[f'CACHE/layers/{layer}/conv_states/0']=('torch.bfloat16',[1,8192,4])
            expect[f'CACHE/layers/{layer}/recurrent_states/0']=('torch.float32',[1,32,128,128])
        for layer in attention:
            for name in ('keys','values'):expect[f'CACHE/layers/{layer}/{name}']=('torch.bfloat16',[1,4,t+1,256])
        check(set(items)==set(expect),'complete_expected_tensor_names',context)
        for name,(dtype,shape)in expect.items():
            item=items.get(name,{})
            c=dict(context,tensor=name)
            check(item.get('dtype')==dtype and item.get('shape')==shape,'tensor_dtype_shape',c)
            check(item.get('bytes')==math.prod(shape)*sizes[dtype],'tensor_descriptor_byte_count',c)
            check(bool(re.fullmatch('[0-9a-f]{64}',item.get('sha256',''))),'tensor_recorded_SHA_syntax',c)
        deep=meta['deep_cache'];deep_layers=deep['attributes']['layers']
        check(len(deep_layers)==32,'deep_cache_32_layers',context)
        for layer in layers:
            attr=deep_layers[layer]['attributes'];state=items[f'CACHE/layers/{layer}/recurrent_states/0'];updated=items[f'GDN/{layer}/updated_state']
            c=dict(context,layer=layer)
            check(deep_layers[layer]['class_name']=='transformers.cache_utils.LinearAttentionLayer','deep_GDN_cache_class',c)
            for stem,name in [('conv_states','conv_states/0'),('recurrent_states','recurrent_states/0')]:
                desc=attr[stem]['0'];item=items[f'CACHE/layers/{layer}/{name}']
                check(all(desc[k]==item[k]for k in ('dtype','shape','sha256')),'deep_cache_tensor_descriptor_agreement',dict(c,attribute=stem))
            check(state['sha256']==updated['sha256'],'current_updated_state_equals_current_cache_recorded_SHA',c)
            if t>0:check(items[f'GDN/{layer}/kwargs/initial_state']['sha256']==previous[layer],'next_initial_state_equals_previous_cache_recorded_SHA',c)
            previous[layer]=state['sha256']
        for layer in attention:
            attr=deep_layers[layer]['attributes'];c=dict(context,layer=layer)
            check(deep_layers[layer]['class_name']=='transformers.cache_utils.DynamicLayer','deep_attention_cache_class',c)
            for name in ('keys','values'):
                desc=attr[name];item=items[f'CACHE/layers/{layer}/{name}']
                check(all(desc[k]==item[k]for k in ('dtype','shape','sha256')),'deep_attention_tensor_descriptor_agreement',dict(c,attribute=name))
        rows.append(dict(token=t,metadata_file=rel,metadata_SHA256=sha(raw),input_id=meta['input_id'],next_input_id=meta['next_input_id'],cache_cursor=meta['cache_cursor'],tensor_descriptors=len(items)))
    cards.append(dict(card=key,tokens=len(paths),range=[0,len(paths)-1]if paths else None,strict_byte_comparison='NOT_RUN_REFERENCE_ONLY',rows=rows))

out=dict(status='PASS_PARTIAL_INPUT_AND_DESCRIPTOR_LINEAGE_NOT_NUMERICAL_EQUIVALENCE'if not errors else 'FAIL_PARTIAL_METADATA_AUDIT',
         observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         task_id=contract['task_id'],execution_run_id=contract['execution_run_id'],
         fixture_prefix_receipt_SHA256=sha((here/'FIXTURE_PREFIX256_IDENTITY.json').read_bytes()),
         small_evidence_catalog_SHA256=sha(catalog_path.read_bytes()),
         bound_contract_SHA256=sha(contract_path.read_bytes()),
         frozen_worker_SHA256=sha((evidence/'gpu3/scripts/cublas_worker.py').read_bytes()),
         input_and_next_input_rule='Frozen worker lines254,270; next_input_id None at t255 even though full fixture is longer',
         dtype_shape_source='Frozen model architecture/contract and native GDN/cache call signatures; no torch import',
         total_token_receipts=sum(x['tokens']for x in cards),
         descriptor_shape_and_dtype_checks=counts['tensor_dtype_shape'],
         current_state_recorded_SHA_checks=counts['current_updated_state_equals_current_cache_recorded_SHA'],
         next_state_recorded_SHA_lineage_checks=counts['next_initial_state_equals_previous_cache_recorded_SHA'],
         check_counts=dict(counts),cards=cards,errors=errors,
         limitations=['Only already committed B0_FP_Q0 self-reference metadata is audited.',
                     'Recorded tensor SHA agreement is metadata lineage; current remote raw stream verification is separately owned by archive audit.',
                     'No B0 second fresh process, B1, performance, H_C128 or Dual-M result exists from these records.',
                     'No full qualification PASS or future long-context numerical bound is inferred.'],
         GPU_imports=False,GPU_queries=False,GPU_dispatches=False)
target=here/'PARTIAL_INPUT_GDN_DESCRIPTOR_AUDIT.json'
with target.open('x')as f:json.dump(out,f,sort_keys=True,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(status=out['status'],total=out['total_token_receipts'],errors=len(errors),path=str(target),sha256=sha(target.read_bytes()))))
