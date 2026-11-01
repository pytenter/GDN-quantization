"""Read-only remote fragment: inspect the two original score passes; do not score again."""
REMOTE = r'''
import collections,csv,math
def equal_number(a,b,label):
 req(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9),'Measured aggregate differs: '+label)
score_passes=[]
for index in (0,1):
 stem='FROZEN_V4_FRESH_PASS_'+str(index)
 score=read(r/(stem+'.json'));started=read(r/(stem+'.PROCESS_STARTED.json'));exited=read(r/(stem+'.PROCESS_EXIT.json'));dispatched=read(r/(stem+'.PROCESS_DISPATCH.json'));inner=read(r/(stem+'.STARTED.json'))
 req(score['status']=='PASS_FRESH_CPU_FROZEN_V4_EXACT_SAVED_PAYLOAD' and score['identity']==started['identity']==exited['identity']==inner['identity'],'Original scorer result/owning identity differs')
 req(exited['actual_exited'] and exited['identity_gone'] and type(exited['returncode']) is int and exited['returncode']==0 and exited['identity_error'] is None,'Original scorer actual exit not zero')
 for receipt in (started,exited,dispatched,inner):
  req(receipt['source_manifest_sha256']==P['source'] and receipt['permit_sha256']==P['permit'],'Original scorer source/permit differs')
 req(dispatched['CUDA_VISIBLE_DEVICES']==inner['CUDA_VISIBLE_DEVICES']=='' and score['identity']['cmdline']==dispatched['argv'],'Original scorer CPU/argv binding differs')
 identities.append(score['identity']);score_passes.append(score)
req(score_passes[0]['identity']!=score_passes[1]['identity'] and score_passes[0]['identity']['pid']!=score_passes[1]['identity']['pid'] and score_passes[0]['payload']==score_passes[1]['payload'],'Two original fresh scorer payloads differ')
payload=score_passes[0]['payload'];rows=payload['rows']
req(payload['source_manifest_sha256']==P['source'] and payload['science_freeze_sha256']==sha(r/'SCIENCE_FREEZE.json') and payload['scorer_sha256']==sha(r/'protocol/aime26_scorer_v4.py') and payload['dataset_sha256']==sha(r/'protocol/aime26_frozen.jsonl'),'Score payload frozen science binding differs')
req(len(rows)==60 and len({(x['problem_id'],x['condition']) for x in rows})==60,'Full score unique denominator differs')
raw_by={(j['problem_id'],j['condition']):(j,w,t) for j,w,t in pairs}
memory=read(r/'comparison/memory_accounting.json');req(memory['status']=='PASS_ACTUAL_PACKED_TENSOR_INVENTORIES' and len(memory['rows'])==60,'Memory inventory denominator/status differs')
mem_by={x['unit']:x for x in memory['rows']};req(len(mem_by)==60,'Duplicate memory inventory unit')
memory_totals=collections.Counter()
for row in rows:
 j,w,t=raw_by[row['problem_id'],row['condition']];result=w['result'];tokens=result['generated_token_ids']
 req(row['raw_result_path']==(t.relative_to(r)/'RESULT.json').as_posix() and row['raw_result_sha256']==sha(t/'RESULT.json') and row['evidence_source_manifest_sha256']==w['source_manifest_sha256'] and row['evidence_permit_sha256']==w['permit_sha256'],'Scored row original result/source binding differs')
 req(row['token_sha256']==result['generated_token_ids_sha256'] and row['generated_token_count']==len(tokens) and row['finish_reason']==result['finish_reason'] and row['prompt_hash']==result['prompt_text_sha256'] and type(row['correct']) is bool,'Scored row token/finish/prompt binding differs')
 for key in ('wall_seconds','TPS','peak_allocated_bytes','peak_reserved_bytes'):req(row[key]==result[key],'Scored row measured field differs: '+key)
 grams=[tuple(tokens[i:i+32]) for i in range(max(0,len(tokens)-31))];equal_number(row['repeated_32gram_fraction'],0 if not grams else 1-len(set(grams))/len(grams),'repeated_32gram')
 audit=result['quantizer_audit'];stores=list(audit['storage_by_layer'].values());inventory=[x for store in stores for x in store['inventory']];categories=collections.Counter()
 for x in inventory:
  name=x['name'];key=('state_scale_exponent' if name.startswith('base.') and name.endswith('.scale_exponents') else 'state_codes' if name.startswith('base.') and name.endswith('.codes') else 'state_scale' if name.startswith('base.') and name.endswith('.scales') else 'state_zero_point' if name.startswith('base.') and name.endswith('.zero_points') else 'buffer_keys' if '.key.' in name or name.endswith('raw_bf16_key') else 'buffer_values' if name.endswith('.value') else 'buffer_beta' if name.endswith('.beta') else 'buffer_log_decay' if name.endswith('.log_decay') else 'other_tensor')
  req(type(x['nbytes']) is int and x['nbytes']>=0,'Invalid actual tensor bytes');categories[key]+=x['nbytes']
  if key=='state_scale_exponent':req(x['dtype'] in ('torch.int8','int8'),'Actual state exponent dtype differs')
  if '.key.' in name:req(not name.endswith('.scale_exponents'),'Forbidden key exponent storage')
 measured=mem_by[j['id']];state_values=24*32*128*128;resident=sum(x['nbytes'] for x in inventory)+audit['native_marker_tensor_nbytes']
 req(measured['condition']==j['condition'] and measured['state_values']==state_values and measured['final_live_tensor_nbytes']==dict(categories) and measured['actual_state_exponent_nbytes']==categories['state_scale_exponent'] and 0<=categories['state_scale_exponent']<=state_values//32,'Actual memory categories/exponent differ')
 req(measured['final_resident_tensor_nbytes']==resident and measured['native_marker_nbytes']==audit['native_marker_tensor_nbytes'],'Actual resident memory differs')
 equal_number(measured['actual_base_bits_per_value'],8*sum(categories[k] for k in ('state_codes','state_scale','state_zero_point','state_scale_exponent'))/state_values,'actual_base_bpv')
 equal_number(measured['final_resident_tensor_bits_per_state_value'],resident*8/state_values,'resident_bpv')
 for key,field in [('peak_prefill_buffer_payload_nbytes','peak_prefill_buffer_nbytes'),('peak_decode_buffer_payload_nbytes','peak_decode_buffer_nbytes'),('decode_buffer_capacity_nbytes','decode_buffer_capacity_nbytes')]:req(measured[key]==sum(x[field] for x in stores),'Measured buffer bytes differ')
 for key in ('peak_allocated_bytes','peak_reserved_bytes'):req(measured[key]==result[key],'Memory peak differs')
 memory_totals.update(categories)
summaries={}
for condition,period in [('R2',8),('R3',1)]:
 arm=[x for x in rows if x['condition']==condition];summary=read(r/(condition+'_PACE48_DP'+str(period))/'summary.json');saved=[json.loads(x) for x in (r/(condition+'_PACE48_DP'+str(period))/'results.jsonl').read_text().splitlines()]
 req(len(arm)==summary['N']==30 and saved==arm and summary==verdict['summaries'][condition],'Per-condition exact score output differs')
 req(summary['correct']==sum(x['correct'] for x in arm) and summary['EOS']==sum(x['finish_reason']=='eos' for x in arm) and summary['length_capped']==sum(x['finish_reason']=='length' for x in arm) and summary['nonfinite']==summary['runtime_errors']==0,'Per-condition measured counts differ')
 for key,expected in [('accuracy',summary['correct']/30),('EOS_rate',summary['EOS']/30),('mean_tokens',math.fsum(x['generated_token_count'] for x in arm)/30),('total_runtime_seconds',math.fsum(x['wall_seconds'] for x in arm)),('mean_runtime_seconds',math.fsum(x['wall_seconds'] for x in arm)/30),('mean_repeated_32gram_fraction',math.fsum(x['repeated_32gram_fraction'] for x in arm)/30)]:equal_number(summary[key],expected,key)
 for key in ('peak_allocated_bytes','peak_reserved_bytes'):req(summary[key]==max(x[key] for x in arm),'Arm peak differs')
 summaries[condition]=summary
by={(x['problem_id'],x['condition']):x for x in rows};paired=read(r/'comparison/paired_summary.json');req(paired==verdict['comparison'] and paired['N_pairs']==30,'Paired summary denominator/binding differs')
with (r/'comparison/paired_results.csv').open(newline='') as f:paired_rows=list(csv.DictReader(f))
req(len(paired_rows)==30 and len({x['problem_id'] for x in paired_rows})==30,'Paired CSV unique denominator differs');categories=collections.Counter()
for x in paired_rows:
 a,b=by[x['problem_id'],'R2'],by[x['problem_id'],'R3'];category='both_correct' if a['correct'] and b['correct'] else 'R2_only_correct' if a['correct'] else 'R3_only_correct' if b['correct'] else 'both_wrong';categories[category]+=1
 req(x['category']==category and x['R2_correct']==str(a['correct']) and x['R3_correct']==str(b['correct']),'Paired score category differs')
 for prefix,entry in [('R2',a),('R3',b)]:
  req(int(x[prefix+'_tokens'])==entry['generated_token_count'] and x[prefix+'_finish']==entry['finish_reason'],'Paired token/finish differs');equal_number(float(x[prefix+'_seconds']),entry['wall_seconds'],'paired_seconds')
req(paired['categories']==dict(categories) and paired['correct_difference']==summaries['R2']['correct']-summaries['R3']['correct'],'Paired category aggregate differs');equal_number(paired['delta_accuracy'],summaries['R2']['accuracy']-summaries['R3']['accuracy'],'paired_delta')
discordant=categories['R2_only_correct']+categories['R3_only_correct'];p_value=1.0 if not discordant else min(1.0,2*sum(math.comb(discordant,k) for k in range(min(categories['R2_only_correct'],categories['R3_only_correct'])+1))/2**discordant);equal_number(paired['exact_McNemar_two_sided_p'],p_value,'McNemar')
accounting=read(r/'CONTINUATION_COMPLETION_ACCOUNTING.json');req(accounting==verdict['continuation_accounting'] and accounting['formal_total_unique']==60 and accounting['reused_formal_completed']==54 and accounting['current_formal_completed']==6 and accounting['original_uninterrupted_60'] is False,'Mixed execution completion accounting differs')
req(len(ledger['intervals'])==10 and sum(x['stage']=='formal' for x in ledger['intervals'])==6 and all(type(x['returncode']) is int and x['returncode']==0 and x['released'] for x in ledger['intervals']),'Current gates/formals actual lifecycle counts differ')
score_review=dict(status='PASS_TWO_ORIGINAL_FRESH_CPU_SCORES_ALL60_BINDINGS_AND_INDEPENDENT_AGGREGATE_MEMORY_RECALCULATION',score_pass_identities=[x['identity'] for x in score_passes],scorer_sha256=payload['scorer_sha256'],dataset_sha256=payload['dataset_sha256'],conditions={k:dict(N=v['N'],correct=v['correct'],accuracy=v['accuracy'],mean_tokens=v['mean_tokens'],EOS=v['EOS'],length_capped=v['length_capped'],mean_runtime_seconds=v['mean_runtime_seconds'],peak_allocated_bytes=v['peak_allocated_bytes'],peak_reserved_bytes=v['peak_reserved_bytes']) for k,v in summaries.items()},paired=paired,actual_memory_category_totals=dict(memory_totals),new_score_passes=0,GPU_calls=0,model_loads=0)
'''
