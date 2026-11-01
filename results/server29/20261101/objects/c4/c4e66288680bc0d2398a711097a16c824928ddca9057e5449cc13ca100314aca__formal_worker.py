"""Single-GPU, single-model full persistent trajectories; never classifies outcomes."""
import ast,csv,gc,json,math,os,shutil,sys,time,traceback
from pathlib import Path
import pilot_engineering as P
import anchor_qualification as A

POSITIONS=list(range(8192,32768,64))
ARMS=['PARENT','NO_SCALE_INFLATION','NO_SCALE_DEFLATION']
ENDPOINTS=[8192,16384,24576,32768]
LEGACY_REDUCER_SOURCE=Path('/data/zypan/worktrees/qwen-gdn-fd-scale-feedback-direction-v1/experiments/QWEN_GDN_FD_SCALE_FEEDBACK_DIRECTION_V1/scripts/direction_replay.py')

def legacy_reducer():
    module=P.N.module(LEGACY_REDUCER_SOURCE,'functional_pilot_unchanged_32K_signed_reducer')
    return module.SignedStreamingReducer(P.torch)

def event(name,obj):
    path=P.ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(obj,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def qparam_code():
    tree=ast.parse(P.N.FORMAL.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='quantize_c128')
    body=next(n for n in fn.body if isinstance(n,ast.For)).body
    nodes=[n for n in body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['source','scale']]
    P.check([n.targets[0].id for n in nodes]==['source','scale'],'exact original source/scale AST')
    return compile(ast.Module(body=nodes,type_ignores=[]),str(P.N.FORMAL),'exec')

SCALE_CODE=qparam_code()
def scale(state):
    ns={'torch':P.torch,'EPS':1e-12,'state':state};exec(SCALE_CODE,ns)
    s=ns['scale'];P.check(s.dtype==P.torch.float32 and list(s.shape)==[1,32,1,128] and bool(P.torch.isfinite(s).all()) and bool((s>0).all()),'current positive finiteFP32 C128 scale')
    return s

def load_spec(doc):
    P.check(not (P.ROOT/'FIRST_FAILURE_GATE.json').exists(),'existing STOP; no subsequent unit')
    spec=json.loads((P.ROOT/'FROZEN_SCIENCE_SPEC.json').read_text())
    P.check(spec['status']=='FROZEN_ALL_GATES_PASS','all science gates')
    import subprocess
    P.check(subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.REPO,text=True).strip()==spec['code_commit'],'frozen production code commit')
    for path,wanted in spec['source_SHA'].items():P.check(P.sha(path)==wanted,'frozen production source '+path)
    for row in spec['model_files']:
        stat=Path(row['path']).stat()
        P.check(stat.st_size==row['bytes'] and stat.st_mtime_ns==row['mtime_ns'],'paired sealed model checkpoint stat identity')
    d=spec['documents'][doc-1]
    P.check(P.sha(d['token_path'])==d['token_SHA'] and P.sha(d['text_path'])==d['raw_text_SHA'],'formal document SHA')
    ids=json.loads(Path(d['token_path']).read_text());P.check(len(ids)==32769,'formal32769 input')
    return spec,d,ids

def identity(spec,d,reference=None,anchor=None,arm=None):
    return {'document_id':d['document_id'],'document_raw_SHA':d['raw_text_SHA'],'document_token_SHA':d['token_SHA'],
        'model_SHA':spec['model_SHA'],'C6R_rotation_SHA':spec['C6R_rotation_SHA'],'quantizer_SHA':spec['quantizer_SHA'],
        'code_commit':spec['code_commit'],'runtime_identity':spec['runtime_identity'],'GPU_UUID':spec['GPU_UUID'],
        'science_spec_SHA':P.sha(P.ROOT/'FROZEN_SCIENCE_SPEC.json'),'FP_reference_identity':reference,
        'anchor_checkpoint_identity':anchor,'intervention_policy_identity':P.digest(spec['intervention']),
        'position_schedule_identity':P.digest(POSITIONS),'condition':arm}

def write_csv(name,rows):
    P.check(bool(rows),'no empty scienceCSV')
    path=P.ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.csv.tmp');P.check(not path.exists() and not tmp.exists(),'write-onceCSV')
    fields=list(rows[0]);P.check(all(set(r)==set(fields) for r in rows),'uniformCSV schema')
    with tmp.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    os.rename(tmp,path)
    return {'path':str(path),'sha256':P.sha(path),'rows':len(rows)}

def progress(doc,phase,cache,p,started):
    record={'utc':P.N.now(),'document_index':doc,'phase':phase,**A.memory(cache,p+1),'seconds':time.monotonic()-started}
    event(f'execution/doc_{doc:02d}_{phase}_progress.jsonl',record)
    print('TRAJECTORY_PROGRESS',doc,phase,p+1,json.dumps(record,sort_keys=True),flush=True)

def reference(doc):
    spec,d,ids=load_spec(doc);F,core,_,model,tokenizer,started=P.setup()
    P.check(tokenizer(Path(d['text_path']).read_text(),add_special_tokens=False).input_ids[:32769]==ids,'formal FP text/token exact pairing')
    h=P.Harness(model,F,core,P.torch);cache=None;scales=[];chunks=[];logits=[];states={};chunk_start=8192
    with P.torch.inference_mode():
        for p in range(32768):
            out,cache=h.forward(ids[p],cache,p,mode='CLEAN_FP_REFERENCE')
            if p>=8192:
                scales.append(P.torch.stack([scale(F.get_state(cache,l)).detach().cpu() for l in P.S.LAYERS]))
                if len(scales)==256 or p==32767:
                    payload={'first_token_index':chunk_start,'last_token_index_exclusive':p+1,'layers':list(P.S.LAYERS),
                             'scales':P.torch.stack(scales),'dtype':'FP32_lossless','coordinate':'direct_C6R_FP_pre_QDQ_equivalent'}
                    needed=payload['scales'].numel()*4+1024**2
                    P.check(shutil.disk_usage(P.ROOT).free>=needed,'physical disk insufficient for atomic current clean-scale chunk')
                    art=A.pt_save(f'references/doc_{doc:02d}/scales/token_{chunk_start:05d}_{p+1:05d}.pt',payload)
                    chunks.append(dict(art,first_token_index=chunk_start,last_token_index_exclusive=p+1));scales=[];chunk_start=p+1
            if p in POSITIONS:logits.append(out.logits.detach().float().cpu().reshape(-1))
            if p+1 in ENDPOINTS:
                states[p+1]={l:F.get_state(cache,l).detach().cpu().clone() for l in P.S.LAYERS}
            if (p+1)%256==0:progress(doc,'CLEAN_FP_REFERENCE',cache,p,started)
        P.check(len(logits)==384 and len(chunks)==96,'complete frozen FP reference schedules')
        log_art=A.pt_save(f'references/doc_{doc:02d}/sampled_FP32_logits.pt',{'positions':POSITIONS,'logits':P.torch.stack(logits),
            'vocabulary_size':248320,'logits_position':'after consuming ids[t], predict ids[t+1]'})
        state_art=A.pt_save(f'references/doc_{doc:02d}/clean_endpoint_states.pt',states)
        ident=identity(spec,d,arm='CLEAN_FP_REFERENCE')
        reference_id=P.digest({'identity':ident,'chunks':chunks,'logits':log_art,'states':state_art})
        P.save(f'references/doc_{doc:02d}/FP_REFERENCE_MANIFEST.json',{'status':'COMPLETE','identity':ident,
            'reference_identity':reference_id,'consumed':32768,'QDQ_writebacks':0,'chunks':chunks,'logits':log_art,'endpoint_states':state_art,
            'scale_shape_per_token':[24,1,32,1,128],'scale_dtype':'torch.float32','scale_observation':'same C6R forward, no inverse approximation or quantization',**A.receipt(started)})
    h.close();print('UNIT_COMPLETE',doc,'CLEAN_FP_REFERENCE',flush=True)

def warmup(doc):
    spec,d,ids=load_spec(doc);ref=json.loads((P.ROOT/f'references/doc_{doc:02d}/FP_REFERENCE_MANIFEST.json').read_text())
    P.check(ref['status']=='COMPLETE','paired FP reference complete before student')
    F,core,_,model,tokenizer,started=P.setup();h=P.Harness(model,F,core,P.torch);cache=None
    with P.torch.inference_mode():
        for p in range(8192):
            out,cache=h.forward(ids[p],cache,p)
            if (p+1)%256==0:progress(doc,'COMMON_PARENT_WARMUP',cache,p,started)
        fp=P.fingerprint(cache,model);payload=P.snapshot(cache,model,8192)
        P.check(P.S.snapshot_hash(payload['cache'])==fp['cache_tensors'],'formal complete anchor snapshot byte identity')
        payload['identity']=identity(spec,d,reference=ref['reference_identity'],arm='COMMON_PARENT_WARMUP')
        artifact=A.pt_save(f'anchors/doc_{doc:02d}/common_student_08192.pt',payload)
        anchor_id=P.digest({'artifact':artifact,'fingerprint':fp,'identity':payload['identity']})
        P.save(f'anchors/doc_{doc:02d}/ANCHOR_MANIFEST.json',{'status':'COMPLETE','artifact':artifact,'anchor_identity':anchor_id,
            'consumed':8192,'fingerprint':fp,'identity':payload['identity'],'prefix_runs':1,
            'persistent_every_token_QDQ_calls':8192,'clean_FP_state_resets':0,**A.receipt(started)})
    h.close();print('UNIT_COMPLETE',doc,'COMMON_PARENT_WARMUP',flush=True)

class ScaleReader:
    def __init__(self,manifest):self.rows=manifest['chunks'];self.current=None;self.index=0
    def get(self,p):
        if self.current is None or not self.current['first_token_index']<=p<self.current['last_token_index_exclusive']:
            row=self.rows[self.index];self.index+=1
            P.check(row['first_token_index']<=p<row['last_token_index_exclusive'],'clean cache exact current-token index')
            path=Path(row['path']);P.check(P.sha(path)==row['sha256'],'atomic FP32 scale chunk SHA')
            self.current=P.torch.load(path,map_location='cpu',weights_only=True)
            P.check(self.current['first_token_index']==row['first_token_index'] and self.current['layers']==list(P.S.LAYERS),'clean scale current layer/token metadata')
            P.check(self.current['scales'].dtype==P.torch.float32,'lossless clean scale noFP16')
        return self.current['scales'][p-self.current['first_token_index']].to('cuda:0')

class Statistics:
    names=['groups','inflation_groups','deflation_groups','activated_groups','signed_log2_sum','abs_effective_log2_offset_sum',
           'clipped_values','saturated_codes','state_values']
    def __init__(self):self.rows={};self.pending={}
    def add(self,window,l,values):
        key=(window,l);self.pending.setdefault(key,[]).append(values)
    def flush(self):
        for key,vals in self.pending.items():
            old=self.rows.setdefault(key,[0.]*len(self.names))
            for i in range(len(self.names)):old[i]=math.fsum([old[i]]+[v[i] for v in vals])
            vals.clear()
    def result(self,ident):
        self.flush();rows=[]
        for (window,l),vals in sorted(self.rows.items()):
            r=dict(ident,window_start=8192+8192*window,window_end=16384+8192*window,layer=l,**dict(zip(self.names,vals)))
            r.update(activation_fraction=r['activated_groups']/r['groups'],mean_signed_log2_scale_ratio=r['signed_log2_sum']/r['groups'],
                mean_abs_effective_log2_offset=r['abs_effective_log2_offset_sum']/r['groups'],clipping_fraction=r['clipped_values']/r['state_values'],
                saturation_fraction=r['saturated_codes']/r['state_values'])
            rows.append(r)
        return rows

def statistics_values(source,adaptive,c,effective,codes):
    signed=P.torch.log2(adaptive.double()/c.double())
    shift=P.torch.log2(effective.double()/adaptive.double()).abs()
    values=P.torch.stack([P.torch.tensor(adaptive.numel(),device='cuda:0',dtype=P.torch.float64),
        (adaptive>c).sum().double(),(adaptive<c).sum().double(),(effective!=adaptive).sum().double(),
        signed.sum(),shift.sum(),(source.abs()>127*effective).sum().double(),(codes.abs()==127).sum().double(),
        P.torch.tensor(source.numel(),device='cuda:0',dtype=P.torch.float64)]).cpu().tolist()
    P.check(all(math.isfinite(v) for v in values),'finite streaming scale/clipping statistics')
    return values

def branch(doc,arm):
    P.check(arm in ARMS,'branch identity')
    spec,d,ids=load_spec(doc);ref=json.loads((P.ROOT/f'references/doc_{doc:02d}/FP_REFERENCE_MANIFEST.json').read_text())
    anchor=json.loads((P.ROOT/f'anchors/doc_{doc:02d}/ANCHOR_MANIFEST.json').read_text())
    P.check(ref['status']==anchor['status']=='COMPLETE','reference/common anchor complete')
    F,core,_,model,tokenizer,started=P.setup();h=P.Harness(model,F,core,P.torch)
    for art in [anchor['artifact'],ref['logits'],ref['endpoint_states']]:P.check(P.sha(art['path'])==art['sha256'],'paired artifact SHA')
    payload=P.torch.load(anchor['artifact']['path'],map_location='cpu',weights_only=False)
    cache=P.restore(payload,model);P.check(P.fingerprint(cache,model)==anchor['fingerprint'],'formal independent branch exact complete restore')
    source_storages={(t.device.type,t.untyped_storage().data_ptr()) for _,t in P.S.tree_tensors(payload['cache'])}
    branch_storages={(t.device.type,t.untyped_storage().data_ptr()) for _,t in P.S.tree_tensors(cache)}
    P.check(not source_storages & branch_storages,'no shared mutable CPU anchor/GPU branch storage')
    frozen_source=P.S.snapshot_hash(payload['cache']);reader=ScaleReader(ref)
    fp_logits=P.torch.load(ref['logits']['path'],map_location='cpu',weights_only=True)
    clean_states=P.torch.load(ref['endpoint_states']['path'],map_location='cpu',weights_only=True)
    P.check(fp_logits['positions']==POSITIONS,'FP logits schedule exact')
    ident=identity(spec,d,reference=ref['reference_identity'],anchor=anchor['anchor_identity'],arm=arm)
    sampled=[];sampled_logits=[];stats=Statistics();legacy=legacy_reducer();state_damage=[];previous={};seen=set()
    recurrent=h.qmod.torch_recurrent_gated_delta_rule
    def checked(query,key,value,*args,**kw):
        if previous:
            P.check(P.torch.equal(kw['initial_state'],previous[h.active]),'formal next-token consumes restored/QDQ branch state')
            seen.add(h.active)
        return recurrent(query,key,value,*args,**kw)
    h.qmod.torch_recurrent_gated_delta_rule=checked
    with P.torch.inference_mode():
        for p in range(8192,32768):
            clean=reader.get(p);audit=(p<8195 or p%256==0 or p==32767)
            previous={l:F.get_state(cache,l).detach().clone() for l in P.S.LAYERS} if audit else {}
            seen=set()
            out,cache=h.forward(ids[p],cache,p,mode=arm,clean=clean,capture=True)
            if audit:P.check(seen==set(P.S.LAYERS),'all24 actual previous branch states consumed')
            for i,l in enumerate(P.S.LAYERS):
                source=h.pre[l];adaptive=scale(source);c=clean[i]
                effective=adaptive if arm=='PARENT' else (P.torch.minimum(adaptive,c) if arm=='NO_SCALE_INFLATION' else P.torch.maximum(adaptive,c))
                codes=P.torch.round(source.float()/effective).clamp(-127,127)
                if audit:P.check(P.torch.equal(F.get_state(cache,l),codes*effective),'formal original code/round/clamp effective writeback')
                vals=statistics_values(source,adaptive,c,effective,codes)
                stats.add((p-8192)//8192,l,vals)
                legacy.add(arm,l,p+1,P.torch.log2(adaptive/c),effective!=adaptive,int(vals[6]),int(vals[8]))
            if p in POSITIONS:
                j=len(sampled);P.check(POSITIONS[j]==p,'FP/student no-off-by-one next-token logits pairing')
                sampled.append(dict(ident,token_index=p,consumed_tokens=p+1,predicted_token_index=p+1,
                    full_vocabulary_size=248320,KL_FP32=P.kl32(fp_logits['logits'][j].to('cuda:0'),out.logits)))
                sampled_logits.append(out.logits.detach().float().cpu().reshape(-1))
            if p+1 in ENDPOINTS:
                vals=[float(core.relative_state_mse(F.get_state(cache,l),clean_states[p+1][l].to('cuda:0'))) for l in P.S.LAYERS]
                P.check(all(math.isfinite(v) for v in vals),'finite same-C6R endpoint state relativeMSE')
                state_damage.append(dict(ident,consumed_tokens=p+1,relative_state_MSE=math.fsum(vals)/24,
                    metric_role='secondary sparse endpoint state damage; not Future-KL'))
            if (p+1)%256==0:stats.flush();legacy.flush();progress(doc,arm,cache,p,started)
        P.check(len(sampled)==384 and reader.index==96,'complete32768 persistent/student scoring/scale schedules')
        P.check(P.S.snapshot_hash(payload['cache'])==frozen_source,'formal anchor immutable after entire branch')
        row_stats=stats.result(ident)
        legacy_rows=legacy.summary()
        P.check(sum(r['count'] for r in legacy_rows)==int(math.fsum(r['groups'] for r in row_stats)),'unchanged32K reducer exact group-count coverage')
        P.check(sum(r['clamp_count'] for r in legacy_rows)==int(math.fsum(r['activated_groups'] for r in row_stats)),'unchanged32K reducer clamp-count parity')
        P.save(f'students/doc_{doc:02d}/{arm}/LEGACY_32K_SIGNED_REDUCER.json',{'status':'COMPLETE','identity':ident,
            'source':str(LEGACY_REDUCER_SOURCE),'source_SHA':P.sha(LEGACY_REDUCER_SOURCE),'rows':legacy_rows,
            'original_bins_preserved':True,'original_saturation_field_means':'source range overflow, separate post-round code saturation in newCSV'})
        student_logits_art=A.pt_save(f'students/doc_{doc:02d}/{arm}/sampled_student_FP32_logits.pt',
            {'positions':POSITIONS,'logits':P.torch.stack(sampled_logits),'identity':ident})
        csvs=[write_csv(f'students/doc_{doc:02d}/{arm}/SAMPLED_FULLVOCAB_FUTURE_KL.csv',sampled),
              write_csv(f'students/doc_{doc:02d}/{arm}/SCALE_INTERVENTION_ACTIVATION.csv',row_stats),
              write_csv(f'students/doc_{doc:02d}/{arm}/STATE_DAMAGE_ENDPOINTS.csv',state_damage)]
        kl_values=[r['KL_FP32'] for r in sampled]
        windows={str(8192+i*8192):math.fsum(kl_values[i*128:(i+1)*128])/128 for i in range(3)}
        summary={'status':'COMPLETE','identity':ident,'consumed':32768,'anchor_consumed':8192,'continuation_QDQ_calls':24576,
            'every_token_QDQ':True,'persistent_INT8_writeback':True,'closed_loop_inputs':'current independent branch hidden/q/k/v/gates/cache',
            'sample_count':384,'K':P.S.mean(kl_values),'window_K':windows,'artifacts':csvs,'sampled_logits':student_logits_art,'finite_all_scored_logits':True,
            'complete_source_anchor_immutable':True,'numerical_writeback_checks':'first3, every256, endpoint; every-token original frozen code',
            'activation_groups':int(math.fsum(r['activated_groups'] for r in row_stats)),
            'total_groups':int(math.fsum(r['groups'] for r in row_stats)),
            'clipped_values':int(math.fsum(r['clipped_values'] for r in row_stats)),
            'saturated_codes':int(math.fsum(r['saturated_codes'] for r in row_stats)),
            'state_values':int(math.fsum(r['state_values'] for r in row_stats)),**A.receipt(started)}
        P.save(f'students/doc_{doc:02d}/{arm}/BRANCH_MANIFEST.json',summary)
    h.qmod.torch_recurrent_gated_delta_rule=recurrent;h.close();print('UNIT_COMPLETE',doc,arm,flush=True)

if __name__=='__main__':
    begin=time.monotonic()
    try:
        phase=sys.argv[1];doc=int(sys.argv[2])
        if phase=='reference':reference(doc)
        elif phase=='warmup':warmup(doc)
        elif phase=='branch':branch(doc,sys.argv[3])
        else:raise ValueError('phase')
    except Exception as exc:
        details={'status':'FAIL_AND_STOP','stage':'FORMAL_UNIT','unit_arguments':sys.argv[1:],'utc':P.N.now(),
            'exception':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'no_retry':True,
            'unit_wall_seconds':time.monotonic()-begin,'unit_GPUh_including_initialization':(time.monotonic()-begin)/3600}
        if getattr(P,'torch',None) is not None and P.torch.cuda.is_initialized():
            details.update(peak_allocated=P.torch.cuda.max_memory_allocated(),peak_reserved=P.torch.cuda.max_memory_reserved())
        if not (P.ROOT/'FIRST_FAILURE_GATE.json').exists():P.save('FIRST_FAILURE_GATE.json',details)
        raise
