"""Prospective 32K-only confirmation. Never starts training or free generation."""
from inherit import *
import argparse, csv, hashlib, itertools, json, math, os, shutil, signal
import statistics, subprocess, time, traceback

def load(p): return json.loads(Path(p).read_text())
sha, row = c.sha, c.row

def emit(relative, value):
    path = Path(relative)
    path = path if path.is_absolute() else D / path
    assert path.is_relative_to(D), ('OUTPUT_OUTSIDE_NEW_NAMESPACE', str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write('\n')

def text_file(relative, value):
    path = D / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as out: out.write(value)

def command(args, cwd=R):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()

def inventory_files(root):
    return sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts)

def clean_holdout():
    verdict = load(PARENT / 'reports/FINAL_REPORT.json')['verdict']
    assert verdict['DISCOVERY_SURROGATE_GATE'] == 'FAIL'
    assert verdict['SELECTED_PROPAGATION_SURROGATE'] == 'NONE'
    assert verdict['CONFIRMATION_COMPLETE'] == '0/64'
    assert verdict['PRIMARY_NEXT_STEP_CATEGORY'] == 'STOP_PROPAGATION_OBJECTIVE_TRAINING'
    docs = [x for x in load(PARENT / 'protocol/frozen_documents.json')['streams'] if x['phase'] == 'CONFIRMATION']
    assert len(docs) == 4
    assert [x['selection_rank'] for x in docs] == [4, 5, 6, 7]
    assert [x['physical_gpu'] for x in docs] == [0, 1, 0, 1]
    treatment = list((PARENT / 'raw/CONFIRMATION').glob('*'))
    markers = list((PARENT / 'run').glob('CONFIRMATION*'))
    assert not treatment and not markers, ('HOLDOUT_ALREADY_USED', treatment, markers)
    # Search prior experiment trees by document identity without opening any
    # potential treatment values. Tokens/source documents are allowed objects.
    suspect = []
    identities = [x['document_id'] for x in docs]
    roots = [PARENT] + c.PARENTS
    for root in roots:
        for path in root.rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts: continue
            if any(doc in path.name for doc in identities):
                rel = path.relative_to(root)
                if 'inputs' not in rel.parts and 'tokens' not in rel.parts:
                    suspect.append(str(path))
    assert not suspect, ('CLEAN_HOLDOUT_UNEXPECTED_RESULT_OBJECT', suspect)
    return dict(gate='PASS',PARENT_CONFIRMATION_COMPLETE='0/64',
        parent_verdict=row(PARENT/'reports/FINAL_REPORT.json'),
        parent_raw_confirmation_files=0, parent_confirmation_start_markers=0,
        searched_prior_roots=[str(x) for x in roots], unexpected_document_objects=suspect,
        evidence='Parent controller stopped after failed discovery; no confirmation markers, treatment results or traces exist. No prior confirmation treatment results were available for scientific inspection.',
        allowed_prior_objects='immutable document identities and token IDs; no prior confirmation FP trace found',
        treatment_results_read_before_preregistration=0, document_ids=identities)

def parent_identity():
    identity = load(D/'provenance/parent_identity.json')
    assert command(['git','rev-parse','HEAD'],PARENT_ROOT) == PARENT_COMMIT
    assert command(['git','status','--porcelain'],PARENT_ROOT) == ''
    for tree in identity['trees']:
        assert [str(p) for p in inventory_files(Path(tree['root']))] == tree['paths'], tree['root']
    for item in identity['files']:
        assert row(item['path']) == item, ('PARENT_BYTE_CHANGED',item['path'])
    v=load(PARENT/'reports/FINAL_REPORT.json')['verdict']
    assert v['DISCOVERY_SURROGATE_GATE']=='FAIL' and v['CONFIRMATION_COMPLETE']=='0/64'
    assert v['PRIMARY_NEXT_STEP_CATEGORY']=='STOP_PROPAGATION_OBJECTIVE_TRAINING'
    return dict(gate='PASS',files_checked=len(identity['files']),parent_verdict='FAIL_PRESERVED')

def gate(panels):
    assert len(panels)==4
    defined=all(p['rho_INJ'] is not None and p['rho_PROP'] is not None for p in panels)
    beats=sum(p['rho_PROP'] is not None and p['rho_INJ'] is not None and p['rho_PROP']>p['rho_INJ'] for p in panels)
    positive=sum(p['rho_PROP'] is not None and p['rho_PROP']>0 for p in panels)
    med=statistics.median(p['rho_PROP'] for p in panels) if all(p['rho_PROP'] is not None for p in panels) else None
    concord=statistics.median(p['concordance_PROP'] for p in panels)
    rules=dict(A=beats>=3,B=positive>=3,C=med is not None and med>=.5,D=concord>=.65)
    supported=defined and all(rules.values())
    return dict(verdict='SUPPORTED' if supported else 'NOT_SUPPORTED',all_panels_defined=defined,
        rules=rules,JPROP_BEATS_JINJ_COUNT=beats,JPROP_POSITIVE_RHO_COUNT=positive,
        MEDIAN_JPROP_RHO=med,MEDIAN_JPROP_CONCORDANCE=concord,
        no_pvalues=True,no_pooled_rotation_document_correlation=True,
        primary_horizon=32768,unit='one document across 16 frozen rotations')

def unit_tests():
    def panels(inj=.4, prop=.5, conc=.65):
        return [dict(rho_INJ=inj,rho_PROP=prop,concordance_PROP=conc) for _ in range(4)]
    cases=0
    for pattern in itertools.product([False,True],repeat=4):
        a=panels()
        for p,win in zip(a,pattern):p['rho_INJ']=.4 if win else .5
        assert gate(a)['rules']['A']==(sum(pattern)>=3);cases+=1
    for pattern in itertools.product([False,True],repeat=4):
        a=panels()
        for p,pos in zip(a,pattern):p['rho_PROP']=.5 if pos else 0
        assert gate(a)['rules']['B']==(sum(pattern)>=3);cases+=1
    assert gate(panels())['verdict']=='SUPPORTED'
    assert gate(panels(prop=.5-1e-9))['rules']['C']==False
    assert gate(panels(conc=.65-1e-9))['rules']['D']==False
    a=panels();a[0]['rho_PROP']=None
    assert gate(a)['verdict']=='NOT_SUPPORTED'
    assert stats.ranks([1,1,3,2])==[1.5,1.5,4.,3.]
    assert stats.spearman(list(range(16)),list(range(16)))==1
    assert stats.spearman([1]*16,list(range(16))) is None
    assert stats.order([1]*16,[1]*16)['both_tied']==120
    assert stats.order([1]*16,[1]*16)['concordance']==0
    return dict(gate='PASS',synthetic_only=True,exhaustive_rule_A_B_cases=cases,
        ties='exact average-rank Spearman; concordant/120; ties non-success and separately reported',
        undefined_panel='fails gate without retry',threshold_boundary_tests='PASS')

def prepare():
    assert not list((D/'raw/confirmation').glob('*'))
    assert command(['git','rev-parse','HEAD'])==PARENT_COMMIT
    assert command(['git','status','--porcelain'],PARENT_ROOT)==''
    audit=clean_holdout();emit('provenance/clean_holdout_audit.json',audit)
    for name in ['premeasurement_sha256.json','final_sha256_inventory.json']:
        inv=load(PARENT/'manifests'/name)
        assert inv['gate']=='PASS'
        for item in inv['files']:
            assert row(item['path'])==item,('INHERITED_INVENTORY_FAIL',item['path'])
    roots=[PARENT]+c.PARENTS
    paths={str(p):p for root in roots for p in inventory_files(root)}
    parent_files=[row(paths[name]) for name in sorted(paths)]
    emit('provenance/parent_identity.json',dict(gate='PASS',parent_commit=PARENT_COMMIT,
        parent_branch=command(['git','branch','--show-current'],PARENT_ROOT),
        files=parent_files,trees=[dict(root=str(root),paths=[str(p) for p in inventory_files(root)]) for root in roots],
        PARENT_DISCOVERY_VERDICT='FAIL_PRESERVED',PARENT_GLOBAL_PROPAGATION_GATE='FAIL_PRESERVED'))
    rt=c.runtime();emit('provenance/runtime_identity.json',rt)
    hw=c.hardware();emit('provenance/hardware_identity.json',hw)
    assert all(x['used_mib']<2000 for x in hw['gpus']),('GPU_EXTERNAL_OCCUPANCY',hw)
    docs=[x for x in load(PARENT/'protocol/frozen_documents.json')['streams'] if x['phase']=='CONFIRMATION']
    for index,doc in enumerate(docs):
        token=doc['token_file'];assert row(token['path'])==token
        ids=load(token['path']);assert len(ids)==32769
        assert hashlib.sha256(b''.join(int(v).to_bytes(4,'little') for v in ids)).hexdigest()==doc['token_ids_sha256']
        assert doc['physical_gpu']==index%2 and doc['gpu_uuid']==c.UUIDS[index%2]
    emit('protocol/inherited_confirmation_docs.json',dict(gate='PASS',parent_manifest=row(PARENT/'protocol/frozen_documents.json'),
        streams=docs,assignment='parent selection_rank%2; equivalent to confirmation index%2',new_document_selection='NO'))
    bank=load(PARENT/'protocol/frozen_rotation_bank.json')
    assert bank['gate']=='PASS' and bank['count']==16
    assert bank['execution_order']==[x['condition_id'] for x in bank['conditions']]
    orth=[]
    for cond in bank['conditions']:
        assert row(cond['artifact']['path'])==cond['artifact']
        rs=core.torch.load(cond['artifact']['path'],map_location='cpu',weights_only=False)['rotations']
        assert set(rs)==set(c.LAYERS)
        for item in cond['layers']:
            tensor=rs[item['layer_id']]
            assert c.tensor_sha(tensor)==item['sha256']
            err=float((tensor.double().T@tensor.double()-core.torch.eye(128,dtype=core.torch.float64)).abs().max())
            assert err<=bank['orthogonality_max_abs_threshold'],('ORTHOGONALITY_FAIL',cond['condition_id'],item['layer_id'],err)
            # Parent records the same metric in double precision; exact source
            # hashes are the stronger matrix identity authority.
            assert abs(err-item['orthogonality_max_abs'])<=1e-12
            orth.append(dict(condition_id=cond['condition_id'],layer_id=item['layer_id'],max_abs=err,sha256=item['sha256']))
    emit('protocol/inherited_rotation_bank.json',dict(gate='PASS',parent_manifest=row(PARENT/'protocol/frozen_rotation_bank.json'),
        count=16,execution_order=bank['execution_order'],conditions=bank['conditions'],orthogonality_reverification=orth,
        new_rotation_selection='NO',rotation_regeneration='NO'))
    config=load(PARENT/'protocol/frozen_metrics_and_execution.json')
    assert config['probe_count']==8 and config['epsilon']==core.EPS==1e-12
    assert config['layer_ids']==c.LAYERS and core.WINDOW_START==1024
    sign=PARENT/'protocol/probe_signs.pt';assert row(sign)==config['probe_sign_schedule']['artifact']
    assert core.torch.equal(core.torch.load(sign,weights_only=True),core.sign_schedule(8,32768))
    (D/'protocol').mkdir(parents=True,exist_ok=True)
    shutil.copyfile(sign,D/'protocol/probe_signs.pt')
    assert sha(sign)==sha(D/'protocol/probe_signs.pt')
    (D/'protocol/probe_signs.pt').chmod(0o444)
    sources=[row(PARENT/'scripts'/name) for name in ['common.py','transition.py','replay_core.py','capture.py','worker.py','surrogate_statistics.py']]
    expected={x['path']:x for x in load(PARENT/'manifests/premeasurement_sha256.json')['files']}
    for source in sources:assert source==expected[source['path']]
    emit('protocol/metric_identity.json',dict(gate='PASS',parent_configuration=row(PARENT/'protocol/frozen_metrics_and_execution.json'),
        inherited_configuration=config,inherited_source_files=sources,
        PARENT_KDA_TRANSITION_IDENTITY_GATE='PASS',PARENT_QDQ_IDENTITY_GATE='PASS',
        PARENT_JINJ_DEFINITION_IDENTITY_GATE='PASS',PARENT_JPROP_DEFINITION_IDENTITY_GATE='PASS',
        operational_adapter=dict(changed_in_memory_globals=['common.D','common.R','common.TASK'],
            scope='output root, git root, experiment namespace only; exact parent source files and scientific functions imported unchanged',
            unused_parent_entrypoints=['worker.main','worker.frozen_guard','surrogate_statistics.analyze','surrogate_statistics.gate']),
        primary_horizon=32768,lower_horizon_raw_scores='ANCILLARY_ONLY; inherited unit output unchanged; never ranked or used by gate',
        J_DECAY='ANCILLARY_ONLY; inherited computation unchanged; no selection or gate',probe_count=8,probe_schedule=row(sign)))
    emit('protocol/preregistration.json',dict(EXPERIMENT=TASK,PARENT_EXPERIMENT='LING_PROPAGATION_SURROGATE_VALIDATION_V1',
        EVIDENCE_TYPE='PROSPECTIVE_HORIZON_SPECIFIC_CONFIRMATION_FROM_PRIOR_DISCOVERY',
        hypothesis='At T=32768 only, exact parent J_PROP ranks J_ACC more reliably than J_INJ on the four untouched parent confirmation docs.',
        PARENT_GLOBAL_PROPAGATION_GATE='FAIL_PRESERVED',primary_horizon=32768,window=[1024,32768],units=64,
        statistical_panel_unit='one document across 16 rotations',baseline='J_INJ',treatment='J_PROP',ground_truth='J_ACC',
        gate=dict(A='rho_PROP > rho_INJ in >=3/4 documents',B='rho_PROP > 0 in >=3/4 documents',C='median rho_PROP >=0.50',D='median concordance_PROP >=0.65'),
        verdict='SUPPORTED iff all four rules pass and correlations are defined; otherwise NOT_SUPPORTED',
        no_pvalues=True,no_discovery_confirmation_pooling=True,no_64_point_primary_correlation=True,
        no_alignment_term_computation=True,hardware='2xRTX4090 only',gpu_uuids=c.UUIDS,
        rotation_order=bank['execution_order'],probe_count=8,system_boundary=config['system_boundary'],
        NEW_TRAINING='NO',NEW_OPTIMIZER_STEP=0,NEW_AIME_GENERATION=0,NEW_FREE_GENERATION=0,
        NEW_ROTATION_LEARNING='NO',NEW_ROTATION_SELECTION='NO',NEW_DOCUMENT_SELECTION='NO',NEW_SURROGATE_SELECTION='NO',NEW_METRIC_FAMILY='NO',
        NEW_TRAINING_AUTHORIZED='NO',AIME_AUTHORIZED='NO',
        next_if_supported='proposal ONLY: LING_ERROR_COHERENCE_DIAGNOSIS_V1; never execute',
        next_if_not_supported='STOP_SIMPLE_PROPAGATION_ENERGY',FINAL_ACTION='STOP',
        failure_policy='completed units immutable; infrastructure crash preserves evidence and can resume incomplete units only; scientific semantic/runtime failure stops without changes'))
    emit('runtime/statistical_unit_tests.json',unit_tests())
    parent_identity()
    print('PREPARATION_IDENTITY_AND_HOLDOUT_GATES=PASS; NEW_SCIENTIFIC_MEASUREMENTS=0',flush=True)

def freeze():
    assert not (D/'run/STARTED.json').exists()
    for gpu in [0,1]:assert load(D/'runtime'/f'capture_canary_gpu{gpu}.json')['gate']=='PASS'
    p=parent_identity();rt=c.runtime();hw=c.hardware()
    emit('provenance/prefreeze_validation.json',dict(gate='PASS',parent=p,runtime=rt,hardware=hw,measurement_count=0))
    external={x['path']:row(x['path']) for x in load(PARENT/'manifests/premeasurement_sha256.json')['files']}
    # Include all parent scientific source bytes and historical/runtime artifacts
    # from its frozen inventory, plus every new preregistered object.
    own=[row(path) for path in inventory_files(D) if path.suffix!='.pt' or path.name=='probe_signs.pt']
    own += [row(path) for path in inventory_files(D) if path.name.startswith('chunk_') and path.suffix=='.pt']
    unique={x['path']:x for x in own+list(external.values())}
    inv=dict(gate='PASS',measurement_count=0,files=[unique[k] for k in sorted(unique)])
    emit('manifests/premeasurement_sha256.json',inv)
    text_file('provenance/PREMEASUREMENT_SHA256SUMS',''.join(f"{x['sha256']}  {x['path']}\n" for x in inv['files']))
    print('PREMEASUREMENT_FREEZE=PASS; MEASUREMENT_COUNT=0',flush=True)

def guard():
    assert not (D/'run/FORMAL_STOP.json').exists()
    receipt=load(D/'run/premeasurement_commit.json')
    assert command(['git','rev-parse','HEAD'])==receipt['commit']
    assert command(['git','status','--porcelain'])==''
    for item in load(D/'manifests/premeasurement_sha256.json')['files']:
        assert row(item['path'])==item,('FROZEN_SOURCE_OR_INPUT_CHANGED',item['path'])
    assert load(D/'provenance/clean_holdout_audit.json')['gate']=='PASS'
    return receipt

def validate_unit(path,doc,cond,commit):
    x=load(path)
    assert x['gate']=='PASS' and x['task']==TASK and x['phase']=='confirmation'
    assert x['document_id']==doc['document_id'] and x['condition_id']==cond['condition_id']
    assert x['physical_gpu']==doc['physical_gpu'] and x['gpu_uuid']==doc['gpu_uuid']
    assert x['selection_rank']==doc['selection_rank'] and x['token_ids_sha256']==doc['token_ids_sha256']
    assert x['rotation_sha256']==cond['artifact']['sha256'] and x['premeasurement_commit']==commit
    assert x['source_manifest_sha256']==sha(D/'manifests/premeasurement_sha256.json')
    assert x['probe_count']==8 and x['tokens_consumed']==32768
    assert x['capture_replay_bitwise_parity']=='PASS' and x['single_rollout_all_horizons'] is True
    assert x['sign_schedule_sha256']==sha(D/'protocol/probe_signs.pt')
    assert x['trace_sha256']==sha(D/'run/traces'/doc['document_id']/'trace_freeze.json')
    assert set(x['scores'])=={'2048','8192','32768'}
    for h,s in x['scores'].items():
        assert s['window']==[1024,int(h)] and s['tokens']==int(h)-1024+1
        assert all(math.isfinite(s[f]) and s[f]>=0 for f in ['J_ACC','J_INJ','J_PROP','J_DECAY'])
    return x

def worker(gpu):
    receipt=guard();c.runtime();hw=c.hardware()
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(gpu) and core.torch.cuda.device_count()==1
    assert hw['gpus'][gpu]['used_mib']<2000,('GPU_EXTERNAL_OCCUPANCY',hw)
    docs=[x for x in load(D/'protocol/inherited_confirmation_docs.json')['streams'] if x['physical_gpu']==gpu]
    assert len(docs)==2
    bank=load(D/'protocol/inherited_rotation_bank.json')
    signs=core.torch.load(D/'protocol/probe_signs.pt',map_location='cuda',weights_only=True)
    assert core.torch.equal(signs.cpu(),core.sign_schedule(8,32768))
    for doc in docs:
        guard();assert row(doc['token_file']['path'])==doc['token_file']
        ids=load(doc['token_file']['path']);assert len(ids)==32769
        root=D/'run/traces'/doc['document_id']
        if root.exists():
            # Only a previously fully frozen FP trace may be resumed. Partial
            # capture remains preserved; no silent overwrite or regeneration.
            assert (root/'trace_freeze.json').is_file(),('PARTIAL_CAPTURE_REQUIRES_REVIEW',str(root))
            trace=load(root/'trace_freeze.json')
        else:
            capture_model=capture.Capture()
            try:trace=capture_model.document(ids[:32768],root)
            finally:capture_model.close()
        assert trace['gate']=='PASS' and trace['tokens']==32768
        token_sha=hashlib.sha256(b''.join(int(v).to_bytes(4,'little') for v in ids[:32768])).hexdigest()
        assert trace['token_sha256']==token_sha and trace['layer_ids']==c.LAYERS
        assert len(trace['chunks'])==64
        for entry in trace['chunks']:assert row(entry['artifact']['path'])==entry['artifact']
        operational_doc=dict(doc,phase='confirmation')
        for cond in bank['conditions']:
            guard()
            path=D/'raw/confirmation'/(doc['document_id']+'__'+cond['condition_id']+'.json')
            if path.exists():
                validate_unit(path,doc,cond,receipt['commit'])
                print('IMMUTABLE_COMPLETE_UNIT_SKIPPED',path.name,flush=True)
                continue
            parent_worker.unit(operational_doc,cond,trace,8,signs,receipt['commit'])
            validate_unit(path,doc,cond,receipt['commit'])
    emit(f'run/GPU{gpu}_COMPLETE.json',dict(gate='PASS',units=32,gpu=gpu,gpu_uuid=c.UUIDS[gpu],commit=receipt['commit']))

def csv_file(relative, records, fields):
    path=D/relative;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(records)

def seal():
    receipt=guard()
    docs=load(D/'protocol/inherited_confirmation_docs.json')['streams']
    bank=load(D/'protocol/inherited_rotation_bank.json');order=bank['execution_order']
    paths=[D/'raw/confirmation'/(doc['document_id']+'__'+cond['condition_id']+'.json') for doc in docs for cond in bank['conditions']]
    assert len(paths)==64 and set(paths)==set((D/'raw/confirmation').glob('*.json'))
    # No numerical raw values are loaded before this inventory is emitted.
    frozen=dict(gate='PASS',complete=64,files=[row(p) for p in paths],before_aggregation=True,time_unix=time.time())
    emit('manifests/confirmation_raw_freeze.json',frozen)
    (D/'manifests/confirmation_raw_freeze.json').chmod(0o444)
    text_file('manifests/CONFIRMATION_RAW_SHA256SUMS',''.join(f"{x['sha256']}  {x['path']}\n" for x in frozen['files']))
    for path in paths:path.chmod(0o444)
    panels=[];pair_rows=[]
    for i,doc in enumerate(docs):
        values=[]
        for cond in bank['conditions']:
            path=D/'raw/confirmation'/(doc['document_id']+'__'+cond['condition_id']+'.json')
            x=validate_unit(path,doc,cond,receipt['commit']);values.append(x['scores']['32768'])
        acc=[v['J_ACC'] for v in values]
        inj=[v['J_INJ'] for v in values];prop=[v['J_PROP'] for v in values]
        ri=stats.spearman(inj,acc);rp=stats.spearman(prop,acc)
        oi=stats.order(inj,acc);op=stats.order(prop,acc)
        panels.append(dict(document_index=i,document_id=doc['document_id'],physical_gpu=doc['physical_gpu'],gpu_uuid=doc['gpu_uuid'],
            rho_INJ=ri,rho_PROP=rp,delta_rho=rp-ri if rp is not None and ri is not None else None,
            concordance_INJ=oi['concordance'],concordance_PROP=op['concordance']))
        for name,counts in [('J_INJ',oi),('J_PROP',op)]:pair_rows.append(dict(document_id=doc['document_id'],surrogate=name,**counts))
    decision=gate(panels)
    summary=dict(**decision,per_document=panels,raw_freeze_sha256=sha(D/'manifests/confirmation_raw_freeze.json'),
        parent_global_verdict='FAIL_PRESERVED',ancillary_lower_horizons_not_ranked=True,J_DECAY='ANCILLARY_ONLY',
        alignment_term_computed=False,premeasurement_commit=receipt['commit'])
    emit('analysis/32k_summary.json',summary)
    csv_file('analysis/per_document_32k_metrics.csv',panels,list(panels[0]))
    csv_file('analysis/pairwise_concordance.csv',pair_rows,list(pair_rows[0]))
    # Seal the primary verdict BEFORE even constructing the conditional proposal.
    emit('provenance/PRIMARY_VERDICT_SEALED.json',dict(verdict=decision['verdict'],summary=row(D/'analysis/32k_summary.json'),time_unix=time.time(),alignment_term_computed=False))
    if decision['verdict']=='SUPPORTED':
        emit('proposals/LING_ERROR_COHERENCE_DIAGNOSIS_V1.json',dict(
            proposal='LING_ERROR_COHERENCE_DIAGNOSIS_V1',status='PROPOSAL_ONLY_NOT_EXECUTED',
            NEW_EXPLORATORY_MECHANISM_HYPOTHESIS='YES',NEW_TRAINING_AUTHORIZED='NO',AIME_AUTHORIZED='NO',
            question='Why does J_PROP rank long-horizon error better at32K while its magnitude explains only a small fraction of J_ACC?',
            suggested_budget=dict(recurrence='e_(t+1)=M_t e_t+eta_(t+1)',
                HISTORY_ENERGY='||M_t e_t||^2',INJECTION_ENERGY='||eta_(t+1)||^2',
                ALIGNMENT_TERM='2 <M_t e_t,eta_(t+1)>',positive_alignment='reinforces accumulated error',negative_alignment='partially cancels existing error'),
            prerequisite='audit one-step budget closure under exact inherited BF16 rotated-write semantics; do not assume residual isolates alignment',
            confirmation_docs_future_policy='current four confirmation docs may not be reused as future confirmatory evidence',
            interpretation='new exploratory mechanism hypothesis, not reasoning evidence',EXECUTION_AUTHORIZED='NO'))
    parent=parent_identity();rt=c.runtime()
    verdict=dict(EXPERIMENT=TASK,PARENT_EXPERIMENT='LING_PROPAGATION_SURROGATE_VALIDATION_V1',
        EVIDENCE_TYPE='PROSPECTIVE_HORIZON_SPECIFIC_CONFIRMATION_FROM_PRIOR_DISCOVERY',
        PARENT_DISCOVERY_VERDICT='FAIL_PRESERVED',NEW_TRAINING='NO',NEW_AIME_GENERATION=0,
        CONFIRMATION_CLEAN_HOLDOUT_GATE='PASS',PARENT_ROTATION_BANK_IDENTITY_GATE='PASS',
        PARENT_RUNTIME_IDENTITY_GATE=rt['gate'],PARENT_KDA_TRANSITION_IDENTITY_GATE='PASS',PARENT_QDQ_IDENTITY_GATE='PASS',
        PARENT_JINJ_DEFINITION_IDENTITY_GATE='PASS',PARENT_JPROP_DEFINITION_IDENTITY_GATE='PASS',
        CONFIRMATION_COMPLETE='64/64',PRIMARY_HORIZON=32768)
    for i,p in enumerate(panels):
        verdict[f'DOC{i}_RHO_JINJ']=p['rho_INJ'];verdict[f'DOC{i}_RHO_JPROP']=p['rho_PROP'];verdict[f'DOC{i}_DELTA_RHO']=p['delta_rho']
    verdict.update(JPROP_BEATS_JINJ_COUNT=f"{decision['JPROP_BEATS_JINJ_COUNT']}/4",JPROP_POSITIVE_RHO_COUNT=f"{decision['JPROP_POSITIVE_RHO_COUNT']}/4",
        MEDIAN_JPROP_RHO=decision['MEDIAN_JPROP_RHO'],MEDIAN_JPROP_CONCORDANCE=decision['MEDIAN_JPROP_CONCORDANCE'],
        LONG_HORIZON_PROPAGATION_CONFIRMATION=decision['verdict'],PARENT_GLOBAL_PROPAGATION_GATE='FAIL_PRESERVED',
        REASONING_IMPROVEMENT='NOT_EVALUATED',NEW_TRAINING_AUTHORIZED='NO',AIME_AUTHORIZED='NO',
        PRIMARY_NEXT_STEP_CATEGORY='PREREGISTER_ERROR_COHERENCE_DIAGNOSIS' if decision['verdict']=='SUPPORTED' else 'STOP_SIMPLE_PROPAGATION_ENERGY',
        FINAL_ACTION='STOP',PREMEASUREMENT_COMMIT_SHA=receipt['commit'])
    if decision['verdict']=='NOT_SUPPORTED':verdict['SIMPLE_PROPAGATION_ENERGY_ROUTE']='NOT_SUPPORTED_AT_32K'
    answers=dict(Q1=f"{'YES' if decision['rules']['A'] else 'NO'}: {decision['JPROP_BEATS_JINJ_COUNT']}/4 documents strictly improve.",
        Q2=f"{'YES' if decision['rules']['C'] else 'NO'}: median J_PROP Spearman={decision['MEDIAN_JPROP_RHO']}.",
        Q3=f"{'YES' if decision['rules']['D'] else 'NO'}: median J_PROP concordance={decision['MEDIAN_JPROP_CONCORDANCE']}.",
        Q4='YES under the frozen 32K advancement rules.' if decision['verdict']=='SUPPORTED' else 'NO under the frozen 32K advancement rules.',
        Q5='NO. Parent V1 remains FAIL; its global gate and final report are unchanged.',
        Q6='NO. Even if supported, only a new mechanism-decomposition proposal is permitted; no training.',
        Q7='NO. No training, AIME or reasoning endpoint was measured.')
    emit('FINAL_REPORT.json',dict(verdict=verdict,per_document=panels,rules=decision['rules'],questions=answers,parent_validation=parent,runtime_validation=rt))
    block='\n'.join(f'{k} = {v if v is not None else "UNDEFINED"}' for k,v in verdict.items())
    table='| Doc | ID | GPU | rho_INJ | rho_PROP | delta_rho | concordance_INJ | concordance_PROP |\n|---|---|---|---|---|---|---|---|\n'
    for p in panels:
        table+='| '+' | '.join(str(p[k]) for k in ['document_index','document_id','physical_gpu','rho_INJ','rho_PROP','delta_rho','concordance_INJ','concordance_PROP'])+' |\n'
    conclusion=('On four previously untouched long-context documents, the exact full-transition J_PROP surrogate prospectively outperformed raw J_INJ for ranking32K accumulated recurrent-state distortion across the frozen16-rotation bank. This does not overturn parent V1 failure.' if decision['verdict']=='SUPPORTED' else 'The separate32K-specific hypothesis did not satisfy the frozen confirmation rules on the four untouched holdout documents. Stop simple propagation-energy advancement; parent V1 remains failed.')
    text_file('FINAL_REPORT.md','```ini\n'+block+'\n```\n\n# Per-document32K results\n\n'+table+
        '\n# Frozen rules\n\n'+json.dumps(decision['rules'],sort_keys=True)+'\n\n# Q1-Q7\n\n'+
        '\n\n'.join('## '+q+'\n\n'+a for q,a in answers.items())+
        '\n\n# Scientific boundary\n\n'+load(D/'protocol/preregistration.json')['system_boundary']+
        '. Not full-model autoregressive on-policy. Each rotation retains its own persistent INT8 writeback. Only32768 panels enter the verdict; no pooled64-point correlation, no p-values, no alignment-term measurement.\n\n'+
        conclusion+'\n\n# Audit\n\nParent byte-identity PASS over '+str(parent['files_checked'])+' files; exact runtime PASS. Raw64/64 SHA freeze preceded ranking. No training, optimizer updates, AIME, free generation or rotation learning.\n\nFINAL_ACTION = STOP\n')
    emit('provenance/COMPLETION.json',dict(gate='PASS',final_verdict=verdict,parent_validation=parent,
        TRAINING_STARTED='NO',OPTIMIZER_STEPS=0,AIME_STARTED='NO',FREE_GENERATION=0,
        ALIGNMENT_TERM_COMPUTED='NO',PARENT_VERDICT_CHANGED='NO',FINAL_ACTION='STOP',time_unix=time.time()))
    final_files=[row(p) for p in inventory_files(D) if 'run' not in p.relative_to(D).parts and 'logs' not in p.relative_to(D).parts and p.name!='FINAL_SHA256SUMS']
    text_file('FINAL_SHA256SUMS',''.join(f"{x['sha256']}  {x['path']}\n" for x in final_files))
    for x in final_files:assert row(x['path'])==x
    emit('run/final_inventory_verification.json',dict(gate='PASS',files_checked=len(final_files),inventory=row(D/'FINAL_SHA256SUMS')))
    print('LONG_HORIZON_PROPAGATION_CONFIRMATION='+decision['verdict']+'; FINAL_ACTION=STOP',flush=True)

def controller():
    receipt=guard();assert not (D/'run/STARTED.json').exists()
    assert not list((D/'raw/confirmation').glob('*.json'))
    parent_identity();c.runtime();hw=c.hardware()
    assert all(x['used_mib']<2000 for x in hw['gpus'])
    emit('run/STARTED.json',dict(commit=receipt['commit'],time_unix=time.time(),parent_verdict='FAIL_PRESERVED'))
    processes=[];handles=[]
    try:
        for gpu in [0,1]:
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1')
            path=D/'logs'/f'CONFIRMATION_GPU{gpu}.log';path.parent.mkdir(parents=True,exist_ok=True)
            stream=path.open('x');handles.append(stream)
            proc=subprocess.Popen([str(c.PYTHON),str(D/'scripts/experiment.py'),'worker','--gpu',str(gpu)],
                cwd=R,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            processes.append(proc)
        emit('run/WORKERS.json',dict(pids=[p.pid for p in processes],gpus=[0,1],gpu_uuids=c.UUIDS))
        while True:
            status=[p.poll() for p in processes]
            if any(x is not None and x!=0 for x in status):raise RuntimeError(('WORKER_FAILURE',status))
            if all(x==0 for x in status):break
            time.sleep(2)
    finally:
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for h in handles:h.close()
    seal()
    subprocess.run(['git','diff','--check'],cwd=R,check=True)
    subprocess.run(['git','add','--',str(D.relative_to(R))],cwd=R,check=True)
    artifacts=['raw/confirmation','analysis','FINAL_REPORT.md','FINAL_REPORT.json','FINAL_SHA256SUMS',
        'provenance/COMPLETION.json','provenance/PRIMARY_VERDICT_SEALED.json','manifests/confirmation_raw_freeze.json',
        'manifests/CONFIRMATION_RAW_SHA256SUMS']
    if (D/'proposals').exists():artifacts.append('proposals')
    subprocess.run(['git','add','-f','--']+[str((D/x).relative_to(R)) for x in artifacts],cwd=R,check=True)
    subprocess.run(['git','commit','-m','experiment: finalize prospective32K propagation confirmation'],cwd=R,check=True)
    emit('run/final_commit_receipt.json',dict(commit=command(['git','rev-parse','HEAD']),worktree=command(['git','status','--porcelain']),push_performed=False,FINAL_ACTION='STOP'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','canary','freeze','worker','controller']);parser.add_argument('--gpu',type=int)
    args=parser.parse_args()
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='canary':capture.canary(args.gpu)
        elif args.mode=='freeze':freeze()
        elif args.mode=='worker':worker(args.gpu)
        else:controller()
    except BaseException as exc:
        name='FORMAL_STOP.json' if (D/'run/STARTED.json').exists() else f'PREMEASUREMENT_{args.mode}_FAILURE_{args.gpu}.json'
        if not (D/'run'/name).exists():emit('run/'+name,dict(gate='FAIL',stage=args.mode,gpu=args.gpu,error=repr(exc),traceback=traceback.format_exc(),time_unix=time.time(),NEW_TRAINING='NO',NEW_AIME_GENERATION=0,FINAL_ACTION='STOP',automatic_retry=False))
        raise
