"""Read-only historical-input recovery into a new, clean experiment tree."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
TASK = 'LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'
ROOT = Path('/data01/user2/worktrees/ling-long-horizon-observability-bridge-v1')
D = ROOT / 'experiments' / TASK
K = Path('/data01/user2/.codex/worktrees/GDN-quantization/ling-kda-int8-rc-granularity')
P = Path('/data01/user2/worktrees/ling-propagation-surrogate-validation-v1/experiments/LING_PROPAGATION_SURROGATE_VALIDATION_V1')
C = Path('/data01/user2/worktrees/ling-long-horizon-propagation-confirm-v1/experiments/LING_LONG_HORIZON_PROPAGATION_CONFIRM_V1')
N = Path('/data01/user2/worktrees/ling-error-coherence-diagnosis-v1/experiments/LING_ERROR_COHERENCE_DIAGNOSIS_V1')
UUID = 'GPU-de53f243-b271-521c-d875-d96611ed8574'

def git(*args, cwd=K):
    return subprocess.check_output(['git', *args], cwd=cwd)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def row(p):
    p = Path(p)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))

def dump(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: json.dump(x, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')

def text(p, s):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: f.write(s)

def main():
    assert git('rev-parse', 'HEAD', cwd=ROOT).decode().strip() == 'd9e23eeee086f527e7ad4e488ab9a9c99d71cc93'
    dirty = git('status', '--porcelain=v1', '--untracked-files=all', cwd=ROOT).decode()
    assert all(l[3:].startswith('experiments/'+TASK+'/') for l in dirty.splitlines()), dirty
    assert not (D/'preregistration.json').exists()
    before = git('status', '--porcelain=v1')
    head = git('rev-parse', 'HEAD').decode().strip()
    text(D/'historical_assets/git_status_porcelain_v1.txt', before.decode())
    text(D/'historical_assets/git_HEAD.txt', head+'\n')
    # Conservative source snapshot: includes dormant helpers but authorizes only
    # the narrowly enumerated historical entry points, never old experiment mains.
    selected = sorted(set(list((K/'experiments/ling').glob('*.py')) + list((K/'tests/ling').glob('*.py'))))
    roots = [K/'results/LING_KDA_QUERY_READOUT_CAUSAL_CLOSURE_V1',
             K/'results/LING_KDA_KERNEL_INTERNAL_DIRECTIONAL_GEOMETRY_DECOMPOSITION_V1']
    for root in roots:
        selected += [p for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    for folder in ['results/ling', 'reports/ling', 'runs/ling_kda_source_error_functional_causal_v1',
                   'runs/ling_kda_query_readout_causal_closure_v1']:
        root=K/folder
        if root.is_dir():
            selected += [p for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
                         and ('source_error_functional_causal' in str(p) or 'query_readout_causal' in str(p))]
    selected=sorted(set(selected))
    tracked=set(git('ls-files','-z').decode().split('\0'))
    inventory=[]
    for p in selected:
        rel=str(p.relative_to(K)); copy=D/'historical_assets/snapshot'/rel
        status=git('status','--porcelain=v1','--untracked-files=all','--',rel).decode()
        classification='untracked' if rel not in tracked else 'tracked-modified' if status else 'tracked-clean'
        old=row(p);copy.parent.mkdir(parents=True,exist_ok=True)
        assert not copy.exists();shutil.copyfile(p,copy)
        assert sha(copy)==old['sha256']; copy.chmod(0o444)
        item=dict(**old,relative_path=rel,classification=classification,git_status=status,
                  snapshot_path=str(copy),snapshot_sha256=sha(copy),byte_identity='PASS')
        if classification=='tracked-modified':
            diff=git('diff','--',rel); staged=git('diff','--cached','--',rel)
            for name,data in [('worktree',diff),('index',staged)]:
                target=D/'historical_assets/diffs'/name/(rel+'.diff')
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as f:f.write(data)
                item[name+'_diff_artifact']=row(target)
        inventory.append(item)
    assert before==git('status','--porcelain=v1') and head==git('rev-parse','HEAD').decode().strip()
    text(D/'historical_assets/git_status_after_snapshot.txt',git('status','--porcelain=v1').decode())
    assert all(sha(i['path'])==i['sha256'] for i in inventory)
    dump(D/'historical_assets/asset_manifest.json',dict(
        legacy_worktree=str(K),legacy_HEAD=head,legacy_branch=git('branch','--show-current').decode().strip(),
        status_artifact=row(D/'historical_assets/git_status_porcelain_v1.txt'),files=inventory,
        HISTORICAL_DIRTY_WORKTREE_DISCLOSED='YES',HISTORICAL_ASSET_BYTE_IDENTITY_GATE='PASS',
        source_tree_claim='FROZEN_HISTORICAL_INPUT_ASSETS; NOT A CLEAN SOURCE TREE; NOT GIT HEAD CONTENT',
        legacy_files_written=0,old_execution_directory_used=False,measurement_count=0))
    names=['run_ling_kda_query_readout_causal_closure_v1.py','pytorch_kda_reference.py',
           'run_ling_kda_source_error_functional_causal_v1.py','run_ling_kda_persistent_error_decomposition_causal_v1.py']
    imports={}
    for name in names:
        p=D/'historical_assets/snapshot/experiments/ling'/name
        source=p.read_text();tree=ast.parse(source)
        imports[name]=[ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    dump(D/'historical_assets/executed_dependency_plan.json',dict(
        roots={'K3':['query_readout','prepared_query'],'KL':['full_logit_metrics'],
               'cache':['clone_cache','cache_stack','replace_cache_stack']},
        required_historical_modules=names,imports=imports,
        edges=[['query.prepared_query','reference._prepared_drivers'],
               ['reference._prepared_drivers','reference._expand_value_heads/_gate_decay/_beta_weights'],
               ['functional.<module>','persistent.<module>'],
               ['functional.cache_stack/replace_cache_stack','persistent.get_cache_state']],
        historical_tests=['test_ling_kda_query_readout_causal_closure.py::test_query_readout_matches_actual_kda_tensor_orientation',
                          'test_ling_kda_kernel_internal_directional_geometry_decomposition.py::test_one_step_reference_exposes_named_kernel_checkpoints',
                          'test_ling_kda_kernel_internal_directional_geometry_decomposition.py::test_exact_state_error_transfer_matches_finite_difference'],
        closure_gate='PENDING_POSTCOMMIT_EXECUTION_VALIDATION',old_mains_forbidden=True,
        bytecode_writes_forbidden=True,unknown_historical_execution='FAIL_AND_STOP',
        external_runtime='Existing qualified environment; every actually executed module/source must be inventoried by the execution tracer, including packages and model code.'))
    inherited=json.loads((N/'source_identity.json').read_text())
    protected=[]
    for root in [P,C,N]:
        protected += [row(p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    dump(D/'source_identity.json',dict(inputs=inherited['inputs'],protected_historical_files=protected,
         dirty_asset_manifest=row(D/'historical_assets/asset_manifest.json'),
         parent_commits={str(P.parents[1]):'dc602d2d2d3b78346b6e89d9b6a71a9ad9ddc53b',
                         str(C.parents[1]):'caf2fff04505d74c1596fb90ec1378fed85b2fc7',
                         str(N.parents[1]):'d9e23eeee086f527e7ad4e488ab9a9c99d71cc93'}))
    docs=[d for d in json.loads((P/'protocol/frozen_documents.json').read_text())['streams'] if d['phase']=='DISCOVERY']
    assert [d['document_id'] for d in docs]==['pg19-test-37702','pg19-test-5396','pg19-test-28988','pg19-test-26493']
    dump(D/'document_manifest.json',dict(documents=docs,evidence_role='EXPLORATORY_DISCOVERY_ONLY',source=row(P/'protocol/frozen_documents.json')))
    bank=json.loads((P/'protocol/frozen_rotation_bank.json').read_text())
    dump(D/'rotation_manifest.json',dict(bank=bank,source=row(P/'protocol/frozen_rotation_bank.json')))
    dump(D/'runtime_manifest.json',dict(expected=json.loads((N/'runtime_manifest.json').read_text()),
         proposed_hardware=dict(host='nlpg-SYS-4029GP-TRT',physical_gpu=6,gpu_uuid=UUID),
         GPU6_FORMAL_AUTHORIZED=False,required_new_gates=['RUNTIME_IDENTITY_GATE','HARDWARE_ASSIGNMENT_GATE','FP_REFERENCE_PAIRING_GATE','FRESH_PROCESS_DETERMINISM_GATE'],
         FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO',cross_hardware_FP_bitwise='FAIL_PRESERVED'))
    text(D/'basis_semantics_audit.md','''# Basis and exact transition definition

The inherited persistent trajectory stores Q in rotated Value coordinates.
The actual captured native Q endpoint is Q_rot @ R.T (FP32), not reconstructed FP+E.
FP is the assigned-GPU native FP trace. E = actual Q_native - actual FP (FP32).
Reference and pulse continue with the actual inherited fused FP KDA transition,
in the same canonical Value basis and with identical frozen native drivers.
No affine/Jacobian shortcut is used. K3 calls frozen historical query_readout
and prepared_query, including historical query normalization/scaling/head expansion.
All K3/output/logit comparisons are in the canonical functional basis.
Orthogonality uses the inherited frozen bank gate, without increasing its tolerance.
This is a definition audit, NOT a numerical BASIS_SEMANTICS_GATE PASS.
''')
    text(D/'pulse_definition.md','''# Pulse definition

Replay the historical actual persistent INT8-R128 transition from zero to each
anchor using the frozen same-GPU FP external drives. Use actual Q_rot @ R.T.
At anchor A=FP, B=actual Q_native; capture E=B-A and verify identical pulse delta.
No QDQ or further injection occurs after anchor in either candidate or target.
Candidate uses exact two-path frozen-driver fused FP KDA recurrence.
Target uses two full-model natural FP paths with identical teacher-forced tokens,
identical non-KDA cache at anchor, and only KDA states replaced with actual Q.
No cache approximation, fresh source-error substitution, or new target selection.
''')
    prereg=dict(EXPERIMENT=TASK,EVIDENCE_CATEGORY='PROSPECTIVE_EXPLORATORY_MECHANISM_BRIDGE',
        document_manifest=row(D/'document_manifest.json'),rotation_manifest=row(D/'rotation_manifest.json'),
        model_checkpoint='Ling-3.0-tiny official revision e3a47d5b986e7141b6efd62597d598ebb392060d; exact inherited SHA inventory',
        primary_anchor=8192,secondary_anchor=2048,future_endpoint=32768,primary_units=64,secondary_units=64,
        future_support='anchor+1 through 32768 inclusive; token index is one-based count of consumed teacher-forced tokens',
        candidate_definition=dict(RAW_ACC_ERROR='sum E_anchor.float32().double().square()',
            LOCAL_K3_ACCUMULATED='historical query_readout(E_anchor,prepared_query_at_anchor); FP32 then FP64 squared norm',
            HISTORICAL_J_PROP='Call unchanged inherited replay_step; support 1024..anchor; mean over tokens then 18 layers',
            J_INJ='Same historical replay_step index1; support 1024..anchor; mean over tokens then 18 layers',
            PULSE_PROP_ENERGY_AUC='sum_future squared Frobenius norm of two-path FP32 state difference, FP64 reduction',
            POR='sum_future squared norm of FP32 historical K3_readout(pulse)-K3_readout(ref), FP64 reduction',
            POR_GAIN='POR/RAW_ACC_ERROR; epsilon=0; exactly zero denominator returns 0'),
        primary_target=dict(name='FUTURE_FULL_LOGIT_KL_AUC',implementation='frozen historical functional.full_logit_metrics[KL]',
             direction='ref||pulse',dtype='historical FP32 softmax/log_softmax/per-token sum',
             auc='streaming FP64 sum of historical per-token FP32 scalar; no averaging across future tokens; no negative-value clamping',
             model_path='natural full-model teacher-forced FP; only actual accumulated KDA cache state pulse at anchor'),
        ranking=dict(implementation='inherited stats.spearman/stats.order; exact ties, no added tolerance',
             top_bottom='set intersection fraction of exact tied minimum/maximum sets, divided by union size',
             statistical_unit='document; no pooled rotation-level p-value'),
        numeric_dtype=dict(state='FP32',drives='inherited BF16 q/k/v, FP32 log_decay/beta',reduction='FP64'),
        basis_definition=row(D/'basis_semantics_audit.md'),pulse_definition=row(D/'pulse_definition.md'),
        determinism=dict(fresh_process_repeats=2,bitwise=True,development_tokens='inherited wikitext2raw-train-000 fixture, first128',
            development_anchor=64,development_end=128,rotation='H',seed=0,PYTHONHASHSEED='0',
            fixed_rmsnorm='BT16_W8',no_tolerance_relaxation=True),
        decision_rule=dict(primary_only=True,PRESENT='rho POR>PROP in >=3/4 docs AND median delta_rho>0 AND concordance POR>PROP in >=3/4 docs AND median rho_POR>0',
             MIXED='not PRESENT AND median delta_rho>0; inconsistent Spearman/concordance or document consistency',
             NOT_PRESENT='otherwise; secondary cannot override primary'),
        hardware=dict(proposed_gpu=6,uuid=UUID,conditional_authorization_only=True,same_gpu_FP_pairing_required=True,
             FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO',cross_hardware_FP_bitwise='FAIL_PRESERVED'),
        historical_inputs=dict(HISTORICAL_DIRTY_WORKTREE_DISCLOSED='YES',asset_manifest=row(D/'historical_assets/asset_manifest.json'),
             closure_plan=row(D/'historical_assets/executed_dependency_plan.json'),legacy_worktree_read_only=True,legacy_execution_forbidden=True),
        endpoint_32K='RAW_ACC_ERROR and LOCAL_K3_ACCUMULATED only, plus historical J_INJ/J_PROP/J_ACC; no future POR',
        storage=dict(full_future_logits_saved=False,top_event_count=5,streaming=True),
        blocking_gates=['HISTORICAL_ASSET_BYTE_IDENTITY_GATE','HISTORICAL_EXECUTED_DEPENDENCY_CLOSURE_GATE',
            'SOURCE_IDENTITY_GATE','MODEL_CHECKPOINT_GATE','ROTATION_BANK_IDENTITY_GATE','DOCUMENT_IDENTITY_GATE',
            'KDA_TRANSITION_IDENTITY_GATE','QDQ_IDENTITY_GATE','BASIS_SEMANTICS_GATE','HISTORICAL_K3_PARITY_GATE',
            'HISTORICAL_JPROP_PARITY_GATE','ACCUMULATED_ERROR_CAPTURE_GATE','INITIAL_PULSE_IDENTITY_GATE',
            'NO_FURTHER_QDQ_GATE','FROZEN_DRIVE_IDENTITY_GATE','FUTURE_KL_HARNESS_PARITY_GATE','FRESH_PROCESS_DETERMINISM_GATE',
            'STORAGE_CAPACITY_GATE','RUNTIME_IDENTITY_GATE','HARDWARE_ASSIGNMENT_GATE','FP_REFERENCE_PAIRING_GATE'],
        FRESH_SOURCE_ERROR_DIRECTION='HISTORICALLY_ESTABLISHED',CURRENT_TARGET='LONG_HORIZON_ACCUMULATED_ERROR_OBSERVABILITY',
        parent_global_propagation='FAIL_PRESERVED',prior32K_confirmation='SUPPORTED_PRESERVED',coherence='DESCRIPTIVE_SIGNAL_NOT_PRESENT_PRESERVED',
        CONFIRMATION_EXECUTED=0,L9_AUTHORIZED='NO',AIME_AUTHORIZED='NO',NEW_TRAINING_AUTHORIZED='NO',FINAL_ACTION='STOP_AFTER_DISCOVERY')
    dump(D/'preregistration.json',prereg)
    text(D/'preregistration.md','# '+TASK+'\n\nThe JSON preregistration is the exact prospective authority. No outcomes have been computed.\n\n```json\n'+json.dumps(prereg,indent=2,sort_keys=True)+'\n```\n')
    dump(D/'provenance.json',dict(source_identity=row(D/'source_identity.json'),preregistration=row(D/'preregistration.json'),
         HISTORICAL_DIRTY_WORKTREE_DISCLOSED='YES',HISTORICAL_ASSET_BYTE_IDENTITY_GATE='PASS',
         HISTORICAL_EXECUTED_DEPENDENCY_CLOSURE_GATE='NOT_YET_EXECUTED',historical_sources_are_HEAD_bytes=False,
         legacy_execution_directory_used=False,legacy_changes_committed=False,legacy_provenance_rewritten=False,
         measurement_count=0,AIME_EXECUTED='NO',L9_TRAINING_EXECUTED='NO',new_rotation_training=False))
    print('HISTORICAL_BYTE_SNAPSHOT_PASS',len(inventory),'PREREGISTRATION_CREATED_NO_MEASUREMENT',flush=True)

if __name__=='__main__':main()
