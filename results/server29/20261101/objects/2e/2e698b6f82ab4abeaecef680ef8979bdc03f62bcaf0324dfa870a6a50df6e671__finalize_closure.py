"""Seal engineering evidence only; never start parent measurements here."""
import argparse, hashlib, json, subprocess
import closure_common as C

def main():
    identity=C.verify_parent()
    C.save('provenance/final_parent_stop_revalidation.json',identity)
    faulty=C.ROOT/'scripts/finalize_closure.failed_001.py'
    if faulty.exists():
        C.save('provenance/report_packaging_incident_001.json',dict(
            type='REPORT_ONLY_SYNTAX_ERROR_BEFORE_EXECUTION',faulty_script_sha256=C.sha(faulty),
            corrected_script_sha256=C.sha(__file__),cause='Invalid reason= spelling in one dictionary literal',
            scientific_execution_repeated=False,parity_results_changed=False,gate_policy_changed=False,
            original_failure_retained=True,report_packaging_repaired=True))
    spec={
        'operator_determinism.json':'FIXED_ORDER_CUMSUM_DETERMINISM_GATE',
        'operator_parity.json':'CUMSUM_NUMERICAL_PARITY_GATE',
        'chunk_gdn_parity.json':'PATCHED_CHUNK_GDN_PARITY_GATE',
        'full_model_probe_parity.json':'FULL_MODEL_SHORT_PROBE_PARITY_GATE',
        'fresh_process_determinism.json':'PATCHED_FULL_MODEL_FRESH_PROCESS_DETERMINISM_GATE'}
    gates={};failed=[]
    for file,key in spec.items():
        p=C.ROOT/'provenance'/file
        if not p.exists(): C.save('provenance/'+file,{key:'NOT_EVALUATED','reason':'Stopped before this gate; no retry or fabricated evidence'})
        result=C.read(p);gates[key]=result[key]
        if result[key]!='PASS': failed.append(key)
    dtype=C.ROOT/'provenance/chunk_dtype_revalidation.json'
    if dtype.exists() and C.read(dtype)['gate']!='PASS':failed.append('CHUNK_DTYPE_REVALIDATION')
    errors=[]
    for p in sorted((C.ROOT/'provenance').glob('*.error.json')):
        errors.append(dict(path=str(p.relative_to(C.ROOT)),sha256=C.sha(p),evidence=C.read(p)))
    blocker=C.ROOT/'provenance/additional_deterministic_blocker_audit.json'
    if not blocker.exists():
        count=sum(bool(e['evidence'].get('additional_unsupported_deterministic_op')) for e in errors)
        C.save('provenance/additional_deterministic_blocker_audit.json',dict(
           ADDITIONAL_UNSUPPORTED_DETERMINISTIC_OP_COUNT=count if count else 'NOT_FULLY_EVALUATED',
           complete_dev256_strict_probe=False,additional_operations_patched=0,errors=errors))
    additional=C.read(blocker)['ADDITIONAL_UNSUPPORTED_DETERMINISTIC_OP_COUNT']
    perf=C.ROOT/'provenance/performance_benchmark.json'
    if not perf.exists():C.save('provenance/performance_benchmark.json',dict(status='NOT_EVALUATED',reason='Requires all equivalence gates PASS',PATCHED_PREFILL_SLOWDOWN=None))
    performance=C.read(perf); slowdown=performance.get('PATCHED_PREFILL_SLOWDOWN')
    ok=not failed and not errors and additional==0 and slowdown is not None and slowdown<=1.5
    verdict=dict(EXPERIMENT='QWEN_GDN_DETERMINISTIC_CUMSUM_CLOSURE_V1',PARENT_EXPERIMENT='QWEN_GDN_PROPAGATION_SURROGATE_VALIDATION_V1',
        EVIDENCE_TYPE='ENGINEERING_DETERMINISTIC_RUNTIME_COMPATIBILITY_CLOSURE',PARENT_STOP_PRESERVED='YES',NEW_TRAINING='NO',
        NEW_AIME_GENERATION=0,NEW_FREE_GENERATION=0,TORCH_VERSION=C.read(C.ROOT/'provenance/original_source_identity.json')['torch_version'],
        CUDA_RUNTIME=C.read(C.ROOT/'provenance/original_source_identity.json')['cuda_runtime'],BLOCKED_OPERATION='torch.cumsum',
        BLOCKED_SOURCE_LOCATION=str(C.ARCH)+':'+str(C.read(C.ROOT/'provenance/original_source_identity.json')['blocked_line']),
        CUMSUM_RUNTIME_DIM_LENGTH=64,**gates,PATCHED_PREFILL_SLOWDOWN=slowdown,
        ADDITIONAL_UNSUPPORTED_DETERMINISTIC_OP_COUNT=additional,
        SURGICAL_PATCH_ROUTE='VIABLE' if ok else 'NOT_SCALABLE' if isinstance(additional,int) and additional>=2 else 'NOT_EVALUATED',
        DETERMINISTIC_CUMSUM_COMPATIBILITY_GATE='PASS' if ok else 'FAIL',
        PARENT_PROPAGATION_EXPERIMENT_AUTHORIZED_TO_RESUME='YES' if ok else 'NO',
        RESUME_FROM_GATE='NATIVE_BASIS_MAPPING_GATE' if ok else 'NONE',NEW_C8_TRAINING_AUTHORIZED='NO',
        FINAL_ACTION='PASS_AND_RESUME_PARENT' if ok else 'FAIL_AND_STOP')
    C.save('provenance/final_verdict.json',dict(verdict=verdict,failed_gates=failed,errors=errors,
            parent_stop_unchanged=True,parent_discovery_started=0,parent_confirmation_started=0))
    header='\n\n'.join(k+' =\n'+str(v) for k,v in verdict.items())+'\n\n'
    text=header+'# Engineering closure\n\n'
    text+='The original parent STOP and all47 sealed files remain byte-identical. No recovery, Cayley reconstruction, rotation audit or scientific metric was repeated. Shared Transformers, model, quantizer and rotations are unchanged. Only an experiment-local compilation of the original chunk function replaces one inclusive scan. Reference-only processes are explicitly labeled and never supply scientific evidence.\n\n'
    text+='## Gate evidence\n\n'
    for k,v in gates.items():text+=f'- {k}: {v}\n'
    text+='\n## Numerical evidence\n\n'
    for file,key in [('operator_parity.json','operator'),('chunk_gdn_parity.json','chunk'),('full_model_probe_parity.json','full model')]:
        d=C.read(C.ROOT/'provenance'/file); rows=d.get('rows',[])
        if rows:text+=f'{key}: {len(rows)} comparisons; max absolute error {max(r["max_abs"] for r in rows):.12g}; maximum relative L2 {max(r["relative_L2"] for r in rows):.12g}.\n\n'
    full=C.read(C.ROOT/'provenance/full_model_probe_parity.json').get('rows',[])
    if full:
        text+='## Full-model failure breakdown\n\n'
        for name in ['historical32','dev256']:
            rr=[r for r in full if r['probe']==name]; ff=[r for r in rr if not r['PASS']]
            ll=next(r for r in rr if r['object']=='logits')
            text+=name+': '+str(len(rr)-len(ff))+'/'+str(len(rr))+' objects PASS; logits max_abs='+str(ll['max_abs'])+', relative_L2='+str(ll['relative_L2'])+'.\n\n'
            if ff:text+='First failing saved evidence object: '+json.dumps(ff[0],sort_keys=True)+'.\n\n'
        text+='The single-token-prefill historical path is exact; the multi-chunk128-token prefill and following cache trajectory are not numerically qualified by the frozen caps. Small FP32 scan association changes can cross BF16 rounding boundaries and compound through the full network. This is a plausible numerical mechanism, not a measured propagation-science result. The reference process disables strict enforcement, so this comparison also cannot isolate scan-association effects from any other backend dispatch differences induced by that flag. No causal attribution to the scan alone is claimed. The observed failure does not justify relaxing caps, writing an optimized alternative, or resuming the parent.\n\n'
    text+='All thresholds and near-zero relative-error floors were prospectively frozen. Raw max_rel and floor-stabilized max_rel are both retained. Operator calibration used only the first8 prefixes of head0 of one captured fixture; downstream caps were dtype-derived prospective engineering caps, not tuned model noise estimates. The chunk output dtype routing incident is recorded in the append-only chunk_dtype_revalidation.json, using the already-frozen stricter FP32 policy; original evidence is retained.\n\n'
    text+='## Performance\n\n'+json.dumps(performance,indent=2)+'\n\n'
    text+='## Questions\n\n'
    answers={
        'Q1':'The preserved traceback identifies strict CUDA cumsum policy support as the blocker. Previously passed model/checkpoint/source identity evidence was reused, not reconstructed. '+('The original path executes and the isolated replacement qualifies; no corruption evidence.' if ok else 'No independent corruption conclusion beyond those frozen audits is inferred from an incomplete closure.'),
        'Q2':'YES, within the unchanged frozen engineering caps.' if ok else 'See individual numerical gates; overall qualification is not established.',
        'Q3':'YES, five fresh strict processes, both fixed probes, all output/state/trace hashes exact.' if gates['PATCHED_FULL_MODEL_FRESH_PROCESS_DETERMINISM_GATE']=='PASS' else 'NOT_ESTABLISHED; see saved gate/error.',
        'Q4':('YES, slowdown '+str(slowdown)+'x; simple implementation accepted; no optimized kernel written.') if ok else 'Not established or not acceptable; see performance evidence.',
        'Q5':'No additional unsupported deterministic operation observed in the completed256-token probe.' if additional==0 else str(additional)+'; preserved evidence; no second operation patched.',
        'Q6':'YES, from NATIVE_BASIS_MAPPING_GATE with the original independent2K/8K/32K protocol, thresholds, bank and advancement rules. Closure itself starts no Discovery.' if ok else 'NO; preserve STOP and wait for explicit decision.',
        'Q7':'NO. No J_INJ/J_DECAY/J_PROP/J_ACC, training, optimizer step, rotation-bank measurement, AIME or free generation.'}
    for k,v in answers.items():text+=k+': '+v+'\n\n'
    text+='## Reproduction\n\nUse /data/ydai/miniconda3/envs/bitdecode/bin/python, GPU0, CUBLAS_WORKSPACE_CONFIG=:4096:8. Capture before implementation; run scripts/run_operator_qualification.sh, run_chunk_parity.py --phase reference/strict/analyze, chunk_dtype_revalidation.py, then scripts/run_full_qualification.sh in that order. Artifacts are write-once, so reproduce into a separate empty experiment-local copy with the same frozen inputs, sources and policy; never overwrite this seal.\n\n'
    text+='## Limits\n\nOnly fixed FP engineering probes and the first actual GDN input fixtures were tested. Changed floating-point association is numerically qualified, not bitwise-equivalent to the original reference. Exact reproducibility is required between patched executions. Benchmark uses the actual frozen first development document, shorter than2048; no fabricated length expansion or8192 test. This is not evidence about rotation quality or propagation-aware metrics.\n'
    C.save('provenance/report_answers.json',answers)
    with (C.ROOT/'FINAL_REPORT.md').open('x') as f:f.write(text)
    if ok:
        patch=C.read(C.ROOT/'provenance/patched_source_identity.json')
        C.save('provenance/parent_resume_authorization.json',dict(ORIGINAL_RUNTIME_STOP='PRESERVED',COMPATIBILITY_CLOSURE='PASS',
          QUALIFIED_RUNTIME='ORIGINAL_QWEN_RUNTIME + EXPERIMENT_LOCAL_DETERMINISTIC_CUMSUM_PATCH',
          RESUME_FROM_GATE='NATIVE_BASIS_MAPPING_GATE',parent_stop_manifest_sha256=identity['manifest_sha256'],
          original_source_sha256=patch['original_source_sha256'],patch_diff_sha256=patch['patch_diff_sha256'],patched_source_sha256=patch['patched_function_sha256'],
          qualification_artifacts={str(p.relative_to(C.ROOT)):C.sha(p) for p in sorted((C.ROOT/'provenance').glob('*.json'))},
          scientific_protocol_changed=False,Ling_outcomes_used=False,closure_discovery_started=0))
    # Required uppercase spellings and lowercase deliverables share immutable content.
    for src,dst in [('provenance/cumsum_runtime_shape_audit.json','provenance/CUMSUM_RUNTIME_SHAPE_AUDIT.json'),('protocol/numerical_parity_policy.json','protocol/NUMERICAL_PARITY_POLICY.json')]:
        if (C.ROOT/src).exists():
            with (C.ROOT/dst).open('xb') as f:f.write((C.ROOT/src).read_bytes())
    all_files=[p for p in sorted(C.ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='FINAL_SHA256SUMS']
    with (C.ROOT/'FINAL_SHA256SUMS').open('x') as f:
        for p in all_files:f.write(C.sha(p)+'  '+str(p.relative_to(C.ROOT))+'\n')
    print(json.dumps(verdict),flush=True)

if __name__=='__main__':main()
