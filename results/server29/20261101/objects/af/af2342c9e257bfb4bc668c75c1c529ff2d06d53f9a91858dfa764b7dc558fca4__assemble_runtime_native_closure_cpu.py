#!/usr/bin/env python3
"""Assemble exact native-file admission data from two fresh CPU receipts.

This runs without Torch and does not inspect a GPU. A later loader-only probe
may contribute separately reviewed exact native rows, but a directory prefix,
basename, or bare successful mapping can never stand in for provenance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

STATUS='PASS_FRESH_CPU_IMPORTS_AND_PROVENANCED_NATIVE_CLOSURE_NO_GPU'
FINAL='PASS_TWO_FRESH_CPU_IMPORTS_AND_PROVENANCED_NATIVE_CLOSURE_NO_GPU'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(path):return json.loads(Path(path).read_text())


def need(ok,message):
    if not ok:raise RuntimeError(message)


def check_row(row):
    need(Path(row['path']).is_absolute(),'Native row is not an absolute canonical path')
    need(all(isinstance(row[k],int)and row[k]>=0 for k in('bytes','inode','device','mtime_ns')),'Invalid native stat')
    need(len(row['sha256'])==64 and all(c in'0123456789abcdef'for c in row['sha256']),'Invalid exact SHA')
    need(isinstance(row['provenance'],dict)and row['provenance'],'Missing native provenance')


def assemble(a,b,paths,base_path,tool_path,extension_path=None):
    base_sha=sha(base_path)
    root=Path(base_path).resolve().parents[1]
    for x in(a,b):
        need(x['status']==STATUS and x['schema_version']==1,'CPU receipt status differs')
        need(x['base_external_bindings_sha256']==base_sha,'Original61 binding bytes differ')
        need(x['CUDA_VISIBLE_DEVICES']==''and x['Torch_CUDA_initialized']is False,'CPU process initialized CUDA')
        need(x['CPU_PID_absent_from_NVML']is True,'CPU process NVML evidence missing')
        need(all(x[k]==0 for k in('model_instances_created','Tensor_forwards','GPU_dispatches','GPU_contexts_created')),'Non-CPU work recorded')
        need(not x['unresolved_required_ELF_dependencies'],'Required native provider unresolved')
        need(x['import_recipe']==a['import_recipe'],'Different CPU import recipe')
        need(len(x['files'])==x['count']==len({f['path']for f in x['files']}),'Duplicate native rows')
        need(sum(f['bytes']for f in x['files'])==x['bytes'],'Native byte count differs')
        for row in x['files']:check_row(row)
        allowed={f['path']for f in x['files']}
        need({m['path']for m in x['mapped_native_files']}<=allowed,'Actual CPU native map omitted')
    need(a['identity']!=b['identity'],'Not two independent fresh CPU processes')
    need(a['files']==b['files'],'Native SHA/stat/provenance changed between fresh CPU processes')
    need(a['aliases']==b['aliases'],'Native aliases changed between fresh CPU processes')
    need(a['selected_installed_distributions']==b['selected_installed_distributions'],'Provider inventory changed')
    def optional_scope(x):
        return [dict(path=r['path'],missing_DT_NEEDED=sorted(set(re.findall(r'(\S+) => not found',r['stdout']))),
            scope=r['scope'],no_runtime_mapping_in_CPU_recipe=r['no_runtime_mapping_in_CPU_recipe'],
            exact_file_still_pinned=r['exact_file_still_pinned'])for r in x['unavailable_optional_audio_plugins']]
    need(optional_scope(a)==optional_scope(b),'Optional provider availability classification changed')
    files={x['path']:x for x in a['files']};extension=None
    manual_system=[x['path']for x in a['files']if x['provenance'].get('manual_system_ELF_dependency')]
    if extension_path:
        extension=load(extension_path)
        need(extension['status']=='PASS_EXACT_LOADER_PROBE_ELF_PROVENANCE_CPU_REVIEW','Unreviewed loader extension')
        need(extension['base_external_bindings_sha256']==base_sha,'Extension original61 mismatch')
        need(extension['probe_receipt']['sha256']and extension['probe_receipt']['identity'],'Missing exact probe evidence')
        for row in extension['files']:
            check_row(row);p=row['provenance']
            package=bool(p.get('wheel_RECORD')or p.get('conda_metadata')or p.get('system_RPM')or p.get('manual_system_driver')or p.get('manual_system_ELF_dependency'))
            generated=p.get('generated_build')
            if generated:
                need(generated['loaded_mapping_matches_exact_ELF']is True,'Generated ELF mapping not bound')
                need(generated['build_source_matches_retained_compilation_input']is True,'Generated source identity unresolved')
                need(generated['source_files']and generated['upstream_pinned_sources'],'Missing generated-source chain')
                need(generated['compiler']['sha256']and generated['build_command'],'Missing compiler/build identity')
            need(package or generated,'Exact loader row lacks package or pinned generated-build provenance')
            if row['path']in files:need(files[row['path']]==row,'Loader extension silently replaces native identity')
            else:files[row['path']]=row
    rows=[files[p]for p in sorted(files)]
    receipts=[]
    for path,x in zip(paths,(a,b)):
        receipts.append(dict(path=str(Path(path).resolve().relative_to(root)),bytes=Path(path).stat().st_size,sha256=sha(path),identity=x['identity'],
                             CPU_only=True,CUDA_initialized=False,CPU_PID_absent_from_NVML=True))
    return dict(schema_version=1,status=FINAL,base_external_bindings_sha256=base_sha,files=rows,
        count=len(rows),bytes=sum(x['bytes']for x in rows),aliases=a['aliases'],cpu_probe_receipts=receipts,
        collector=dict(path=str(Path(tool_path).resolve().relative_to(root)),bytes=Path(tool_path).stat().st_size,sha256=sha(tool_path)),
        actual_CPU_import_recipe=a['import_recipe'],manual_system_ELF_dependency_paths=manual_system,
        manual_system_provenance_limit='Exact current stat/SHA/ELF build-id and incoming native loader dependency chain verified; upstream package/build origin is unresolved, not claimed verified.',selected_installed_distributions=a['selected_installed_distributions'],
        unavailable_optional_audio_plugins=optional_scope(a),
        scope=dict(actual_CPU_mapped_ELF_covered=True,required_Torch_CUDA_Triton_driver_profiler_static_dependencies_resolved=True,
                   installed_optional_audio_plugins_not_claimed_executable=True,
                   runtime_allowance='INDIVIDUAL_CANONICAL_PATH_STAT_SHA_ONLY_NO_PREFIX_ALLOW',
                   GPU_qualification=False,model_forward_qualification=False,TPS_measured=False,
                   old_STOP_rejudged=False),
        loader_only_extension=None if extension is None else dict(path=str(extension_path),sha256=sha(extension_path),
            probe_receipt=extension['probe_receipt'],additional_file_count=len(rows)-len(a['files'])))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--first',type=Path,required=True)
    ap.add_argument('--second',type=Path,required=True);ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--collector',type=Path,required=True);ap.add_argument('--extension',type=Path)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    result=assemble(load(a.first),load(a.second),(a.first,a.second),a.base,a.collector,a.extension)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x')as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(status=result['status'],count=result['count'],bytes=result['bytes'],sha256=sha(a.output))))


if __name__=='__main__':main()
