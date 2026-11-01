"""CPU-only source verification and bounded fill attribution, no runtime imports."""
from pathlib import Path
import ast,hashlib,json
R=Path(__file__).resolve().parents[1]
cat=R/'analysis/fill_source_catalog.json';j=json.loads(cat.read_text());checks=[]
for row in j['files']:
 if row.get('status')=='UNAVAILABLE':continue
 p=R/row['local'];raw=p.read_bytes();checks.append(dict(check='SHA_AND_BYTES',path=row['local'],PASS=hashlib.sha256(raw).hexdigest()==row['sha256'] and len(raw)==row['bytes']))
 if p.suffix=='.py':ast.parse(raw.decode());checks.append(dict(check='PYTHON_AST_PARSE_NO_IMPORT',path=row['local'],PASS=True))
def read(n):return (R/'evidence/fill_official_pytorch'/n).read_text()
base='aten/src/ATen/native/'
requirements=[('cuda_empty_fill_requires_two_flags',base+'cuda/TensorFactories.cu','deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory()'),('cuda_resize_preserves_old_and_grows_unknown',base+'cuda/Resize.cpp','std::min(storage->nbytes(), size_bytes)'),('cuda_resize_fill_guard',base+'cuda/Resize.cpp','fill_resize_deterministic_(self, old_storage_nbytes)'),('clone_has_copy_after_alloc',base+'TensorFactories.cpp','self.copy_(src);'),('to_copy_has_write',base+'TensorConversions.cpp','r.copy_(self, non_blocking);'),('repeat_interleave_materializes_clone',base+'Repeat.cpp','input.clone(at::MemoryFormat::Contiguous).flatten(dim, dim + 1)'),('arange_bounded_every_index_write',base+'cuda/RangeFactories.cu','data[idx] = f(idx);'),('depthwise_output_assign_not_accumulate',base+'cuda/DepthwiseConv2d.cu','output.data()[linearIndex] = static_cast<scalar_t>(value);'),('flash_scratch_allocations',base+'transformers/cuda/flash_attn/flash_api.cpp','params.oaccum_ptr = out_accum.data_ptr();'),('flash_masked_output_copy',base+'transformers/cuda/flash_attn/flash_fwd_kernel.h','/*Clear_OOB_K=*/false'),('flash_uninitialized_shared_mask_comment',base+'transformers/cuda/flash_attn/flash_fwd_kernel.h',"We don't need to clear the sK smem tiles since we'll mask out the scores anyway.")]
for name,n,needle in requirements:checks.append(dict(check=name,path=n,PASS=needle in read(n)))
d=json.loads((R/'evidence/fill_previous/analysis/trace_kernel_delta.json').read_text());fill=[x for x in d['rows'] if 'FillFunctor<' in x['name']];summary=[]
for x in fill:
 summary.append(dict(name=x['name'],old_per_token=x['old_count']/128,new_per_token=x['new_count']/128,delta_per_token=x['delta_per_token'],new_parent_paths_shown_per_token={k:v/128 for k,v in x['new_paths'].items()},new_unshown_parent_count_per_token=(x['new_count']-sum(x['new_paths'].values()))/128))
checks.append(dict(check='fill_actual_kernel_delta_724',PASS=sum(x['delta_per_token'] for x in fill)==724))
checks.append(dict(check='all_delta847',PASS=sum(x['delta_per_token'] for x in d['rows'])==847))
out=dict(status='PASS_CPU_SOURCE_AND_COUNT_CHECKS_NOT_MEMORY_SAFETY_QUALIFICATION' if all(x['PASS'] for x in checks) else 'FAIL_CLOSED_STOP',checks=checks,fill_kernel_attribution=summary,total_fill_delta_per_token=724,caused_by_fill_flag_exact_count='UNPROVEN',global_fill_disable_approved=False,memory_safety_status='FILL_OPT_SAFETY_UNPROVEN',GPU_dispatches=0)
(R/'analysis/fill_source_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(out['status'],len(checks),'checks')
assert all(x['PASS'] for x in checks)
