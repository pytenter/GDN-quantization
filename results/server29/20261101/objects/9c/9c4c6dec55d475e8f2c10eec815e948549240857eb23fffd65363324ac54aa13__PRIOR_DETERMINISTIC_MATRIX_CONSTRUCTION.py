"""One comprehensive CPU-only V2 attribution and canonical-basis construction."""
import sys,os,ast,json,math,hashlib,traceback
from pathlib import Path
R=Path(sys.argv[1]).resolve();sys.path[:0]=[str(R/'src'),str(R/'inherited/group32/src')]
import execution_control as C
import numpy as np
import torch
from canonical_h128 import hadamard,GDN_LAYERS
torch.set_num_threads(1)
def norm2(a):return float(np.linalg.norm(a,2))
def maximum_diag_error(a,c,target):return float(np.max(np.abs(np.diag(a.T@c@a)-target)))
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
contract=C.read(R/'V2_NUMERICAL_QUALIFICATION_CONTRACT.json');cf=C.read(R/'CPU_ATTRIBUTION_SOURCE_FREEZE.json')
for x in cf['files']:C.require(C.sha(R/x['path'])==x['sha256'],'Pre-frozen CPU attribution source changed:'+x['path'])
C.require(not (R/'CPU_ATTRIBUTION_STARTED.json').exists(),'CPU attribution already started; no rerun selecting a candidate')
C.atomic_new(R/'CPU_ATTRIBUTION_STARTED.json',dict(identity=C.process_identity(os.getpid()),contract_sha256=C.sha(R/'V2_NUMERICAL_QUALIFICATION_CONTRACT.json'),source_freeze_sha256=C.sha(R/'CPU_ATTRIBUTION_SOURCE_FREEZE.json')))
try:
 asset=C.read(R/'COVARIANCE_ASSET_COMPLETENESS.json');C.require(asset['status']=='PASS_COMPLETE24_LAYERS_SHARED32_HEADS_4096_EXISTING_SAMPLES','Existing complete covariance not verified')
 cq_path=R/asset['copied_Cq']['path'];C.require(C.sha(cq_path)==asset['copied_Cq']['sha256'],'Original Cq bytes changed')
 source=(R/'reuse/V1_build_eigenbasis_cpu.py').read_text();fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='basis');ns=dict(np=np,C=C);exec(compile(ast.Module(body=[fn],type_ignores=[]),'<unchanged_V1_basis_function>','exec'),ns);basis=ns['basis']
 H64=hadamard(dtype=torch.float64).numpy();H32=hadamard().numpy();C.require(np.array_equal(H64.astype(np.float32),H32),'Canonical FP64 H single cast differs from original H32; cannot silently alter H')
 unit=2.**-24;gamma=128*unit/(1-128*unit);eps64=np.finfo(np.float64).eps
 output=R/'calibration/rotations';output.mkdir(exist_ok=False);rows=[];layers=[];failures=[];shared=[]
 with np.load(cq_path) as moments:
  for layer in GDN_LAYERS:
   all_C=moments[str(layer)];C.require(all_C.shape==(32,128,128),'Actual Cq shape differs')
   shares=[np.array_equal(all_C[2*h],all_C[2*h+1]) for h in range(16)];shared.append(dict(layer=layer,pairs_equal=shares));C.require(all(shares),'Actual shared Query Cq pair bytes differ')
   us=[];deploy=[];eigenvalues=[];head_assets=[]
   for head in range(16):
    c=all_C[2*head];C.require(np.isfinite(c).all(),'Nonfinite frozen Cq');w,U,clusters=basis(c);mu=float(np.trace(c)/128);cn=max(float(np.max(np.abs(w))),1e-300);fro=max(float(np.linalg.norm(c)),1e-300)
    A64=U@H64;A32=A64.astype(np.float32);Af=A32.astype(np.float64);U32=U.astype(np.float32);c32=c.astype(np.float32);c32f=c32.astype(np.float64)
    M2=(torch.from_numpy(U32)@torch.from_numpy(H32)).numpy();M2f=M2.astype(np.float64)
    M3=(torch.from_numpy(A32).T@torch.from_numpy(c32)@torch.from_numpy(A32)).numpy().astype(np.float64)
    raw0=maximum_diag_error(A64,c,mu);raw1=maximum_diag_error(Af,c,mu);raw2=maximum_diag_error(M2f,c,mu);raw3=float(np.max(np.abs(np.diag(M3)-mu)))
    use_relative=mu>1e-24;denom=mu if use_relative else 1.;mode='relative_to_trace_C_over128' if use_relative else 'absolute_near_zero_mu'
    da=norm2(Af-A64);a64n=norm2(A64);afn=norm2(Af);dc=norm2(c32f-c);evaluation_noise=512*eps64*cn*max(1.,a64n*a64n)
    representation=cn*(2*da+da*da);c_budget=afn*afn*dc;round_budget=(2*gamma+gamma*gamma)*float(np.max(np.diag(np.abs(Af).T@np.abs(c32f)@np.abs(Af))))
    eval1=Af.T@c@Af;eval_c32=Af.T@c32f@Af;c_change=float(np.max(np.abs(np.diag(eval_c32-eval1))));checker_change=float(np.max(np.abs(np.diag(M3-eval_c32))))
    orth_U=float(np.max(np.abs(U.T@U-np.eye(128))));orth_A=float(np.max(np.abs(A64.T@A64-np.eye(128))));orth_A32=norm2(Af.T@Af-np.eye(128));res=float(np.linalg.norm(c@U-U*w[None,:])/fro)
    orth_budget=2*da+da*da+5e-11
    checks=dict(C_symmetric=np.array_equal(c,c.T),PSD=float(w[-1])>=-1e-12*cn,FP64_eigen=res<=5e-11,FP64_U_orth=orth_U<=2e-11,FP64_A_orth=orth_A<=2e-11,FP64_diagonal=raw0<=((5e-11*mu) if use_relative else (5e-11*cn+1e-300)),FP32_single_cast=np.array_equal(A32,A64.astype(np.float32)),FP32_A_orth=orth_A32<=orth_budget and orth_A32<=2*gamma,FP32_representation_explained=raw1<=raw0+representation+evaluation_noise,C32_cast_explained=c_change<=c_budget+evaluation_noise,FP32_checker_explained=checker_change<=round_budget+evaluation_noise)
    Ucast=U32.astype(np.float64)@H64;Hcast=U@H32.astype(np.float64);both_cast=U32.astype(np.float64)@H32.astype(np.float64)
    row=dict(layer=layer,unique_query_head=head,recurrent_heads=[2*head,2*head+1],sample_count=4096,mode=mode,mu=mu,lambda_max=float(w[0]),lambda_min=float(w[-1]),eigenvalues=w.tolist(),condition_number=float(w[0]/w[-1]) if w[-1]>0 else None,degenerate_clusters=clusters,FP64_eigen_relative=res,FP64_U_orth_max=orth_U,FP64_A_orth_max=orth_A,FP32_A_orth_spectral=orth_A32,FP32_A_orth_budget=orth_budget,delta_A_spectral=da,M0=raw0/denom,M1=raw1/denom,M2_V1_construct_FP32_evaluate_FP64=raw2/denom,M3_requested_FP32_checker=raw3/denom,M0_absolute=raw0,M1_absolute=raw1,M2_absolute=raw2,M3_absolute=raw3,representation_budget=representation/denom,representation_budget_absolute=representation,C_cast_actual=c_change/denom,C_cast_budget=c_budget/denom,checker_actual=checker_change/denom,checker_budget=round_budget/denom,U32_only=maximum_diag_error(Ucast,c,mu)/denom,H32_only=maximum_diag_error(Hcast,c,mu)/denom,U32_H32_product_evaluated_FP64=maximum_diag_error(both_cast,c,mu)/denom,FP32_product_rounding_spectral=norm2(M2f-both_cast),A_single_storage_rounding_spectral=da,FP64_evaluation_noise_budget=evaluation_noise/denom,V1_2e6_exceeded_M2=bool(raw2/(mu if mu>1e-24 else 1)>2e-6),checks={k:bool(v) for k,v in checks.items()},status='PASS' if all(checks.values()) else 'FAIL',Cq_sha256=digest(c),U64_sha256=digest(U),A64_sha256=digest(A64),A32_sha256=digest(A32))
    rows.append(row)
    if not all(checks.values()):failures.append(dict(layer=layer,unique_query_head=head,recurrent_heads=row['recurrent_heads'],failed_checks=[k for k,v in checks.items() if not v]))
    us.append(U);deploy.append(A32);eigenvalues.append(w);head_assets.append(dict(unique_query_head=head,recurrent_heads=[2*head,2*head+1],U64_sha256=row['U64_sha256'],A32_sha256=row['A32_sha256']))
   p=output/('layer_%02d.npz'%layer)
   with p.open('xb') as out:np.savez(out,U_fp64_unique=np.stack(us),A_fp32_unique=np.stack(deploy),eigenvalues_unique=np.stack(eigenvalues),recurrent_to_unique=np.arange(32)//2);out.flush();os.fsync(out.fileno())
   layers.append(dict(layer=layer,path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=C.sha(p),unique_heads=16,recurrent_heads=32,head_matrices=head_assets))
 metrics=['M0','M1','M2_V1_construct_FP32_evaluate_FP64','M3_requested_FP32_checker','C_cast_actual','checker_actual','representation_budget','FP32_A_orth_spectral','FP64_eigen_relative','FP64_U_orth_max','delta_A_spectral'];distribution={}
 for key in metrics:
  a=np.array([r[key] for r in rows]);distribution[key]=dict(maximum=float(np.max(a)),P50=float(np.quantile(a,.5)),P95=float(np.quantile(a,.95)),P99=float(np.quantile(a,.99)))
 status='PASS_ALL384_UNIQUE_HEADS_CANONICAL_SINGLE_CAST' if not failures else 'FAIL_U_BRANCH_ALL384_SCAN_COMPLETE'
 C.atomic_new(R/'FP32_M0_M3_NUMERICAL_ATTRIBUTION.json',dict(status=status,unique_heads=384,recurrent_heads=768,original_V1_checker_actual_dtype='A constructed FP32, then evaluated in FP64 with original FP64 C. M2 reproduces that original path. M3 is requested additional full-FP32 checker diagnostic, not a misdescription of V1.',canonical_choice_fixed_before_scan='M1 only, independent of V1 threshold crossing',rows=rows,distributions=distribution,failures=failures,shared_head_checks=shared,FP32_unit_roundoff=unit,gamma128=gamma,contract_sha256=C.sha(R/'V2_NUMERICAL_QUALIFICATION_CONTRACT.json'),Cq_sha256=C.sha(cq_path),new_Query_forwards=0,AIME_used=False))
 C.atomic_new(R/'FP64_MATH_QUALIFICATION.json',dict(status='PASS' if not [r for r in rows if not all(v for k,v in r['checks'].items() if k.startswith('FP64') or k in ['C_symmetric','PSD'])] else 'FAIL',unique_heads=384,all_head_rows='FP32_M0_M3_NUMERICAL_ATTRIBUTION.json',shared_Query_pairs_exact=True,FP64_H_sha256=digest(H64),original_H32_sha256=digest(H32)))
 C.atomic_new(R/'CANONICAL_UH_MATRICES_AND_SHA.json',dict(status='PASS_SEALED_SHARED_CANONICAL_UH' if not failures else 'NOT_QUALIFIED_U_BRANCH_STOP',layers=layers,canonical_path='Original C64, deterministic U64, H64, A64=U64@H64, single FP32 conversion',unique_heads_per_layer=16,recurrent_mapping=[i//2 for i in range(32)],contract_sha256=C.sha(R/'V2_NUMERICAL_QUALIFICATION_CONTRACT.json'),attribution_sha256=C.sha(R/'FP32_M0_M3_NUMERICAL_ATTRIBUTION.json'),C1_S3_same_files=True,refit_from_AIME=False))
 C.atomic_new(R/'branches/U_CPU_ELIGIBILITY.json',dict(status='PASS' if not failures else 'FAIL_CLOSED_U_ONLY',eligible=not failures,failures=failures,P1_not_blocked=True))
 text='V2 canonical path is fixed M1, not selected by crossing the old threshold. Unit roundoff u=2^-24; gamma128=128u/(1-128u). For each head, the diagonal representation budget uses norm(C,2)*(2*norm(DeltaA,2)+norm(DeltaA,2)^2), plus actual M0 residual and an explicit FP64 evaluation budget. Near-zero mu<=1e-24 uses absolute errors. C32 conversion and two FP32 matrix products have separate norm/componentwise budgets. A32 orthogonality is checked against the representation perturbation bound and2*gamma128. This does not replace actual9B FP functional qualification. V1 remains FAIL under its original threshold.\n\n'+json.dumps(distribution,indent=2)+'\n\nFailures: '+json.dumps(failures)+'\n'
 (R/'FP32_ERROR_BUDGET_ANALYSIS.md').write_text(text)
 print(json.dumps(dict(status=status,unique_heads=384,failures=failures,distributions=distribution)),flush=True)
except BaseException as exc:
 if not (R/'branches/U_CPU_FAILURE.json').exists():C.atomic_new(R/'branches/U_CPU_FAILURE.json',dict(status='V2_NUMERICAL_CONTRACT_FAIL_CLOSED_STOP_U_ONLY',error=repr(exc),traceback=traceback.format_exc(),P1_not_blocked=True))
 raise
