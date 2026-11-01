"""Once-only deterministic CPU FP64 eigenbasis, independent of AIME results."""
import sys,os,json,hashlib,traceback
from pathlib import Path
R=Path(sys.argv[1]).resolve();sys.path.insert(0,str(R/'src'));sys.path.insert(0,str(R/'inherited/group32/src'))
import execution_control as C
import numpy as np
import torch
from canonical_h128 import hadamard,GDN_LAYERS
torch.set_num_threads(1)
def basis(c):
 w,u=np.linalg.eigh(c);w=w[::-1].copy();u=u[:,::-1].copy();scale=max(float(w[0]),1e-30);clusters=[];a=0
 while a<128:
  b=a+1
  while b<128 and w[b-1]-w[b]<=1e-12*scale:b+=1
  if b-a>1:
   projector=u[:,a:b]@u[:,a:b].T;cols=[]
   for j in range(128):
    v=projector[:,j].copy()
    for _ in range(2):
     for q in cols:v-=q*np.dot(q,v)
    norm=np.linalg.norm(v)
    if norm>1e-10:cols.append(v/norm)
    if len(cols)==b-a:break
   C.require(len(cols)==b-a,'Canonical degenerate subspace basis incomplete');u[:,a:b]=np.stack(cols,axis=1);clusters.append([a,b])
  a=b
 for j in range(128):
  if u[int(np.argmax(np.abs(u[:,j]))),j]<0:u[:,j]*=-1
 return w,u,clusters
try:
 C.require(C.read(R/'CALIBRATION_SESSION_RESULT.json')['status']=='PASS_COLLECTION_EXIT_AND_RELEASE','Collection process not released')
 audit=C.read(R/'QUERY_SECOND_MOMENT_AUDIT.json');C.require(C.sha(R/audit['moments']['path'])==audit['moments']['sha256'],'Cq bytes changed')
 C.require(not (R/'U_EIGENBASIS_MATRICES_AND_SHA.json').exists(),'U already sealed; no refit')
 # Fixed degenerate controls, two LAPACK invocations must yield the same canonical basis.
 rng=np.random.default_rng(20261022);q,_=np.linalg.qr(rng.normal(size=(128,128)));synthetic=q@np.diag(np.repeat(np.arange(1,17),8))@q.T
 w,u,_=basis(synthetic);w2,u2,_=basis(synthetic.copy());C.require(np.array_equal(u,u2),'Canonical repeated eigensolve differs')
 C.require(np.max(np.abs(u.T@u-np.eye(128)))<2e-11,'Degenerate synthetic orthogonality failed')
 h=hadamard(128).numpy();h64=h.astype(np.float64);layers=[];diagnostics=[];path=R/'calibration/rotations';path.mkdir(exist_ok=False)
 with np.load(R/audit['moments']['path']) as moments:
  for layer in GDN_LAYERS:
   cq=moments[str(layer)];U=[];eigs=[];details=[]
   for head in range(32):
    c=cq[head];norm=max(np.linalg.norm(c),1e-30);C.require(np.isfinite(c).all() and np.array_equal(c,c.T),'Cq symmetry/finite failed')
    w,u,clusters=basis(c);C.require(w[-1]>=-1e-12*max(w[0],1e-30),'Cq not PSD within frozen tolerance')
    orth=float(np.max(np.abs(u.T@u-np.eye(128))));res=float(np.linalg.norm(c@u-u*w[None,:])/norm)
    C.require(orth<=2e-11 and res<=5e-11,'Eigenbasis FP64 mathematical qualification failed')
    u32=u.astype(np.float32);a32=(torch.from_numpy(u32)@torch.from_numpy(h)).numpy();orth32=float(np.max(np.abs(a32.astype(np.float64).T@a32.astype(np.float64)-np.eye(128))))
    transformed=a32.astype(np.float64).T@c@a32.astype(np.float64);target=float(np.trace(c)/128);uniform=float(np.max(np.abs(np.diag(transformed)-target))/max(abs(target),1e-30))
    C.require(orth32<=2e-6 and uniform<=2e-6,'FP32 deployed orthogonality or Query diagonal identity failed')
    details.append(dict(head=head,eigenvalues=w.tolist(),negative_roundoff_count=int((w<0).sum()),minimum_eigenvalue=float(w[-1]),condition=float(w[0]/w[-1]) if w[-1]>0 else 'UNRESOLVED_SINGULAR_OR_ROUNDOFF',degenerate_clusters=clusters,orthogonality_max_abs=orth,eigendecomposition_relative_frobenius=res,deployed_A_orthogonality_max_abs=orth32,uniform_diagonal_relative_max=uniform,U_FP64_sha256=hashlib.sha256(u.tobytes()).hexdigest(),U_FP32_sha256=hashlib.sha256(u32.tobytes()).hexdigest(),A_FP32_sha256=hashlib.sha256(a32.tobytes()).hexdigest()))
    U.append(u);eigs.append(w)
   U=np.stack(U);U32=U.astype(np.float32);A=(torch.from_numpy(U32)@torch.from_numpy(h)).numpy();C.require(np.array_equal(U[::2],U[1::2]),'Shared actual Query heads got different U')
   p=path/('layer_%02d.npz'%layer)
   with p.open('xb') as f:np.savez(f,U_fp64=U,U_fp32=U32,A_fp32=A,eigenvalues=np.stack(eigs));f.flush();os.fsync(f.fileno())
   layers.append(dict(layer=layer,path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=C.sha(p),head_matrices=details));diagnostics.extend(dict(layer=layer,**x) for x in details)
 C.atomic_new(R/'U_EIGENBASIS_MATRICES_AND_SHA.json',dict(status='PASS_SEALED_SHARED_U_AND_A',Cq_sha256=audit['moments']['sha256'],dataset_sha256=C.sha(R/'QUERY_CALIBRATION_DATASET_FREEZE.json'),H_actual_FP32_sha256=hashlib.sha256(h.tobytes()).hexdigest(),H_source_sha256=C.sha(R/'inherited/group32/src/canonical_h128.py'),numerical_contract_sha256=C.sha(R/'EIGENBASIS_NUMERICAL_CONTRACT.json'),layers=layers,C1_and_S3_load_exact_same_files=True,orientation='A=U@H; row query/key transform x@A, state A.T@S; P only codec permutation',eigenvalue_order='descending; fixed canonical within frozen near-degenerate clusters',sign_convention='first max-abs component positive',AIME_used=False))
 C.atomic_new(R/'UHP_ROTATION_MATH_AUDIT.json',dict(status='PASS_FP64_EIGENBASIS_AND_FP32_QUERY_DIAGONAL',heads=len(diagnostics),diagnostics=diagnostics,negative_roundoff='Retained original FP64 eigenvalues, never silently clipped or adjusted',query_sensitivity_equalization_not_accuracy_guarantee=True))
except BaseException as exc:
 if not (R/'FIRST_FAILURE.json').exists():C.atomic_new(R/'FIRST_FAILURE.json',dict(status='QUERY_EIGENBASIS_QUALIFICATION_FAIL_CLOSED_STOP',error=repr(exc),traceback=traceback.format_exc(),unit='CPU_EIGENBASIS'))
 raise
