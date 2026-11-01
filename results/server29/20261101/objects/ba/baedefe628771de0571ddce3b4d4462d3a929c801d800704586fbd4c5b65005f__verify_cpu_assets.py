"""Independent CPU verification of all sealed unique-head matrices and PBR math."""
import sys,hashlib,json,math
from pathlib import Path
R=Path(sys.argv[1]).resolve();sys.path[:0]=[str(R/'src'),str(R/'inherited/group32/src')]
import execution_control as C
import numpy as np
import torch
import group_codecs as B
import rotation_codec as K
from canonical_h128 import hadamard
torch.set_num_threads(1)
out=Path(sys.argv[2]);checks=[];rng=np.random.default_rng(20261022)
# Independently slice actual Key groups and preserve every original signed-zero bit.
for fixture,x in enumerate([np.zeros((1,2,128,128),np.float32),np.full((1,2,128,128),-0.,np.float32),rng.normal(size=(1,2,128,128)).astype(np.float32),rng.uniform(-1e-15,1e-15,(1,2,128,128)).astype(np.float32)]):
 for cond in ('P1','C1','S3'):
  inp=x[...,list(K.P),:] if cond in ('P1','S3') else x;decoded=np.empty_like(inp);expected_codes=np.empty((1,2,4,32,128),np.int8);expected_scale=np.empty((1,2,4,128),np.float32);negative=np.empty((1,2,4,32,128),bool)
  for g in range(4):
   values=inp[...,g*32:(g+1)*32,:];scale=np.maximum(np.abs(values).max(-2),np.float32(1e-12))/np.float32(127);rounds=np.rint(values/scale[...,None,:]);clip=np.clip(rounds,-127,127);codes=clip.astype(np.int8);neg=(clip==0)&np.signbit(clip);d=codes.astype(np.float32)*scale[...,None,:];decoded[...,g*32:(g+1)*32,:]=np.where(neg,np.float32(-0.),d);expected_codes[:,:,g]=codes;expected_scale[:,:,g]=scale;negative[:,:,g]=neg
  if cond in ('P1','S3'):decoded=decoded[...,list(K.P),:]
  e=K.encode(torch.from_numpy(x.copy()),cond)
  for name,got,want in [('decoded',K.decode(e).numpy(),decoded),('codes',e.codes.numpy(),expected_codes),('scale',e.scales.numpy(),expected_scale),('negative_zero',e.negative_zero_flags.numpy(),negative)]:C.require(got.dtype==want.dtype and got.shape==want.shape and got.tobytes()==want.tobytes(),'Independent CPU PBR field mismatch:'+str((fixture,cond,name)))
  checks.append(dict(fixture=fixture,condition=cond,fields=4,status='PASS_EXACT'))
manifest=C.read(R/'CANONICAL_UH_MATRICES_AND_SHA.json');attribution=C.read(R/'FP32_M0_M3_NUMERICAL_ATTRIBUTION.json');matrix_checks=[];H=hadamard(dtype=torch.float64).numpy();mu_rows={(r['layer'],r['unique_query_head']):r for r in attribution['rows']}
with np.load(R/'reuse/QUERY_SECOND_MOMENTS_FP64.npz') as cq:
 for row in manifest['layers']:
  p=R/row['path'];C.require(C.sha(p)==row['sha256'],'Sealed canonical matrix file changed')
  with np.load(p) as z:
   for h in range(16):
    U=z['U_fp64_unique'][h];A=z['A_fp32_unique'][h];eigs=z['eigenvalues_unique'][h];c=cq[str(row['layer'])][2*h];principal=(U@H).astype(np.float32);C.require(A.tobytes()==principal.tobytes(),'A32 not single whole-matrix cast of canonical A64');C.require(np.array_equal(z['recurrent_to_unique'],np.arange(32)//2),'State-head mapping differs');C.require(hashlib.sha256(A.tobytes()).hexdigest()==row['head_matrices'][h]['A32_sha256'],'Per-head A identity differs')
    # Independent per-column quadratic forms, rather than the attribution GEMM diagonal.
    diagonal=np.array([np.einsum('i,ij,j->',A[:,j].astype(np.float64),c,A[:,j].astype(np.float64),optimize=False) for j in range(128)]);r=mu_rows[(row['layer'],h)];absolute=float(np.max(np.abs(diagonal-r['mu'])));C.require(absolute<=r['M0_absolute']+r['representation_budget_absolute']+r['FP64_evaluation_noise_budget']*(r['mu'] if r['mode'].startswith('relative') else 1),'Independent diagonal functional error budget failed')
    matrix_checks.append(dict(layer=row['layer'],unique_head=h,status='PASS_INDEPENDENT_COLUMN_QUADRATIC_FORMS',absolute_diagonal_error=absolute))
C.require(len(matrix_checks)==384,'Incomplete independent all-head verification')
# Independent FP64 covariance/rotation identity over many recurrent steps.
Q,_=np.linalg.qr(rng.normal(size=(128,128)));A=Q@H;S=rng.normal(size=(128,128));SA=A.T@S;worst=0.
for t in range(64):
 k=rng.normal(size=128);k/=np.linalg.norm(k);q=rng.normal(size=128);v=rng.normal(size=128);alpha=float(rng.uniform(.5,1));beta=float(rng.uniform(0,1));S=alpha*(np.eye(128)-beta*np.outer(k,k))@S+beta*np.outer(k,v);ka=A.T@k;qa=A.T@q;SA*=alpha;SA+=np.outer(ka,beta*(v-ka@SA));state=np.linalg.norm(SA-A.T@S)/np.linalg.norm(S);read=np.linalg.norm(SA.T@qa-S.T@q)/np.linalg.norm(S.T@q);worst=max(worst,float(state),float(read));C.require(state<2e-12 and read<2e-12,'Independent multi-token FP64 rotation recurrence failed')
C.atomic_new(out,dict(status='PASS_INDEPENDENT_CPU_ASSETS_AND_CODEC',canonical_manifest_sha256=C.sha(R/'CANONICAL_UH_MATRICES_AND_SHA.json'),contract_sha256=C.sha(R/'V2_NUMERICAL_QUALIFICATION_CONTRACT.json'),matrix_heads=384,matrix_checks=matrix_checks,codec_checks=checks,FP64_steps=64,FP64_worst_relative=worst,GPU_dispatches=0))
print('PASS independent CPU384head assets,48codec fields,64step recurrence',flush=True)
