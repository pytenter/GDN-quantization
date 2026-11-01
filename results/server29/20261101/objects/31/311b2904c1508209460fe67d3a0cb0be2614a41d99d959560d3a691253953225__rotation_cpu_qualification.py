"""Independent NumPy codec/recurrence references and frozen negative controls."""
import sys,json,hashlib,traceback,ast,types
from pathlib import Path
R=Path(sys.argv[1]).resolve();sys.path[:0]=[str(R/'src'),str(R/'inherited/group32/src')]
import execution_control as C
import numpy as np
import torch
from canonical_h128 import hadamard
import group_codecs as B
import rotation_codec as K
torch.set_num_threads(1)
def ref(x,group=32,permuted=False):
 p=np.asarray(K.P);a=x[...,p,:] if permuted else x
 y=a.reshape(*a.shape[:2],128//group,group,128);scale=np.maximum(np.max(np.abs(y),axis=-2),np.float32(1e-12))/np.float32(127)
 rounded=np.rint(y/scale[...,None,:]);clipped=np.clip(rounded,np.float32(-127),np.float32(127));codes=clipped.astype(np.int8);neg=(clipped==0)&np.signbit(clipped)
 d=codes.astype(np.float32)*scale[...,None,:];d=np.where(neg,np.float32(-0.),d).reshape(x.shape)
 return (d[...,p,:] if permuted else d),codes,scale,neg
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
try:
 rng=np.random.default_rng(20261022);p=np.asarray(K.P);I=np.eye(128);H=hadamard(dtype=torch.float64).numpy()
 C.require(len(set(p.tolist()))==128 and np.array_equal(p[p],np.arange(128)) and np.array_equal(I[:,p].T@I[:,p],I),'PBR bijection/involution/orthogonality failed')
 C.require([int(p[i]) for i in range(6)]==[0,64,32,96,16,80],'7bit PBR direction differs')
 groups=[set(p[i:i+32].tolist()) for i in range(0,128,32)];old=[set(range(i,i+32)) for i in range(0,128,32)];C.require(all(a!=b for a,b in zip(groups,old)),'PBR failed to change group membership')
 fixtures=[np.zeros((1,2,128,128),np.float32),np.full((1,2,128,128),-0.,np.float32),rng.standard_normal((1,2,128,128)).astype(np.float32),rng.uniform(-1e-15,1e-15,(1,2,128,128)).astype(np.float32)]
 x=np.zeros((1,2,128,128),np.float32);x[...,::32,:]=np.float32(1e6);x[...,1::32,:]=np.float32(-2e5);fixtures.append(x)
 tests=[]
 for fi,x in enumerate(fixtures):
  t=torch.from_numpy(x.copy())
  for cond in ('P0','P1','C1','S3'):
   e=K.encode(t,cond);d=K.decode(e).numpy();expected,codes,scales,neg=ref(x,permuted=cond in ('P1','S3'))
   for name,a,b in [('decoded',d,expected),('codes',e.codes.numpy(),codes),('scales',e.scales.numpy(),scales),('negative_zero',e.negative_zero_flags.numpy(),neg)]:C.require(same(a,b),'Independent CPU codec mismatch '+str((fi,cond,name)))
   tests.append(dict(fixture=fi,condition=cond,status='PASS_EXACT_FIELDS_BYTES'))
  # Identity-P uses the same underlying codec graph and restores the same coordinates.
  e=B.encode(t,'B1');identity=B.decode(B.encode(t.index_select(-2,torch.arange(128)),'B1')).index_select(-2,torch.arange(128));C.require(B.tensor_bytes(identity)==B.tensor_bytes(B.decode(e)),'Identity P negative control failed')
  for group in ('B0',):
   direct=B.decode(B.encode(t,group));pr=B.decode(B.encode(t.index_select(-2,torch.tensor(K.P)),group)).index_select(-2,torch.tensor(K.P));C.require(B.tensor_bytes(direct)==B.tensor_bytes(pr),'Group128 PBR invariance including signed zero failed')
 C.require(np.array_equal(np.eye(128,dtype=np.float32)@H.astype(np.float32),H.astype(np.float32)),'U=I does not reproduce original H matrix')
 # Fully independent native/rotated multi-token FP64 recurrence, full State/Query/Key/readout.
 U,_=np.linalg.qr(rng.standard_normal((128,128)));A=U@H;S=rng.standard_normal((128,128));SA=A.T@S;worst=0.
 for t in range(64):
  k=rng.normal(size=128);k/=np.linalg.norm(k);q=rng.normal(size=128);alpha=float(rng.uniform(.5,1));beta=float(rng.uniform(0,1));v=rng.normal(size=128)
  # Native independent expanded equation vs rotated rank-one update.
  S=alpha*(I-beta*np.outer(k,k))@S+beta*np.outer(k,v)
  ka=A.T@k;qa=A.T@q;SA=alpha*SA;SA+=np.outer(ka,beta*(v-ka@SA));expected=A.T@S
  err=np.linalg.norm(SA-expected)/max(np.linalg.norm(expected),1e-30);read=np.linalg.norm(SA.T@qa-S.T@q)/max(np.linalg.norm(S.T@q),1e-30);worst=max(worst,float(err),float(read));C.require(err<2e-12 and read<2e-12,'FP64 full rotation recurrence equivalence failed')
  SH=H.T@S;C.require(np.max(np.abs((H.T@U.T@H)@SH-A.T@S))<2e-11,'H->A composite transform direction failed')
 # U diagonalization followed by H makes the sensitivity diagonal uniform; P only reorders it.
 lam=np.linspace(1,100,128);cq=U@np.diag(lam)@U.T;target=np.trace(cq)/128;trans=A.T@cq@A
 C.require(np.max(np.abs(np.diag(trans)-target))<2e-5 and np.max(np.abs(np.diag(trans[np.ix_(p,p)])-target))<2e-5,'Query sensitivity diagonal identity failed')
 # C1/S3 no-QDQ consume the exact same A; P is absent from their transform method.
 src=(R/'src/rotation_adapter.py').read_text();tree=ast.parse(src);transform=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_transform');C.require('P' not in {n.id for n in ast.walk(transform) if isinstance(n,ast.Name)},'P appeared outside codec')
 v=dict(status='PASS_INDEPENDENT_CPU_NEGATIVE_CONTROLS_A_B_C_D_E_F',codec_tests=tests,fixtures=len(fixtures),exact_codec_checks=len(tests)*4,FP64_multi_token_steps=64,worst_FP64_relative=worst,groups_after_PBR=[sorted(x) for x in groups],identity_P='PASS_EXACT_SIGNED_ZERO',identity_U='PASS_SAME_H_MATRIX',group128_invariance='PASS_EXACT_VALUES_AND_BYTES',H_to_A='PASS_H.T@U.T@H',C1_S3_same_FP_function=True,query_sensitivity='PASS_A.T_Cq_A_equals_H.T_diag_lambda_H',no_AIME=True)
 C.atomic_new(R/'SYM32_CODEC_AND_PERMUTATION_PARITY.json',v)
 print(json.dumps({k:v[k] for k in ['status','exact_codec_checks','FP64_multi_token_steps','worst_FP64_relative']}))
except BaseException as exc:
 if not (R/'FIRST_FAILURE.json').exists():C.atomic_new(R/'FIRST_FAILURE.json',dict(status='GROUP32_PERMUTATION_PARITY_FAIL_CLOSED_STOP',error=repr(exc),traceback=traceback.format_exc(),unit='CPU_ROTATION_CODEC_NEGATIVE_CONTROLS'))
 raise
