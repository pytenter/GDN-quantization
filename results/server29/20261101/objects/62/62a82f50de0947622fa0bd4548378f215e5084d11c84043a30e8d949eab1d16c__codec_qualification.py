"""Exact independent CPU reference vs production CPU/CUDA fields, no AIME."""
import json, hashlib, sys
from pathlib import Path
import numpy as np
import torch
from int4_codec import encode, decode, tensor_bytes, require_consumed
from int4_reference import reference, fixtures, rejection_fixtures

def qualify(target,device):
    target=Path(target);target.mkdir(parents=True,exist_ok=True)
    checks=[]
    for name,x in fixtures():
        expected=reference(x)
        e,actual=encode(torch.from_numpy(x.copy()).to(device),debug=True)
        for field,want in expected.items():
            got=actual[field].detach().cpu().numpy()
            if got.dtype!=want.dtype or got.shape!=want.shape or got.tobytes()!=want.tobytes():
                # Preserve all original input and every stage, plus the first real32-value group.
                np.savez(target/'FIRST_NUMERICAL_DIFFERENCE.npz',input=x,
                         **{'expected_'+k:v for k,v in expected.items()},
                         **{'actual_'+k:v.detach().cpu().numpy() for k,v in actual.items()})
                differing=np.argwhere(got.view(np.uint8).reshape(got.shape+(-1,))!=want.view(np.uint8).reshape(want.shape+(-1,)))
                idx=tuple(int(v) for v in differing[0][:-1])
                b,h=idx[:2];g=idx[2]//32 if field=='decoded' else idx[2];v=idx[-1]
                diagnostic=dict(status='FAIL_EXACT_FIELD_BYTES',fixture=name,field=field,index=idx,
                                first_input_group_fp32_bits=x[b,h,g*32:(g+1)*32,v].view(np.uint32).tolist(),
                                expected_dtype=str(want.dtype),actual_dtype=str(got.dtype),threshold_relaxed=False)
                (target/'FIRST_NUMERICAL_DIFFERENCE.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
                raise RuntimeError('Independent exact field differs:'+name+':'+field)
        assert e.codes.dtype==torch.uint8 and e.scales.dtype==torch.float32 and e.zero_points.dtype==torch.uint8
        assert bool(((e.codes>=0)&(e.codes<=15)).all()) and bool(((e.zero_points>=0)&(e.zero_points<=15)).all())
        require_consumed(decode(e),e.cpu())
        checks.append(dict(fixture=name,fields_exact=list(expected),input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                           floor_groups=int((expected['span']<np.float32(1e-12)).sum()),
                           clipped_values=int(((expected['shifted']<0)|(expected['shifted']>15)).sum()),
                           codes_at_0=int((expected['codes']==0).sum()),codes_at_15=int((expected['codes']==15).sum()),
                           decoded_sha256=hashlib.sha256(expected['decoded'].tobytes()).hexdigest()))
        if name=='TIES_ZP_ODD':
            alternate=np.rint(expected['levels']+expected['zero_points'][...,None,:].astype(np.float32)).clip(0,15).astype(np.uint8)
            assert alternate.tobytes()!=expected['codes'].tobytes(),'Tie-order fixture failed to distinguish formulas'
    rejected=[]
    for name,x in rejection_fixtures():
        for label,fn in [('independent',lambda:reference(x)),('production',lambda:encode(torch.from_numpy(x.copy()).to(device)))]:
            try:fn()
            except FloatingPointError:pass
            else:raise RuntimeError('Illegal extreme/nonfinite accepted:'+name+':'+label)
        rejected.append(name)
    # Independently visible axis-negative case and widened unsigned subtraction.
    x=np.empty((1,1,128,128),np.float32)
    for k in range(128):
        for v in range(128):x[0,0,k,v]=(1+v/128)*(1+k//32)*(-1 if k%2 else 1)
    t=torch.from_numpy(x).to(device);proper=decode(encode(t));wrong=decode(encode(t.transpose(-1,-2))).transpose(-1,-2)
    assert tensor_bytes(proper)!=tensor_bytes(wrong)
    assert int(torch.tensor([0],device=device,dtype=torch.uint8).to(torch.int16)[0]-torch.tensor([15],device=device,dtype=torch.uint8).to(torch.int16)[0])==-15
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'inherited/group32/src'))
    from canonical_h128 import hadamard
    h=hadamard(128,device=device);identity=torch.eye(128,device=device)
    error=float((h.T@h-identity).abs().max());assert error<2e-6
    q=torch.from_numpy(np.random.default_rng(108).normal(size=(1,1,128,128)).astype(np.float32)).to(device)
    inverse_error=float((h.T@(h@q)-q).abs().max());assert inverse_error<5e-6
    rotated=h.T@q;_,fields=encode(rotated,debug=True);ref=reference(rotated.detach().cpu().numpy())
    assert all(tensor_bytes(fields[k])==v.tobytes() for k,v in ref.items())
    return dict(status='PASS_INDEPENDENT_EXACT_INT4_CODEC',device=str(device),cases=checks,rejected=rejected,
                codes_dtype='UINT8_0_TO_15',scale_dtype='FP32',zero_point_dtype='UINT8_INTEGER',group_axis='Key',group=32,
                rounding='RNE(x/s)+integer_z',range_floor=1e-12,effective_scale_floor=float(np.float32(np.float32(1e-12)/np.float32(15))),
                all_contracted_field_tolerances=0,H_original_sha256=hashlib.sha256(tensor_bytes(h)).hexdigest(),
                H_orthogonality_max_abs=error,H_inverse_max_abs=inverse_error,rotation_tolerances_inherited_unchanged=True,
                persistent_carrier='FP32_decoded_QDQ',physical_packed_INT4=False,AIME_used=False)

if __name__=='__main__':
    assert len(sys.argv)==3
    print('RESULT_JSON='+json.dumps(qualify(sys.argv[1],sys.argv[2]),sort_keys=True))
