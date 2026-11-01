"""CPU-only append-only tensor-hash serialization audit; no model or GPU calls."""
from common import *
import argparse
import shutil
import torch

FOLDER=ROOT/'provenance/engineering_recovery_attempt3_serialization'

def zero():
    assert not list((ROOT/'runtime').glob('*launch.json'))
    assert not list((ROOT/'runtime').glob('*smoke.json'))
    assert not (ROOT/'run/training').exists() and not (ROOT/'run/fresh_b').exists()
    assert not torch.cuda.is_initialized()
    return dict(formal_optimizer_updates=0,disposable_optimizer_updates=0,
                Fresh_B_treatment_forwards=0,AIME_generation=0,SCIENTIFIC_GPU_FORWARD_COUNT=0)

def main(mode):
    counts=zero();identity=verify_inventory(ROOT/'provenance/v3_scientific_objects_before_engineering_recovery.json')
    source=ROOT/'scripts/contemporary_smoke.py'
    if mode=='before':
        try:tensor_sha(torch.tensor(1.))
        except RuntimeError as error:observed=repr(error)
        else:raise AssertionError('EXPECTED_SYNTHETIC_SERIALIZATION_DEFECT_ABSENT')
        snapshot=FOLDER/'source_contemporary_smoke.py';snapshot.parent.mkdir(parents=True,exist_ok=True)
        assert not snapshot.exists();shutil.copyfile(source,snapshot)
        emit(FOLDER/'BEFORE_REPAIR.json',dict(classification='PREMEASUREMENT_ENGINEERING_DEFECT',
            observed_exception=observed,root_cause='zero-dimensional tensor byte-view in audit serialization',
            allowed_change='flatten read-only byte-hash input; preserve recorded original shape/dtype and all numerical operations',
            source_sha256=sha(source),scientific_identity=identity,measurements=counts))
    else:
        before=load(FOLDER/'BEFORE_REPAIR.json')
        smoke=import_file(source,'cpu_only_smoke_serialization_test');tests=[]
        for x in [torch.tensor(1.),torch.tensor(1.,dtype=torch.bfloat16),torch.tensor(1,dtype=torch.int64),
                  torch.arange(6,dtype=torch.float32).reshape(2,3)]:
            saved=x.clone();r=smoke.safe_tensor(x)
            expected=hashlib.sha256(x.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            assert r==dict(shape=list(x.shape),dtype=str(x.dtype),sha256=expected) and torch.equal(x,saved)
            if x.ndim:assert r['sha256']==tensor_sha(x)
            tests.append(r)
        zero()
        emit(FOLDER/'AFTER_REPAIR.json',dict(gate='PASS',old_source_sha256=before['source_sha256'],
            new_source_sha256=sha(source),function='safe_tensor',tests=tests,scientific_identity=identity,
            SCIENTIFIC_OBJECT_IDENTITY_AFTER_ENGINEERING_FIX='PASS',measurements=counts,
            tensor_values_changed=False,shape_dtype_changed=False,numerical_operations_changed=False))
        base=load(ROOT/'provenance/PREMEASUREMENT_ENGINEERING_RECOVERY_ENVELOPE_V1.json')
        emit(ROOT/'provenance/ENGINEERING_RECOVERY_CLOSURE.json',dict(gate='PASS',
            base_envelope_sha256=sha(ROOT/'provenance/PREMEASUREMENT_ENGINEERING_RECOVERY_ENVELOPE_V1.json'),
            ENGINEERING_RECOVERY_ATTEMPTS=base['ENGINEERING_RECOVERY_ATTEMPTS']+1,
            CURRENT_SUCCESSFUL_PREPARATION_ATTEMPT=base['CURRENT_SUCCESSFUL_PREPARATION_ATTEMPT'],
            serialization_receipt_sha256=sha(FOLDER/'AFTER_REPAIR.json'),
            SCIENTIFIC_OBJECT_IDENTITY_ACROSS_RECOVERY='PASS',SCIENTIFIC_PROTOCOL_CHANGED='NO',measurements=counts))
    print('CPU_SERIALIZATION_AUDIT_'+mode.upper()+'=PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['before','after']);main(p.parse_args().mode)
