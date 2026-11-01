"""V2 manifest comparison, CPU-only, no model weights or forwards."""
from common import *
import argparse

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,required=True);args=parser.parse_args()
    assert os.environ.get('PYTHONNOUSERSITE')=='1'
    assert Path(sys.executable).resolve()==PYTHON.resolve()
    manifest=load(V2/'runtime/cleanroom_environment_manifest.json');assert manifest['gate']=='PASS'
    original=load(V2/'runtime/cleanroom_config_observation.json')
    p=import_file(V2/'scripts/runtime_probe.py','immutable_v2_config_probe')
    output=ROOT/f'runtime/preparation_attempt{args.attempt}/current_config_observation.json'
    output.parent.mkdir(parents=True,exist_ok=True);assert not output.exists()
    sys.argv=[str(V2/'scripts/runtime_probe.py'),str(MODEL),str(output)]
    p.main();current=load(output)
    fields=['python','pip_freeze','cuda_runtime','model_files','raw_config','effective_config','config_mutations','conda_history_sha256']
    comparisons={k:original[k]==current[k] for k in fields}
    for name in original['packages']:
        comparisons['package_'+name]=original['packages'][name]==current['packages'][name]
    assert all(comparisons.values()),comparisons
    assert current['pip_freeze']==manifest['pip_freeze']
    emit(ROOT/'runtime/v3_runtime_identity.json',dict(gate='PASS',V3_RUNTIME_IDENTITY_GATE='PASS',
        environment_reused=str(PYTHON.parent.parent),V2_manifest_sha256=sha(V2/'runtime/cleanroom_environment_manifest.json'),
        V2_observation_sha256=sha(V2/'runtime/cleanroom_config_observation.json'),
        current_observation_sha256=sha(output),comparison=comparisons,
        current_observation_path=str(output),frozen_launch_profile=dict(PYTHONNOUSERSITE='1'),
        recorded_byte_inventory='all eight recorded package Python source trees, module bytes, model/config/tokenizer bytes, conda history; exact package/pip inventory',
        unrecorded_historical_binary_bytes='V2 did not record an exhaustive ELF/.so byte inventory; no claim beyond V2 manifest fields',
        versions={n:v['version'] for n,v in current['packages'].items()},model_config='raw V2 reconstructed exact semantic config',
        config_mutations=[],historical_output_comparison=False))
    print('V3_RUNTIME_IDENTITY_GATE=PASS',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:stop('RUNTIME_IDENTITY',e);raise
