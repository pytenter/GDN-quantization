"""Append-only engineering failure classification before any launch-profile repair."""
from common import *
import shutil

def main():
    assert not list((ROOT/'runtime').glob('*smoke.json'))
    assert not (ROOT/'run/training').exists() and not (ROOT/'run/fresh_b').exists()
    identity=verify_inventory(ROOT/'provenance/v3_scientific_objects_before_engineering_recovery.json')
    a=load(V2/'runtime/cleanroom_config_observation.json');b=load(ROOT/'runtime/current_config_observation.json')
    assert a['packages']==b['packages'] and a['effective_config']==b['effective_config'] and a['model_files']==b['model_files']
    assert a['environment_variables']['PYTHONNOUSERSITE']=='1' and b['environment_variables']['PYTHONNOUSERSITE'] is None
    assert not (set(a['pip_freeze'].splitlines())-set(b['pip_freeze'].splitlines()))
    folder=ROOT/'provenance/preparation_attempts/attempt2'
    for name in ['prepare_phase_a.py','verify_runtime.py']:
        dest=folder/('source_'+name);assert not dest.exists();shutil.copyfile(ROOT/'scripts'/name,dest)
    emit(folder/'ENGINEERING_DEFECT_CLASSIFICATION.json',dict(gate='PASS',classification='PREMEASUREMENT_ENGINEERING_DEFECT',
        root_cause='CLI launch profile omitted existing frozen V2 PYTHONNOUSERSITE=1; user-site packages leaked into pip inventory',
        V3_PREPARATION_ATTEMPT_2='FAIL_ENGINEERING_RUNTIME_LAUNCH_PROFILE',
        private_core_package_bytes_unchanged=True,V2_environment_modified=False,package_repairs=0,
        scientific_objects=identity,SCIENTIFIC_GPU_FORWARD_COUNT=0,formal_optimizer_updates=0,
        disposable_optimizer_updates=0,Fresh_B_treatment_forwards=0,AIME_generation=0,
        V2_environment_variables=a['environment_variables'],attempt2_environment_variables=b['environment_variables'],
        extra_user_site_inventory=sorted(set(b['pip_freeze'].splitlines())-set(a['pip_freeze'].splitlines())),
        preserved_failed_observation=str(ROOT/'runtime/current_config_observation.json'),
        failed_observation_sha256=sha(ROOT/'runtime/current_config_observation.json'),
        allowed_repair='restore exact frozen V2 launch profile; no pip/environment/scientific runtime repair',
        repeated_root_cause=False,engineering_recovery_loop_detected=False))
    print('ATTEMPT2_CLASSIFIED_ENGINEERING_LAUNCH_PROFILE; SCIENTIFIC_OBJECT_IDENTITY=PASS; GPU_FORWARD=0',flush=True)

if __name__=='__main__':main()
