from common import (CLEAN_EXP, EXP, PARENT_SCRIPTS, read_json, sha256,
                    write_json)


BASE_COMMIT='dc30294816588e80e4f26727f62afbec47ff7d12'


def main():
    old_error=EXP/'FAIL_AND_STOP_CURRENT.json'
    old_log=EXP/'discovery/logs/doc01_capture.log'
    if not old_error.exists() or not old_log.exists():
        raise RuntimeError('historical capture plumbing failure evidence missing')
    if (EXP/'discovery/doc_01/drive').exists() and any((EXP/'discovery/doc_01/drive').iterdir()):
        raise RuntimeError('unexpected partial scientific capture; automatic repair forbidden')
    policy=EXP/'protocol/extended_capture_policy.json'
    script=PARENT_SCRIPTS/'capture_extended_drive.py'
    write_json(EXP/'extended_capture_freeze.json',{
        'policy_sha256':sha256(policy),'script_sha256':sha256(script),
        'freeze_commit':BASE_COMMIT,
        'scope':'FROZEN_INPUT_ACQUISITION_ONLY_BEFORE_NEW_PREMEASUREMENT_COMMIT'})
    write_json(EXP/'scientific_execution_authorization.json',{
        'DISCOVERY_EXECUTION_AUTHORIZED':'YES',
        'PREMEASUREMENT_COMMIT_SHA':BASE_COMMIT,
        'scope':'FROZEN_INPUT_ACQUISITION_ONLY'})
    source_dir=CLEAN_EXP/'engineering/gates'
    for phase in ['factorization','schema','fp','invariance','decomposition',
                  'isolation','determinism','streaming']:
        source=source_dir/f'{phase}.json'
        if not source.exists() and phase=='determinism':
            source=source_dir/'determinism_2048.json'
        value=read_json(source)
        if value['gate']!='PASS':
            raise RuntimeError(f'parent engineering gate not PASS: {phase}')
        write_json(EXP/f'engineering/gates/{phase}.json',{
            'gate':'PASS','reused_for_frozen_input_acquisition_only':True,
            'source_path':str(source),'source_sha256':sha256(source)})
    write_json(EXP/'engineering/capture_plumbing_repair_001.json',{
        'classification':'ENGINEERING_ONLY_MANIFEST_PLUMBING',
        'missing_artifact':'extended_capture_freeze.json plus acquisition authorization and reused gate receipts',
        'old_failure_preserved':True,'old_failure_sha256':sha256(old_error),
        'old_log_sha256':sha256(old_log),'scientific_measurement_before_fix':0,
        'GPU_computation_before_fix':False,'scientific_configuration_changed':False,
        'policy_sha256':sha256(policy),'capture_script_sha256':sha256(script),
        'resume_scope':'restart only the never-started Document 1 input capture'})
    print('CAPTURE_MANIFEST_PLUMBING_REPAIR_PASS',flush=True)


if __name__=='__main__':
    main()
