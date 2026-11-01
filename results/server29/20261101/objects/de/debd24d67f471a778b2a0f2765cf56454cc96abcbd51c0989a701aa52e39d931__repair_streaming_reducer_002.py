from common import EXP, sha256, write_json


def main():
    error=EXP/'discovery/logs/qualification_rank1.error.json'
    group=EXP/'execution/qualification_doc01_run1.json'
    if not error.exists() or not group.exists():
        raise RuntimeError('historical deterministic histc failure evidence missing')
    write_json(EXP/'engineering/streaming_reducer_repair_002.json',{
        'classification':'ENGINEERING_ONLY_STREAMING_REDUCER_IMPLEMENTATION',
        'failure':'CUDA torch.histc has no deterministic implementation',
        'repair':'Transfer already-computed signed ratios to CPU and execute the identical frozen-range/frozen-bin torch.histc reducer on CPU.',
        'scientific_trajectory_math_changed':False,
        'intervention_changed':False,'quantizer_changed':False,
        'histogram_range_changed':False,'histogram_bin_count_changed':False,
        'deterministic_algorithms_remain_enabled':True,'TF32_remains_disabled':True,
        'failed_qualification_preserved':True,
        'failed_rank_error_sha256':sha256(error),
        'failed_group_record_sha256':sha256(group),
        'formal_measurements_before_fix':0,
        'resume_scope':'qualification only; reuse sealed exact drive and qualified anchors'})
    print('STREAMING_REDUCER_CPU_HIST_REPAIR_RECORDED',flush=True)


if __name__=='__main__':
    main()
