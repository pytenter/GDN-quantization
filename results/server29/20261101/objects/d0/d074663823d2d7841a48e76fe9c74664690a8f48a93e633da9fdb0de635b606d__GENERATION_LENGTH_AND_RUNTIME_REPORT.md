# 实测长度、EOS与运行时间

{
  "I4-N": {
    "N": 30,
    "total_tokens": 134918,
    "mean_tokens": 4497.266666666666,
    "median_tokens": 4376.5,
    "EOS": 30,
    "length": 0,
    "nonfinite_count": 0,
    "runtime_error_count": 0,
    "generation_wall_seconds": 15044.279722318053,
    "aggregate_generation_TPS": 8.968059786860408
  },
  "I4-H": {
    "N": 30,
    "total_tokens": 46504,
    "mean_tokens": 1550.1333333333334,
    "median_tokens": 1629.5,
    "EOS": 30,
    "length": 0,
    "nonfinite_count": 0,
    "runtime_error_count": 0,
    "generation_wall_seconds": 7013.37838357687,
    "aggregate_generation_TPS": 6.630755886335431
  }
}

含真实逐Prompt token与Decode QDQ/采样；加载、资格、退出释放成本另见GPU_RESOURCE_LEDGER.jsonl与RESOURCE_LEDGER.json。单卡串行；EXACT_GPUh=UNRESOLVED，不把保守区间当精确值。
