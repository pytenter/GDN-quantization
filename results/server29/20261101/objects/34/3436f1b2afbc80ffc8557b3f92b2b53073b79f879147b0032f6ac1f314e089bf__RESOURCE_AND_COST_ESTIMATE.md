# 实测单GPU资源条件预测

{
  "status": "ACTUAL_INT4_NONAIME_RATES_CONDITIONAL_NOT_GUARANTEE",
  "model_load_seconds": {
    "I4-N": 17.38650704920292,
    "I4-H": 17.645722404122353
  },
  "rates": {
    "I4-N": {
      "late": {
        "TPS": 9.204011385575214,
        "fixed_replay_input": true,
        "measured": 128,
        "ms_per_token": 108.64827933255583,
        "observer": false,
        "sampling_invoked": true,
        "seconds": 13.906979754567146,
        "warmup": 3968
      },
      "short": {
        "TPS": 9.2750759970314,
        "fixed_replay_input": true,
        "measured": 128,
        "ms_per_token": 107.81582817435265,
        "observer": false,
        "sampling_invoked": true,
        "seconds": 13.800426006317139,
        "warmup": 128
      },
      "start_128": 71192488.59677094,
      "start_3968": 71192909.05660029
    },
    "I4-H": {
      "late": {
        "TPS": 7.670258172899785,
        "fixed_replay_input": true,
        "measured": 128,
        "ms_per_token": 130.37370808888227,
        "observer": false,
        "sampling_invoked": true,
        "seconds": 16.68783463537693,
        "warmup": 3968
      },
      "short": {
        "TPS": 7.712665824587988,
        "fixed_replay_input": true,
        "measured": 128,
        "ms_per_token": 129.65685571543872,
        "observer": false,
        "sampling_invoked": true,
        "seconds": 16.596077531576157,
        "warmup": 128
      },
      "start_128": 71194560.2924427,
      "start_3968": 71195062.15584326
    }
  },
  "conditional_mean_tokens_scenarios": {
    "8192": {
      "single_GPU_generation_wall_hours_low": 16.211468553543092,
      "single_GPU_generation_wall_hours_high": 16.317234341303507,
      "GPUh_estimate_same_serial_intervals_not_exact": true
    },
    "20000": {
      "single_GPU_generation_wall_hours_low": 39.57878064829856,
      "single_GPU_generation_wall_hours_high": 39.836997903573014,
      "GPUh_estimate_same_serial_intervals_not_exact": true
    },
    "40000": {
      "single_GPU_generation_wall_hours_low": 79.15756129659712,
      "single_GPU_generation_wall_hours_high": 79.67399580714603,
      "GPUh_estimate_same_serial_intervals_not_exact": true
    },
    "81920": {
      "single_GPU_generation_wall_hours_low": 162.1146855354309,
      "single_GPU_generation_wall_hours_high": 163.17234341303507,
      "GPUh_estimate_same_serial_intervals_not_exact": true
    }
  },
  "qualifications_and_model_loading_seconds_recorded_in_ledger": true,
  "all60_cap_new_tokens": 4915200,
  "all60_cap_generation_hours": 163.17234341303507,
  "limitations": "Teacher4096 with original sampler, four short/late windows; future free-generation lengths and81K speed unknown. Qualification/loading/prompt/postflight/archive separate. Not a hard completion bound or exactphysicalGPUh."
}

同卡串行时，生成墙钟与生成占用GPU小时的估算数值接近；物理精确GPUh仍UNRESOLVED。实际加载、资格、退出释放见append-only账本，不能借用INT8吞吐当INT4实测。
