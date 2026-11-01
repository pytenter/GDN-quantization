# dP32 数值与生命周期验证

[
  {
    "path": "validation/DP32_FRESH1.json",
    "status": "PASS_DP32_ORIGINAL_NUMERICAL_GATES",
    "worst_NRMSE": {
      "output": 0.0031871852774899593,
      "delta": 0.0005101615715249476,
      "boundary_state": 0.0003446360899604491,
      "actual_delta_state_readout": 0.0004748053641599006
    },
    "same_base_worst_NRMSE": {
      "same_base_output": 0.0006774473667646323,
      "same_base_delta": 1.4999283262811912e-07,
      "same_base_boundary": 1.1674895185491193e-07
    },
    "local_unexplained": 0,
    "cases": 24
  },
  {
    "path": "validation/DP32_FRESH2.json",
    "status": "PASS_DP32_ORIGINAL_NUMERICAL_GATES",
    "worst_NRMSE": {
      "output": 0.0031871852774899593,
      "delta": 0.0005101615715249476,
      "boundary_state": 0.0003446360899604491,
      "actual_delta_state_readout": 0.0004748053641599006
    },
    "same_base_worst_NRMSE": {
      "same_base_output": 0.0006774473667646323,
      "same_base_delta": 1.4999283262811912e-07,
      "same_base_boundary": 1.1674895185491193e-07
    },
    "local_unexplained": 0,
    "cases": 24
  }
]

沿用原0.5%跨 replay、0.1%同 actual-state 读出的工程门禁，不改阈值。覆盖31/32/33、63/64/65、prompt_end、未满窗口preserve、强弱衰减、exponent、实际32 heads及257步弱衰减；原跨轨迹局部标签保留，同窗口base反事实单独解释继承分叉。验证使用dense reference，吞吐路径的capture_validation=False。验证失败后没有启动性能测试。原代码和旧正式输出均未修改。
