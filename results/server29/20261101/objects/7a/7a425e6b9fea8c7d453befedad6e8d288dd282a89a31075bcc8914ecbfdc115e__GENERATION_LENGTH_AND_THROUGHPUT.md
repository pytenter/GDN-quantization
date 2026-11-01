# 实际生成长度与吞吐

{
  "Q0": {
    "total_tokens": 1331923,
    "EOS": 22,
    "length": 8,
    "seconds": 135797.15566942096,
    "aggregate_generation_TPS": 9.808180395489128
  },
  "Q1": {
    "total_tokens": 1204813,
    "EOS": 27,
    "length": 3,
    "seconds": 130545.25699235499,
    "aggregate_generation_TPS": 9.229082907780835
  },
  "Q2": {
    "total_tokens": 1087356,
    "EOS": 28,
    "length": 2,
    "seconds": 136551.78850834072,
    "aggregate_generation_TPS": 7.962956852327007
  },
  "Q3": {
    "total_tokens": 897357,
    "EOS": 29,
    "length": 1,
    "seconds": 119131.51915407181,
    "aggregate_generation_TPS": 7.532490195474261
  }
}

含各样本Prompt/采样/QDQ；按问题长度和GPU分层，不把不同题目构成解释为设备速度差。模型加载计入资源账本，未计入此生成循环TPS。
