# Qwen/GDN Asymmetric INT4 Group32 AIME26 Seed1

状态：FORMAL_COMPLETE_60_60_AND_STOP。60/60条合法生成；原FrozenV4与两个独立CPU重评分完全一致；自身进程已退出且NVML已释放。

| Codec | Correct/30 | Accuracy |
|---|---:|---:|
| I4-N | 0/30 | 0.00% |
| I4-H | 0/30 | 0.00% |

配对结果：

{
  "both_correct": 0,
  "both_wrong": 30,
  "Native_only": 0,
  "Hadamard_only": 0,
  "net_correct_H_minus_N": 0,
  "delta_accuracy": 0.0
}

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

{
  "status": "PARTIAL_EXPLORATORY",
  "N": 28,
  "problem_ids": [
    "aime26_01",
    "aime26_02",
    "aime26_03",
    "aime26_04",
    "aime26_05",
    "aime26_06",
    "aime26_07",
    "aime26_08",
    "aime26_09",
    "aime26_10",
    "aime26_11",
    "aime26_12",
    "aime26_13",
    "aime26_14",
    "aime26_15",
    "aime26_16",
    "aime26_17",
    "aime26_18",
    "aime26_19",
    "aime26_20",
    "aime26_21",
    "aime26_22",
    "aime26_23",
    "aime26_24",
    "aime26_25",
    "aime26_26",
    "aime26_27",
    "aime26_30"
  ],
  "correct_counts": {
    "I4-N": 0,
    "I4-H": 0,
    "Q1": 24,
    "Q3": 24
  },
  "deltas": {
    "native_4_minus_8": -0.8571428571428571,
    "H_4_minus_8": -0.8571428571428571,
    "H_minus_N_INT4": 0.0,
    "interaction": 0.0
  },
  "protocols_equal": true,
  "cross_UUID_questions": [
    "aime26_02",
    "aime26_03",
    "aime26_04",
    "aime26_05",
    "aime26_06",
    "aime26_07",
    "aime26_09",
    "aime26_11",
    "aime26_12",
    "aime26_13",
    "aime26_14",
    "aime26_15",
    "aime26_16",
    "aime26_17",
    "aime26_18",
    "aime26_20",
    "aime26_21",
    "aime26_23",
    "aime26_24",
    "aime26_26",
    "aime26_30"
  ],
  "cross_device_bitwise_claim": false,
  "historical_test_observed": true,
  "matched_generation_lengths": {
    "I4-N": {
      "N": 28,
      "mean_tokens": 4479.142857142857,
      "median_tokens": 4360.0,
      "EOS": 28,
      "length": 0,
      "nonfinite_count": 0,
      "runtime_error_count": 0
    },
    "I4-H": {
      "N": 28,
      "mean_tokens": 1545.857142857143,
      "median_tokens": 1629.5,
      "EOS": 28,
      "length": 0,
      "nonfinite_count": 0,
      "runtime_error_count": 0
    },
    "Q1": {
      "N": 28,
      "mean_tokens": 38546.32142857143,
      "median_tokens": 34463.0,
      "EOS": 26,
      "length": 2,
      "nonfinite_count": 0,
      "runtime_error_count": 0
    },
    "Q3": {
      "N": 28,
      "mean_tokens": 26870.214285714286,
      "median_tokens": 24126.0,
      "EOS": 28,
      "length": 0,
      "nonfinite_count": 0,
      "runtime_error_count": 0
    }
  },
  "paired_bitwidth_changes": {
    "I4-N-Q1": {
      "rescue": 0,
      "lost": 24,
      "net": -24
    },
    "I4-H-Q3": {
      "rescue": 0,
      "lost": 24,
      "net": -24
    }
  }
}

I4-N与I4-H只在Native/H128坐标路径存在预期差别；同一个独立Canonical AsymINT4Key32。H矩阵/符号/方向与原Q/K和递归数学保持冻结历史实现。FP32scale、integerzero-point、RNE(x/s)+z、范围0..15、包括0极值、Key轴4x32分组、int16扩展后减z、即时FP32重建写回。rangefloor1e-12先于除15，有效scale下限EPS/15，详见数值合同，与INT8EPS/255的位宽改变显式报告。

逐Prompt token从token0执行QDQ，q_length1，24层每次写回；p+n-1实际forward与24(p+n-1)写回，最后输出token不再消费。32/128/512两fresh逐token完整Tensor值/bytes与下一kernel真实读取检查；正式/4096为冻结稀疏位置全值/字节核验。82293为活模型合成KV及nativeSDPA资源压力，不是81920token数值误差上界。

本轮为逻辑4-bit整数QDQ，codes实际UINT8承载0..15，persistent State实际nativeFP32重建48MiB；理论packedcode6MiB另有scale/zmetadata，未实现物理INT4缓存，不声称实际4倍/8倍显存压缩。显存allocated/reserved峰值见每条原记录和压力回执。每条独立fresh进程/模型请求、Generator、Cache和generated-onlymask；单UUID串行，奇偶题交替顺序冻结。模型BF16权重不改，Sampler/topk20含原生ties行为不改。

这是看到INT8阶段性结果后设计的新探索实验；AIME26历史已多次观察，30题单seed限制必须保留。不能据少数题差异推断泛化或确定机制。跨UUID输入/协议匹配不等于验证跨硬件bitwise等价；旧HISTORICAL_PORT完整历史二进制EXACT身份仍UNKNOWN。INT4质量差本身不触发数学失败；没有按答案挑选重跑、缩短cap或额外科学分支。资源时间只有同主机monotonic区间估计，EXACT_GPUh=UNRESOLVED。
