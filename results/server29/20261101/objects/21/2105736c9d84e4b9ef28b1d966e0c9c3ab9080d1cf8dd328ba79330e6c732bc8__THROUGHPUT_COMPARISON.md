# dP8 / dP16 / dP32 同卡吞吐比较

状态：COMPLETE_NO_CLEAR_SPEEDUP

仅新目录、物理GPU3/RTX3090、TP1、BF16。主结论使用本轮同卡重测；原GPU1的6.65/6.79仅作历史辅助。原非AIME16-token输入、原sampler与检查、32 warmup、128实测decode forward、每配置3次且顺序轮换。Short直接调用字节未改的model_engineering.fixed_run。Long使用同一token流和相同逐token forward，计数为真实decode forward，非并行prefill。每次重建独立请求cache；没有自由生成、AIME答案或评分。

## short

|配置|每次 token/s|中位数|最小/最大|中位 TPOT ms|
|---|---|---:|---|---:|
|dP8|5.000000, 5.000000, 5.000000|5.000000|5.000000/5.000000|200.000000|
|dP16|5.000000, 5.000000, 5.000000|5.000000|5.000000/5.000000|200.000000|
|dP32|5.000000, 5.000000, 5.000000|5.000000|5.000000/5.000000|200.000000|

{"speedup32_vs16": 1.0, "delta32_vs16_percent": 0.0, "speedup32_vs8": 1.0, "delta32_vs8_percent": 0.0, "observed_ranges32_16_overlap": true, "not_a_statistical_significance_test": true}

## long

|配置|每次 token/s|中位数|最小/最大|中位 TPOT ms|
|---|---|---:|---|---:|
|dP8|5.000000, 5.000000, 5.000000|5.000000|5.000000/5.000000|200.000000|
|dP16|5.000000, 5.000000, 5.000000|5.000000|5.000000/5.000000|200.000000|
|dP32|5.000000, 5.000000, 5.000000|5.000000|5.000000/5.000000|200.000000|

{"speedup32_vs16": 1.0, "delta32_vs16_percent": 0.0, "speedup32_vs8": 1.0, "delta32_vs8_percent": 0.0, "observed_ranges32_16_overlap": true, "not_a_statistical_significance_test": true}

## 相同上下文区间

|Decode forward区间|dP8中位TPS|dP16中位TPS|dP32中位TPS|
|---|---:|---:|---:|
|2048–8192|5.000000|5.000000|5.000000|
|8192–16384|5.000000|5.000000|5.000000|

重复间范围接近或重叠时不宣称可明确区分快慢；区间分离也不是总体统计显著性证明。长测试历时较长，其他卡上的原实验照常运行，CPU/温度/功耗环境见逐15秒telemetry，不能冒充隔离服务吞吐或SM occupancy。写回减少和pending增加的净效应依据本轮真实测量判断，不通过改kernel优化dP32。
