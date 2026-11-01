# Kernel 与边界时间

Packed base两个matvec、pending向量读出、delta与output融合在原同一个Triton实现中，实际分两种位宽各launch一次。不能不改kernel就分离各内部组件的排他时间：这些分项为UNMEASURED；提供合并时间、不同pending数量和独立host dispatch时间。prep的Torch kernel总数未profile，不把2个fused launch冒充全步总launch数。

边界的重建、原encoder总时间和packing分别使用相同诊断wrapper测量，诊断不进入主模型吞吐计时。encoder扣除packing的剩余为INFERRED，含scale/zp/exponent/RNE/layout/checks/dispatch，不能称纯scale算子时间。CUDA event为stream elapsed，可能包含host发射空隙，不是SM active时间。每组10次，原始值在KERNEL_RAW.json。

|周期|最大非边界pending|fused中位µs|边界全步中位µs|重建中位µs|encoder中位µs|packing中位µs|
|---|---:|---:|---:|---:|---:|---:|
|8|6|24.156160|14839.597046|7194.111824|3208.047986|455.632001|
|16|14|30.289919|25664.187968|16330.160141|3898.096085|538.032010|
|32|30|45.066240|36947.563291|27994.031906|3416.703939|472.576007|
