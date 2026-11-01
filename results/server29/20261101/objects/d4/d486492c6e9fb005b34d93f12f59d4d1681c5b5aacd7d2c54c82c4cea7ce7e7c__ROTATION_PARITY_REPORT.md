# H128 旋转准入报告

C2正式状态：STOP_ROTATION_PARITY_FAILED，未生成HMMT。

RUNTIME_VERIFIED：V1归一化前旋转并转BF16，FP输出NRMSE1.05379%>原0.5%，实际worker退出2/release。原失败源码、快照和日志保留。V2在原BF16 norm后FP32旋转，理想坐标最大3.907678e-7，生产输出0.0156168%、State2.80353e-7、旋转Direct对Replay0.0163220%，均通过原算子门槛。

随后真实9B、49次teacher-forced非HMMT前向，FP_H logits最坏4.2068837%>正式前固定1%；多层State漂移最高1.93627%。C0观察器与原生FP的logits及FP32 cache逐字相同，四卡短序列无nonfinite。不能把单算子PASS冒充模型级PASS。保留容差，不自动尝试新H变体，不推断C2准确率。

CODE_VERIFIED：H128只乘Key轴，decay[B,H]在K/V轴广播，标量每head故与H可交换。V2原物理K4包保留native preimage、原codec/G32不变；使用时一次原BF16 norm后在FP32施H，packed State驻H，无Value/head旋转，无边界逆变换。H矩阵共享65536bytes/device。因模型级门禁失败，这些为诊断代码事实，不是正式C2资格。

证据：releases/POSTNORM_H128_V2/validation/MATH_CUDA_V2.json、FULL_MODEL_GPU0.json及runtime真实owning收据；根目录validation/MATH_CUDA_V1.json。
