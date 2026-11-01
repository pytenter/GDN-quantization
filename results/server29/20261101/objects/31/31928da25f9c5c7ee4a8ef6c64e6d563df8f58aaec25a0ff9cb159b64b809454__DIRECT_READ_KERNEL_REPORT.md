# Packed Direct Read核验

CODE_VERIFIED：src/packed_direct_read.py；INT4/INT8 heads分别launch，沿Key维对packed codes、整数zero、FP16 mantissa和optional INT8 exponent解码；FP32 accumulation；enable_fp_fusion=False。未保存全部Query/原V/beta，pending只存压缩K和actualu/g。

RUNTIME_VERIFIED：FAST_BATCH_KEYS_V5 40 cases、CUDA_FRESH_V4独立fresh进程、SAME_STATE_ORACLE_V1 300独立FP64对照、四卡FAST_MODEL_SMOKE_V2、FULL_MODEL_PARITY_V1。实际loaded model flags在各receipt中。CUDA_KERNEL_V1真实编译失败及exit1/release仍保留。

Kernel与全模型吞吐分开，图重放CUDA event核时间不包含Python dispatch和K批准备。无官方kernel吞吐或SM active时间断言。
