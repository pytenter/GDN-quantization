# Canonical Asymmetric INT4 Group32 V1

State FP32 [B,H,Key128,Value128] reshape[B,H,4,32,128] reduction axis -2. Every Value column independent, no crosshead/layer grouping. Include0 extrema canonical+0.

a=min(min_i(x),0); b=max(max_i(x),0); span=FP32(b-a); safe=max(span,FP32(1e-12)); s=FP32(safe / FP32-tensor(15)); z=uint8(clamp(RNE(FP32(-a/s)),0,15)); q=uint8(clamp(FP32(RNE(FP32(x/s))+FP32(z)),0,15)); decoded=FP32(FP32(int16(q)-int16(z))*s), canonical+0 on zero.

Recovered old INT8 uses a **range floor** EPS=1e-12 before dividing by a FP32 tensor255, not a scale floor1e-12. New V1 preserves this exact operation ordering with denominator15. Its effective minimum scale is FP32(FP32(1e-12)/15), approximately6.6666665e-14, whereas INT8 is FP32(EPS/255). This bitwidth-derived effective scale change is explicit; it is not a claim of identical numerical minimum scale. No AIME calibration.

RNE(x/s)+z is selected to match inherited Asym32. RNE(x/s+z) is explicitly different at odd-zero-point half ties, covered by fixtures. Tensor divisor avoids scalar reciprocal substitution. All contracted field comparisons are exact dtype/shape/bytes, no numerical tolerance. Span overflow/nonfinite input, scales, levels or decode is a real rejection, never hidden by epsilon/dtype change. Valid finite extreme fixtures and separate expected-rejection cases distinguish this.

Native/H128 use identical new codec, with only qualified original coordinate patch differing. Original H128 matrix/sign/order, normalize_qk and recurrence are reused unmodified. Persistent nativeFP32 decoded-state carrier; UINT8 temporarycodes0..15, UINT8integerz, FP32scales; no packedINT4storage, no continuation highprecision shadow. 24x32x128x128 FP32 states remain48MiB. Hypothetical packed4bit codes6MiB (plus scale/zmetadata) are theoretical and not measured memory compression.
