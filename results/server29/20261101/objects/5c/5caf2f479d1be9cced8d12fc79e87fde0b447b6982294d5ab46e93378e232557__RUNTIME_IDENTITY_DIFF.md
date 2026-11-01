# 历史与当前真实路径CPU审计

|维度|历史AIME|当前短profile|证据状态|
|---|---|---|---|
|runner|f552 git原文件rawSHA9ff56d|冻结core aa338/worker e04ec/measurement0961|OBSERVED differentorchestration|
|modeling_qwen3_5.py|Sept17 frozenmanifest/snapshot90d929|currentandlive90d929|OBSERVED samebytes|
|Torch/CUDA/Transformers|rawrows2.5.1+cu121/12.1/5.16.0.dev0|configureassertTorch/CUDA; samepinnedsrc|sameversions, completeoldpackageinventoryUNKNOWN|
|weights|BF16 samepath+configd088|BF16 samepath/config|historicalweightshardSHAUNKNOWN|
|GDNrecurrent|native90d|native90d|OBSERVED identicalsource, notproof identicaldevicekernels underflags|
|chunk cumsum|native|qualifiedcompensatedsite|OBSERVED occurs onlyprefill in cachedstream|
|determinism|runner/shellunset|strictTrue|oldFalseINFERRED fromfreshdefault; historicflagtelemetryUNKNOWN|
|TF32|runner/shellunset|matmul/cudnnFalse|olddefaultcurrentprobeFalse/True, historicalnotrecorded|
|threads|unset; currentfresh24/24|1/1+OMP/MKL1|OBSERVED sourceconfigdifference; historicaleffectiveUNKNOWN|
|H|normalize+q/k FP32matrix rebuiltpertransform|identicalmathAST, matrixmemoized|OBSERVED lifecycle difference|
|C128|samequantizeAST, firstQDQafterfirstdecode|sameAST, QDQalsoafterfirsttokenprefill|OBSERVED protocol difference, notnumericalequivalence|
|FPguard|full-logitfinite; samplingchecks|24perstatefinite+full-logitfinite, PythonObserve|OBSERVED additional checks|
|prefill/cache|45..373 prompt,9K..81Kgenerated|1prefill,128warmup,128measure,cache128..256|OBSERVED workloaddifference|
|statistics|60requestsumN/sumT|3decodewindowmedianN/T|OBSERVED differentestimands, insufficienttobudget40.7%alone|
|timing|prefill+decode+sampling, notload/textdecode|stablefixed-tokenwindow, no sampling|OBSERVED scope difference|
|hardware|4index0..3+qwen3090serverlabel|single0 exactGPU-a781...|historicalUUID/clocks/loadUNKNOWN|

compensated_scan仅替换90d chunk中的`g.cumsum`一个callsite；recurrent helper源码没有cumsum。当前之前原始记录确认每256forward只有24chunk、6120recurrent；因此不能把每token Decode减速直接归因于补偿scan。全局deterministic仍可能影响其它ATen/cuBLAS/cuDNN选择，其耗时效应尚待同条件验证。旧H与新H六个关键函数AST一致只说明抽取数学关系，raw源码身份仍以各自rawSHA为准。
