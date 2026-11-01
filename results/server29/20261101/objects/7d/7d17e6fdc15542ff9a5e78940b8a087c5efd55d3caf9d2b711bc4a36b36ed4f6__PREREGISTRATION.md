# Clean-qparam causal intervention

The only scientific change from frozen parent P is the source of the dynamic quantizer scale. F retains its own persistent INT8 state, every-token QDQ, and every-token low-bit writeback, but applies scale computed from the paired clean FP state in the same condition basis. Decision rules and the explosion threshold are frozen in `preregistration.json`.
