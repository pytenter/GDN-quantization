# Token-local tail specification

For N=4096 and q=0.99: h=(N-1)q=4054.05, lo=4054, hi=4055. Linear p99 is `x[lo] + (h-lo)*(x[hi]-x[lo])`. Top-42 begins at ascending index 4054, so offsets 0 and 1 recover p99. `K_CVAR99=ceil(0.01*4096)=41`; the 41 largest values are top-42 offsets 1..41. Max is offset 41. Thus top-42 is sufficient for exact p99, CVaR99, and max.
