# Ling recurrent memory source-code evidence

- Trainer SHA256: `d1c433598875ea84c80c6f6d4fb5fe48088d5ca0bf44ccff9d950a77560fd9ae`
- QDQ source SHA256: `62e769dfad73170279daf0bdb56855b7d7e4aaf77478de897958fa87b2608067`
- Full-path probe SHA256: `3f04f8b9ba089e023d2167998692315dcc005449512a7d6ac263047f26bd323d`
- Recurrent detach semantics: **PASS**.
- Persistent loss/history tensors: no graph-bearing CUDA tensor append was found.
- QDQ uses local exact-QDQ intermediates and a detach-based identity STE; no custom autograd context or module cache was found.
- Dynamic cache, weakref and post-validation memory-return probes are deferred because formal generation is active.

The source audit is limited to Ling and makes no cross-model comparison.
