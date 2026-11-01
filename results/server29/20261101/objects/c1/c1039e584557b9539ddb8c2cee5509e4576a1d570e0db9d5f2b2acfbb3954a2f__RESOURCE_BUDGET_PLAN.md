# C6R four-GPU resource plan

The human requested four GPUs and storage on29 before any new forward. The earlier single-GPU draft is preserved beside this file; shared20GPUh and protective19.5GPUh are unchanged.

| Item | GPUh |
|---|---:|
| Historical samecodec ten actual reservations |14.357984323|
| Five times each codec historical slowest |14.691678917|
| Initial formal forecast, historical worst ×1.20 |17.630014700|
| Four C6R initial/fresh128 qualifications,0.35 each |1.400000000|
| Shared checkpoint wait / held-IO planning reserve |0.100000000|
| Exit forecast reserve900seconds |0.250000000|
| Total |19.380014700|

This leaves7.20 minutes to the19.5 protective forecast threshold, with a further0.5GPUh to the20 hard limit for stopping/release. It is a conditional estimate, not a guarantee. Actual new qualification replaces its allowance. Actual max4 OFFmedian ratios may only raise the historical1.20floor, without multiplying the margin twice; subsequent samecodec32K overruns only raise remaining predictions. No Delta is used.

Fixed schedule: doc1+5GPU0, doc2GPU1, doc3GPU2, doc4GPU3, both new codecs for one doc on that samecard. Every card needs its own initial/fresh128 qualification, and the four fresh outputs must match at128; this does not bound full32K numerical uncertainty.

Global checkpoint: all lane checkpoint links point to the same new/tmp directory. Whole original checkpoint body runs under one global flock. Atomic bound2×1.52GB +512MiB reserve requires3,576,870,912 bytes versus observed/tmp3,804,200,960. No four retained currentstates are promised; every retired version remains cataloged. Do not archiveCURRENT until all writers/GPUworkers have exited.

Scores: /data keeps at most four unarchivedscience outputs, one perlane. Four384MB files + one384MB write allowance +256MiB canaryscores +512MiB metadata +512MiB reserve needs3,262,177,280 bytes versus observed/data3,655,974,912. A boundedCPU archive thread copies each completedscore and staticreceipts to a fresh permanent29 path, independently verifies SHA/bytes and atomically seals it, then removes only that exactnew local score. The lane does not advance until archival passes. All historical assets remain untouched.

Archival after workerrelease consumes CPU/networkwalltime, not fabricatedGPUh; every other stillactive GPU remains charged while it proceeds. Any archive/disk/identity/parity/finite/FP16-risk/foreignprocess failure causes failclosed stop of all exactowned workers. Active intervals are summed and protective19.5 is checked during runtime; no card has its own20h allowance. tmux ancestry, residentstdout,30-minute no-progress watchdog and singleattempt guard are required. No retries, extra samples, training, AIME or successor experiment.
