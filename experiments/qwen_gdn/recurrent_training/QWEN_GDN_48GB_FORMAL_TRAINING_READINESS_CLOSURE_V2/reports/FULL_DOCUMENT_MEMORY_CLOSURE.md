# Full-document recurrent memory closure

The unchanged canonical C5/C6 loop uses one 1024-token document per optimizer update, eight frozen captures, segment backward at each occupied H boundary, cache detach every H tokens, and one Adam step after the document. No training formula or graph-lifetime strategy was changed.

## C5

H32 single full document: **PASS**; 1024 tokens, 8 captures, 6 occupied segment backprops; peak allocated 23.215 GiB, reserved 23.746 GiB; 24/24 valid rotation gradients; orthogonality PASS.
Three-update lifetime: **PASS**; peak allocated by document [23.119 GiB, 23.120 GiB, 23.123 GiB]; end-allocated span 0 bytes; memory creep NO; validation 16 documents/128 captures, return-to-baseline PASS (delta 0 bytes).

## C6

H32 single full document: **PASS**; 1024 tokens, 8 captures, 6 occupied segment backprops; peak allocated 23.215 GiB, reserved 23.746 GiB; 24/24 valid rotation gradients; orthogonality PASS.
Three-update lifetime: **PASS**; peak allocated by document [23.119 GiB, 23.120 GiB, 23.123 GiB]; end-allocated span 0 bytes; memory creep NO; validation 16 documents/128 captures, return-to-baseline PASS (delta 0 bytes).

The hard peak-free margin is 5% of device VRAM, with 10% preferred. All horizon decisions are resource/stability-only; loss and validation score did not select a horizon.

C5 H64 one-update: **PASS**, peak allocated 30.261 GiB, reserved 31.092 GiB, minimum conservative free 33.42%; sustained **PASS**.
C6 H64 one-update: **PASS**, peak allocated 30.261 GiB, reserved 31.111 GiB, minimum conservative free 33.37%; sustained **PASS**.
C5 H128 one-update: **BLOCKED** (OutOfMemoryError); last peak allocated 46.791 GiB, reserved 46.932 GiB, free 16.7 MiB; 1 capture(s), 0 completed backprops. No retry.
C6 H128 one-update: **BLOCKED** (OutOfMemoryError); last peak allocated 46.791 GiB, reserved 46.932 GiB, free 16.7 MiB; 1 capture(s), 0 completed backprops. No retry.
