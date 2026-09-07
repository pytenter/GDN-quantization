# Postconv Operator Graph

Post-conv structured rotation insertion point:

hidden -> in_proj_qkv -> native learned depthwise short convolution -> split/reshape/repeat q,k
-> [apply H to q and k last/key dimension] -> recurrent gated delta core.

Value is not rotated. beta/g decay are unchanged. q/k l2norm remains inside the recurrent kernel
after the insertion point and commutes with orthogonal H. The recurrent state cache is initialized
as zero, so H*0=0; subsequent updates use k'=Hk, therefore the stored recurrent state remains in
the rotated key basis without any explicit full-state rotation per token.

C128 semantics after rotation: for each value column j, the 128 entries along the rotated key axis
share one INT8 scale. The grouping axis and bit budget are identical to canonical C128.
