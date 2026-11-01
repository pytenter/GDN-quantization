"""Synthetic statistics and genuine 256-position schema smoke; no outcome inference."""
import numpy as np
import pandas as pd

import analyze_stability_v2 as A


def main():
    layer = pd.read_parquet(A.OUT / "source_audit/smoke_aime01_layers.parquet")
    logit = pd.read_parquet(A.OUT / "source_audit/smoke_aime01_logits.parquet")
    base = {"problem_id":"aime26_01","seed":1,"rotation":"Identity","clean59":True,"group":"stable_correct","fp_correct":True,"outcome_correct":True,"token_count":256}
    summary, layers, frame = A.summarize_rotation(layer, logit, 256, base)
    assert len(layers) == 24 and len(frame) == 256
    assert all(np.isfinite(summary[k]) for k in ("state_median","readout_median","kl_median","readout_late_log_growth_mean"))
    rows=[]
    groups=["stable_correct"]*18+["hadamard_rescue"]*23+["stable_failure"]*18
    for i, group in enumerate(groups):
        for rotation, offset in (("Identity",0.0),("Hadamard",.01)):
            row={"problem_id":f"synthetic_{i:02d}","seed":1,"group":group,"rotation":rotation,"clean59":True}
            row.update({name:float(i)/100+offset for name in A.METRIC_ORIENTATION})
            rows.append(row)
    paired, stats = A.pair_and_test(pd.DataFrame(rows))
    assert len(paired) == 59*len(A.METRIC_ORIENTATION)
    assert stats["readout_onset_fraction"]["groups"]["hadamard_rescue"]["n"] == 23
    assert A.exact_sign_p(23,0) < 1e-5 and A.auroc([0,1,0,1],[0,2,1,3]) == 1.0
    print("ANALYZER_SMOKE=PASS",len(paired),len(layers),len(frame))


if __name__ == "__main__":
    main()
