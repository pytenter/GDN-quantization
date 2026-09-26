# Quantizer Compatibility

| Structure | Storage cost | Calibration cost | Runtime cost | Compatible with current INT8 quantizer? | Requires dynamic q? | Requires dense operation? | Scientific fidelity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M0 raw reconstruction | none | none | low | yes | no | no | baseline |
| query-only | none | none | dynamic readout q | conditional | yes | no | dynamic baseline |
| static covariance x value weight | per-layer/head 128x128 covariance plus value weights | moderate | matrix-vector per value column | no/directly expensive | no | yes within key rows | high if ranking retained |
| static element-wise | per-layer/head 128x128 weights | low | weighted MSE | yes | no | no | primary deployable candidate |
| static separable row x column | per-layer/head row+column vectors | low plus rank-1 SVD | weighted MSE | yes | no | no | simplest deployable candidate if close to element |
| row-only | row vector per layer/head | low | row weighted MSE | yes for row/group search | no | no | structural baseline |
| column-only | column vector per layer/head | low | column weighted MSE | yes for column/group search | no | no | structural baseline |
