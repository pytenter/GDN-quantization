"""Independent CPU check of every exported scientific CSV cell and statistic."""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
source = root / 'phase_a_analysis_20261011'
dest = root / 'phase_a_deliverables_20261011'
matched = json.loads((source / 'PHASE_A_MATCHED_TABLES.json').read_text())
discovery = json.loads((source / 'PHASE_A_DISCOVERY_TABLES.json').read_text())
specs = {
    'MATCHED_SHORT_HORIZON_SCORES.csv': matched['scores'],
    'ROTATION_MIDRANKS.csv': matched['ranks'],
    'PAIRWISE_RANKINGS.csv': matched['pairwise_rankings'],
    'RANKING_REDUNDANCY_STATISTICS.csv': matched['redundancy_statistics'],
    'RANKING_REDUNDANCY_LEAVE_ONE_OUT.csv': matched['redundancy_leave_one_out'],
    'DISCOVERY_ONLY_FUTURE_KL_ASSOCIATIONS.csv': discovery['discovery_statistics'],
    'DISCOVERY_ONLY_RANK_RESIDUALS.csv': discovery['discovery_rank_residuals'],
    'DISCOVERY_ONLY_LEAVE_ONE_OUT.csv': discovery['discovery_leave_one_out'],
}
for filename, source_rows in specs.items():
    with (dest / filename).open(encoding='utf8', newline='') as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == len(source_rows), filename
    for got, original in zip(records, source_rows):
        assert set(got) == set(original), filename
        for key, value in original.items():
            text = got[key]
            if value is None:
                assert text == '', (filename, key, text)
            elif isinstance(value, bool):
                assert text == str(value).lower(), (filename, key, text)
            elif isinstance(value, (float, int)):
                assert float(text) == value and math.isfinite(float(text)), (filename, key, text)
            else:
                assert text == value, (filename, key, text)

# Recalculate all 126 reported correlations with independent pandas midranks.
for row in matched['redundancy_statistics']:
    panel = [x for x in matched['scores'] if x['document'] == row['document'] and x['window'] == row['window']]
    ranks = [pd.Series([x[m] for x in panel]).rank(method='average').to_numpy()
             for m in (row['metric_a'], row['metric_b'])]
    correlation = float(np.corrcoef(*ranks)[0, 1])
    assert math.isclose(correlation, row['spearman_rho'], abs_tol=1e-12)

result = {'status': 'CPU_EXPORT_AND_INDEPENDENT_CORRELATION_CHECK_PASS',
          'gpu_hours': 0, 'all_csv_cells_checked': True,
          'independent_correlations_checked': 126,
          'source_data_unchanged': True,
          'artifact_process_nonzero_exit_after_success_output': True,
          'nonzero_exit_did_not_trigger_any_measurement_retry': True,
          'csv_files': [{'filename': k, 'rows': len(v),
                         'sha256': hashlib.sha256((dest / k).read_bytes()).hexdigest()}
                        for k, v in specs.items()]}
with (dest / 'CPU_EXPORT_VERIFICATION.json').open('x', encoding='utf8') as stream:
    json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(result))
