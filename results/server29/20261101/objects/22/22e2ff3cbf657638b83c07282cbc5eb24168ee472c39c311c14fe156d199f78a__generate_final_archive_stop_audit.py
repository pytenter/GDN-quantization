#!/usr/bin/env python3
"""Reduce independently verified small storage receipts, without any GPU access."""
import collections
import hashlib
import json
from pathlib import Path
import time

D = Path(__file__).resolve().parents[1]
summary = json.loads((D / 'ARCHIVE_AUDIT_SUMMARY.json').read_bytes())
assert summary['status'] == 'PASS_INDEPENDENT_STORAGE_DELIVERY_SCIENCE_REMAINS_INCOMPLETE_STOP'
downloads = {}
for row in summary['downloaded_metadata']:
    raw = Path(row['local']).read_bytes()
    assert len(raw) == row['bytes']
    assert hashlib.sha256(raw).hexdigest() == row['sha256']
    assert row['status'] == 'PASS_EXACT_SMALL_METADATA_DOWNLOAD'
    downloads[Path(row['local']).name] = row
assert len(downloads) == 6
inventory = json.loads((D / 'PHYSICAL_ARCHIVE_INVENTORY_AND_UNCOMMITTED.json').read_bytes())
assert inventory['status'] == 'PASS_PHYSICAL_ARCHIVE_ACCOUNTING_STORAGE_ONLY_SCIENCE_INCOMPLETE'
assert not inventory['missing_published']
assert inventory['no_deletion'] and inventory['no_GPU_calls'] and inventory['no_large_Mac_download']
rows = inventory['rows']
assert len({r['path'] for r in rows}) == len(rows)
physical = {r['path']: r for r in rows}
expected_counts = {'gpu0': 252, 'gpu1': 253, 'gpu2': 253, 'gpu3': 256}
expected_bytes = {'gpu0': 39765831648, 'gpu1': 39927977448, 'gpu2': 39927977448, 'gpu3': 40414611456}
per_card = []
accepted_paths = set()
for card, count in expected_counts.items():
    name = card + '_ACCEPTED_RAW_VERIFICATION.json'
    value = json.loads((D / name).read_bytes())
    assert value['status'] == 'PASS_FRESH_REMOTE_ARCHIVE_VERIFICATION'
    assert value['objects'] == count
    assert value['no_GPU_calls'] and value['no_large_Mac_download'] and not value['large_raw_downloaded']
    assert value['real_scientific_status'] == 'FAIL_CLOSED_STOP_INCOMPLETE'
    assert len(value['cards']) == 1
    bound = value['cards'][0]
    assert bound['status'] == 'PASS_FRESH_REMOTE_RAW_SHA_BYTES'
    assert bound['card_key'] == card
    objects = bound['objects']
    assert len(objects) == count
    byte_count = sum(r['bytes'] for r in objects)
    assert byte_count == expected_bytes[card]
    for obj in objects:
        assert obj['fresh_independent_stream_SHA'] is True
        stored = physical[obj['path']]
        assert stored['bytes'] == obj['bytes'] and stored['sha256'] == obj['sha256']
        assert stored['classification'] == 'PUBLISHED_REFERENCE_RAW_BYTES_VERIFIED_SCIENCE_INCOMPLETE'
        assert stored['scientific_qualification'] is False
        assert obj['path'] not in accepted_paths
        accepted_paths.add(obj['path'])
    catalog = bound['catalog']
    assert catalog['atomic_no_clobber_published'] is True
    stored_catalog = physical[catalog['path']]
    assert stored_catalog['bytes'] == catalog['bytes'] and stored_catalog['sha256'] == catalog['sha256']
    assert stored_catalog['classification'] == 'STORAGE_CATALOG_METADATA'
    per_card.append({
        'card_key': card, 'files': count, 'bytes': byte_count,
        'catalog': catalog, 'verification_receipt': downloads[name],
        'archive_config_sha256': value['archive_config_sha256'],
        'frozen_archive_module_sha256': value['archive_module_sha256'],
        'started_unix': value['started_unix'], 'finished_unix': value['finished_unix'],
        'scientific_qualification': False,
    })
actual_classes = collections.defaultdict(lambda: {'files': 0, 'bytes': 0})
for row in rows:
    actual_classes[row['classification']]['files'] += 1
    actual_classes[row['classification']]['bytes'] += row['bytes']
    assert row['scientific_qualification'] is False
assert dict(actual_classes) == inventory['classifications'] == summary['classifications']
assert len(rows) == inventory['files'] == summary['files'] == 1022
assert sum(r['bytes'] for r in rows) == inventory['bytes'] == summary['bytes'] == 160401436262
assert len(accepted_paths) == inventory['published_expected_files'] == 1014
assert sum(x['bytes'] for x in per_card) == 160036398000
uncommitted = [r for r in rows if r['classification'].startswith('UNCOMMITTED_')]
probe = [r for r in rows if r['classification'] == 'ENGINEERING_ONLY_PROBE']
other = [r for r in rows if r['classification'] == 'UNREFERENCED_ARCHIVE_ASSET_NOT_SCIENTIFICALLY_QUALIFIED']
assert len(uncommitted) == 3 and sum(r['bytes'] for r in uncommitted) == 360577048
assert len(probe) == 1 and sum(r['bytes'] for r in probe) == 4194304
assert not other
result = {
    'schema': 'FINAL_ARCHIVE_STOP_AUDIT_V1',
    'status': 'PASS_STORAGE_BYTES_AND_SMALL_DELIVERY_ONLY',
    'scientific_status': 'FAIL_CLOSED_STOP',
    'scientific_completeness': 'INCOMPLETE',
    'scientific_qualification': False,
    'scope': 'RAW_ARCHIVE_BYTE_PRESERVATION_AND_ACCOUNTING_ONLY',
    'accepted_published': {'files': len(accepted_paths), 'bytes': sum(x['bytes'] for x in per_card), 'per_card': per_card},
    'physical_archive': {
        'archive_root': inventory['archive_root'], 'files': inventory['files'], 'bytes': inventory['bytes'],
        'classifications': inventory['classifications'],
        'inventory_receipt': downloads['PHYSICAL_ARCHIVE_INVENTORY_AND_UNCOMMITTED.json'],
        'fresh_verification_note': 'All 1014 accepted raw objects were freshly independently streamed and SHA/bytes verified by the frozen per-card archive verifier on .29. The subsequent physical inventory reused those immediate per-card proofs and freshly streamed every other physical file. No storage proof is a scientific numeric parity PASS.',
    },
    'uncommitted_assets': uncommitted,
    'uncommitted_assets_files': len(uncommitted),
    'uncommitted_assets_bytes': sum(r['bytes'] for r in uncommitted),
    'engineering_probe_assets': probe,
    'other_unreferenced_assets': other,
    'catalog_precheck_receipt': downloads['ARCHIVE_CATALOG_PRECHECK.json'],
    'catalog_publication': 'ONCE_PER_CARD_ATOMIC_NO_CLOBBER_AFTER_PRECHECK_ABSENT',
    'all_current_physical_assets_accounted': True,
    'missing_published_assets': [],
    'all_downloaded_metadata_SHA_and_bytes_verified': True,
    'no_large_raw_downloaded': True,
    'no_GPU_calls': True,
    'no_deletion': True,
    'no_failed_or_partial_unit_restarted': True,
    'no_science_reclassification_of_orphan_temp_or_probe': True,
    'independent_audit_source_sha256': summary['remote_script_sha256'],
    'source_summary': {'path': str(D / 'ARCHIVE_AUDIT_SUMMARY.json'), 'sha256': hashlib.sha256((D / 'ARCHIVE_AUDIT_SUMMARY.json').read_bytes()).hexdigest()},
    'generated_unix': time.time(),
}
output = D / 'receipts' / 'FINAL_ARCHIVE_STOP_AUDIT.json'
raw = (json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode()
with output.open('xb') as f:
    f.write(raw)
print(json.dumps({'path': str(output), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'status': result['status'], 'accepted_files': 1014, 'accepted_bytes': 160036398000, 'physical_files': 1022, 'physical_bytes': 160401436262}))
