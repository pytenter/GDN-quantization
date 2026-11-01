"""CPU-only exact official WikiText calibration windows from a verified parquet.

The official calibration_segments body is unchanged. Its dataset loader alone
is bound to the full fixed-revision local parquet, never to prior selected raw
documents. No model weights or CUDA are loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer
import datasets
from stepquant.cli import calibration_segments

REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
PARQUET_SHA = 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def run(root, parquet, model):
    root, parquet, model = Path(root), Path(parquet), Path(model)
    if torch.cuda.is_initialized():
        raise ValueError('CPU data preparation has initialized CUDA')
    raw = parquet.read_bytes()
    if digest(raw) != PARQUET_SHA:
        raise ValueError('Official fixed-revision parquet SHA mismatch')
    table = pq.read_table(parquet, columns=['text'])
    texts = table.column('text').to_pylist()
    if len(texts) != 36718 or any(not isinstance(x, str) for x in texts):
        raise ValueError('Official train rows/schema mismatch')
    text = '\n\n'.join(texts)
    tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True, local_files_only=True)
    ids = tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids[0]
    args = SimpleNamespace(text_file=None, data_revision=REVISION, sequence_length=2048, seed=0, segments=32)
    calls = []
    def fixed_local_dataset(name, subset, split, revision):
        if (name, subset, split, revision) != ('Salesforce/wikitext', 'wikitext-2-raw-v1', 'train', REVISION):
            raise ValueError('Official dataset identity request changed')
        calls.append(dict(name=name, subset=subset, split=split, revision=revision))
        return {'text': texts}
    with patch.object(datasets, 'load_dataset', fixed_local_dataset):
        actual = list(calibration_segments(tokenizer, args))
    generator = torch.Generator().manual_seed(0)
    records = []
    combined = hashlib.sha256()
    for i, value in enumerate(actual):
        start = int(torch.randint(len(ids)-2048+1, (), generator=generator))
        expected = ids[start:start+2048][None]
        if value.shape != (1, 2048) or value.dtype != torch.int64 or not torch.equal(value, expected):
            raise ValueError('Unchanged official window differs from independently reconstructed window')
        token_bytes = value.numpy().astype('<i8', copy=False).tobytes()
        combined.update(token_bytes)
        records.append(dict(segment_index=i, start=start, token_ids_sha256=digest(token_bytes),
                            input_ids=value[0].tolist()))
    if len(records) != 32 or len(calls) != 1:
        raise ValueError('Official calibration segment/loader count changed')
    dest = root / 'calibration_data/official_wikitext32x2048_tokens.jsonl'
    with dest.open('x') as f:
        for record in records:
            f.write(json.dumps(record, separators=(',', ':')) + '\n')
    tokenizer_files = {p.name: digest(p.read_bytes()) for p in model.iterdir()
                       if p.name in ('tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','chat_template.jinja')}
    result = dict(status='CPU_INPUTS_FROZEN', dataset='Salesforce/wikitext:wikitext-2-raw-v1:train',
        revision=REVISION, parquet_sha256=PARQUET_SHA, parquet_bytes=len(raw), rows=len(texts),
        join_rule='\\n\\n.join(all full-train rows in original order)', joined_text_utf8_sha256=digest(text.encode('utf-8')),
        joined_text_CR_count=text.count('\r'), all_train_token_count=len(ids),
        full_train_token_sha256=digest(ids.numpy().astype('<i8', copy=False).tobytes()),
        tokenizer_files=tokenizer_files, tokenizer_class=type(tokenizer).__name__,
        segments_path=str(dest.relative_to(root)), segments_sha256=digest(dest.read_bytes()),
        segments=32, sequence_length=2048, seed=0, total_calibration_tokens=65536,
        official_combined_token_ids_sha256=combined.hexdigest(), window_start_indices=[r['start'] for r in records],
        official_function_vs_independent_windows='EXACT', data_loader_IO_only_override=calls,
        sample_every=8, max_snapshots=64, lifetime_horizon=2048, nominal_bits=6,
        pivots=32, value_group_size=32, gamma=.25, fit_device='cpu',
        AIME_used=False, calibration_plan='NOT_GENERATED', CUDA_initialized=torch.cuda.is_initialized())
    path = root / 'CALIBRATION_INPUTS.json'
    with path.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--parquet', required=True)
    p.add_argument('--model', required=True)
    a = p.parse_args()
    print(json.dumps(run(a.root, a.parquet, a.model), ensure_ascii=False, indent=2))
