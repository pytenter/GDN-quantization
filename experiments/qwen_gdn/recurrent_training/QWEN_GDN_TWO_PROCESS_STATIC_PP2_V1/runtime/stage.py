#!/usr/bin/env python3
"""Permanent stage-local Qwen modules; no full-model instantiation or sharding."""
from __future__ import annotations

import gc
import json
from contextlib import ExitStack
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as qmod

MODEL_PATH = Path('/data/zypan/modelscope_models/Qwen3.5-9B')


class CheckpointReader:
    """Read only selected tensors; safetensors shards remain CPU-mapped."""

    def __init__(self, root: Path = MODEL_PATH):
        self.root = root
        self.index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
        self.stack = ExitStack()
        self.handles = {}

    def tensor(self, full_key: str) -> torch.Tensor:
        filename = self.index[full_key]
        if filename not in self.handles:
            self.handles[filename] = self.stack.enter_context(
                safe_open(str(self.root / filename), framework='pt', device='cpu'))
        return self.handles[filename].get_tensor(full_key)

    def load_module(self, module: nn.Module, checkpoint_prefix: str, device: torch.device) -> nn.Module:
        state = {name: self.tensor(checkpoint_prefix + name)
                 for name in module.state_dict().keys()}
        incompatible = module.load_state_dict(state, strict=True, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f'Checkpoint mismatch at {checkpoint_prefix}: {incompatible}')
        module.to(device=device)
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        del state
        gc.collect()
        return module

    def close(self):
        self.stack.close()


class StaticQwenStage(nn.Module):
    def __init__(self, rank: int, config, device: torch.device):
        super().__init__()
        if rank not in (0, 1):
            raise ValueError(rank)
        self.rank = rank
        self.config = config
        self.device = device
        self.block_ids = tuple(range(0, 16) if rank == 0 else range(16, 32))
        self.layers = nn.ModuleDict()
        self.rotary_emb = qmod.Qwen3_5TextRotaryEmbedding(config=config).to(device)
        self.embed_tokens = None
        self.norm = None
        self.lm_head = None

    @classmethod
    def load(cls, rank: int, device: torch.device):
        outer = AutoConfig.from_pretrained(str(MODEL_PATH), trust_remote_code=True,
                                           local_files_only=True)
        config = outer.text_config
        if config.num_hidden_layers != 32 or config.hidden_size != 4096:
            raise RuntimeError('Unexpected text model config')
        stage = cls(rank, config, device)
        reader = CheckpointReader()
        try:
            if rank == 0:
                with torch.device('meta'):
                    embedding = nn.Embedding(config.vocab_size, config.hidden_size,
                                             config.pad_token_id)
                stage.embed_tokens = reader.load_module(
                    embedding, 'model.language_model.embed_tokens.', device)
            for block_id in stage.block_ids:
                with torch.device('meta'):
                    layer = qmod.Qwen3_5DecoderLayer(config, block_id)
                stage.layers[str(block_id)] = reader.load_module(
                    layer, f'model.language_model.layers.{block_id}.', device)
            if rank == 1:
                with torch.device('meta'):
                    norm = qmod.Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                    head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                stage.norm = reader.load_module(norm, 'model.language_model.norm.', device)
                stage.lm_head = reader.load_module(head, 'lm_head.', device)
        finally:
            reader.close()
        stage.eval()
        return stage

    def new_cache(self):
        return qmod.DynamicCache(config=self.config)

    def global_layer_list(self):
        """Sparse hook map for frozen RecurrentExperimentPatch; no off-stage module."""
        result = [None] * 32
        for block_id in self.block_ids:
            result[block_id] = self.layers[str(block_id)]
        return result

    def forward_stage(self, *, input_ids=None, boundary_hidden=None, cache=None,
                      start_position: int, use_cache: bool = True):
        if self.rank == 0:
            if input_ids is None or boundary_hidden is not None:
                raise ValueError('rank0 takes input_ids only')
            hidden = self.embed_tokens(input_ids)
        else:
            if boundary_hidden is None or input_ids is not None:
                raise ValueError('rank1 takes boundary_hidden only')
            hidden = boundary_hidden
        if cache is None and use_cache:
            cache = self.new_cache()
        batch, seq, _ = hidden.shape
        positions = torch.arange(start_position, start_position + seq,
                                 device=hidden.device)
        position_ids = positions.view(1, 1, -1).expand(4, batch, -1)
        text_position_ids = position_ids[0]
        mask_kwargs = {'config': self.config, 'inputs_embeds': hidden,
                       'attention_mask': None, 'past_key_values': cache,
                       'position_ids': text_position_ids}
        masks = {'full_attention': qmod.create_causal_mask(**mask_kwargs),
                 'linear_attention': qmod.create_recurrent_attention_mask(**mask_kwargs)}
        position_embeddings = self.rotary_emb(hidden, position_ids[1:])
        for block_id in self.block_ids:
            hidden = self.layers[str(block_id)](
                hidden, position_embeddings=position_embeddings,
                attention_mask=masks[self.config.layer_types[block_id]],
                position_ids=text_position_ids, past_key_values=cache,
                use_cache=use_cache)
        if self.rank == 1:
            hidden = self.norm(hidden)
            logits = self.lm_head(hidden)
            return hidden, logits, cache
        return hidden, None, cache
