#!/usr/bin/env python3
"""Verify the copied chunk function differs only by the declared cumsum call."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
from pathlib import Path


CANONICAL = Path('/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py')
COPY = Path(__file__).with_name('qwen_chunk_reference.py')
EXPECTED_SHA256 = '90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a'


def function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


class RestoreCanonicalCumsum(ast.NodeTransformer):
    def __init__(self):
        self.replacements = 0

    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'fixed_left_to_right_cumsum':
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Name) or node.args[0].id != 'g':
                raise RuntimeError('unexpected candidate call')
            self.replacements += 1
            return ast.Call(func=ast.Attribute(value=ast.Name(id='g', ctx=ast.Load()),
                                               attr='cumsum', ctx=ast.Load()),
                            args=[], keywords=[ast.keyword(
                                arg='dim', value=ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1)))])
        return node


def main() -> None:
    observed_sha = hashlib.sha256(CANONICAL.read_bytes()).hexdigest()
    if observed_sha != EXPECTED_SHA256:
        raise RuntimeError(f'canonical source hash changed: {observed_sha}')
    original = function(CANONICAL, 'torch_chunk_gated_delta_rule')
    candidate = function(COPY, 'deterministic_torch_chunk_gated_delta_rule')
    original.decorator_list = []
    candidate.name = original.name
    transformer = RestoreCanonicalCumsum()
    candidate = transformer.visit(candidate)
    if transformer.replacements != 1:
        raise RuntimeError(f'expected one cumsum patch, got {transformer.replacements}')
    same = ast.dump(original, include_attributes=False) == ast.dump(candidate, include_attributes=False)
    result = {'canonical_sha256': observed_sha, 'candidate_sha256': hashlib.sha256(COPY.read_bytes()).hexdigest(),
              'declared_patch_replacements': transformer.replacements,
              'other_function_AST_differences': not same,
              'SOURCE_DELTA_GATE': 'PASS' if same else 'FAIL'}
    target = Path(__file__).resolve().parents[2] / 'analysis/deterministic_chunk_source_delta.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(result, sort_keys=True), flush=True)
    if not same:
        print('\n'.join(difflib.unified_diff(ast.unparse(original).splitlines(),
                                             ast.unparse(candidate).splitlines(),
                                             fromfile='canonical', tofile='candidate',
                                             lineterm='')), flush=True)
        left, right = ast.dump(original, include_attributes=False), ast.dump(candidate, include_attributes=False)
        first = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
        print(json.dumps({'first_AST_dump_difference_index': first,
                          'canonical_context': left[max(0, first-120):first+120],
                          'candidate_context': right[max(0, first-120):first+120]}), flush=True)
        raise RuntimeError('copy differs beyond explicit cumsum replacement')


if __name__ == '__main__':
    main()
