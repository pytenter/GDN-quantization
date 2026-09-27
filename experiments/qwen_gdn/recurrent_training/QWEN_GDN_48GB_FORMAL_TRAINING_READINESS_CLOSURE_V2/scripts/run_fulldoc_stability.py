#!/usr/bin/env python3
"""Three diagnostic full-document updates and unchanged validation panel."""
import argparse
from fulldoc_common import run

parser = argparse.ArgumentParser()
parser.add_argument('--condition', choices=('C5', 'C6'), required=True)
parser.add_argument('--horizon', type=int, choices=(32, 64, 128), default=32)
args = parser.parse_args()
run(args.condition, args.horizon, sustained=True)
