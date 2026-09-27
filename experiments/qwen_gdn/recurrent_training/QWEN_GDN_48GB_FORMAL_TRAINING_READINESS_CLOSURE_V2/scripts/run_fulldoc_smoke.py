#!/usr/bin/env python3
"""One diagnostic full-document C5/C6 update from theta=0."""
import argparse
from fulldoc_common import run

parser = argparse.ArgumentParser()
parser.add_argument('--condition', choices=('C5', 'C6'), required=True)
parser.add_argument('--horizon', type=int, choices=(32, 64, 128), default=32)
args = parser.parse_args()
run(args.condition, args.horizon, sustained=False)
