#!/usr/bin/env python3
"""Three diagnostic full-document updates and unchanged validation panel."""
import argparse
from fulldoc_common import run

parser = argparse.ArgumentParser()
parser.add_argument('--condition', choices=('C5', 'C6'), required=True)
args = parser.parse_args()
run(args.condition, 32, sustained=True)
