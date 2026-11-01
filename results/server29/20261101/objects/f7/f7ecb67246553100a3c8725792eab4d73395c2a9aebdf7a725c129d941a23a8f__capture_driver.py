import argparse
import os
import sys

from common import EXP, PARENT_SCRIPTS, read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--document', type=int, required=True, choices=range(1, 5))
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('canonical capture must use physical GPU0')
    authorization = read_json(EXP / 'execution_authorization.json')
    if authorization['CAPTURE_INPUT_ACQUISITION_AUTHORIZED'] != 'YES':
        raise RuntimeError('capture input acquisition is not authorized')
    sys.path.insert(0, str(PARENT_SCRIPTS))
    import capture_original_drive as capture_base
    capture_base.AMEND = EXP
    import capture_extended_drive
    capture_extended_drive.execute('discovery', args.document)


if __name__ == '__main__':
    main()

