#!/usr/bin/env python3

import sys
sys.path.append("/home/hchoi/Nextcloud/link-prediction/src/trainers/OpenKE")

import argparse
import trainers
from trainers import analogy, transe, transd, transh, distmult, complex, rotate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run models for link prediction task")
    parser.add_argument('-dataset', type=str, default='rezojdm16k', help="Choose a dataset")

    args = parser.parse_args()


    trainers.transe.run(args.inpath)
    trainers.transh.run(args.inpath)
    trainers.transd.run(args.inpath)
    trainers.distmult.run(args.inpath)
    trainers.complex.run(args.inpath)
    trainers.rotate.run(args.inpath)
