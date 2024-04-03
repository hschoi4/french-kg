#!/usr/bin/env python3

import sys
sys.path.append("/home/hchoi/Nextcloud/fr-link-prediction/french-kg/trainers/OpenKE")

import argparse
import trainers
from trainers import analogy, transe, transd, transh, distmult, complex, rotate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run models for link prediction task")
    parser.add_argument('-dataset', type=str, default='rezojdm16k', help="Choose a dataset")

    args = parser.parse_args()


    trainers.transe.run(args.dataset)
    trainers.transh.run(args.dataset)
    trainers.transd.run(args.dataset)
    trainers.distmult.run(args.dataset)
    trainers.complex.run(args.dataset)
    trainers.rotate.run(args.dataset)
