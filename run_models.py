#!/usr/bin/env python3

import sys
sys.path.append("./french-kg/trainers/OpenKE")

import argparse
import trainers
from trainers import analogy, transe, transd, transh, distmult, complex, rotate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run models for link prediction task")
    parser.add_argument('-dataset', type=str, default='rezojdm16k', help="Choose a dataset")

    args = parser.parse_args()

    path = f"data/{args.dataset}/openke/"

    trainers.transe.run(path)
    trainers.transh.run(path)
    trainers.transd.run(path)
    trainers.distmult.run(path)
    trainers.complex.run(path)
    trainers.rotate.run(path)
