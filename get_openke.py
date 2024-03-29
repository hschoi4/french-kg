#!/usr/bin/env python3

import sys
import os
import pandas as pd
import warnings
import re
import datetime
import argparse
from pathlib import Path

def reverse_order(df):
    """
        switch order of columns
    """

    df = df[df.columns[::-1]]
    df_ = df.copy()

    # add size of dataframe in the first line of the file
    df_.loc[-1] = [len(df), ""]  # adding a row on dat
    df_.index = df_.index + 1  # shifting index
    df_.sort_index(inplace=True)

    return df_

def convert(dataset):
    """
        convert files into openKE format
    """
    pathdir = Path('data/') / Path(dataset)

    entities = pd.read_csv(f'{pathdir}/entities.txt', delimiter="\t", names=['id', 'name'])
    relations = pd.read_csv(f'{pathdir}/relations.txt', delimiter="\t", names=['id', 'name'])

    files = ['train', 'test', 'valid']
    path_openke = Path(f'{pathdir}/openke/')
    path_openke.mkdir(parents=True, exist_ok=True)

    ent_map = dict(zip(entities['name'].tolist(), entities['id'].tolist()))
    rel_map = dict(zip(relations['name'].tolist(), relations['id'].tolist()))


    for file in files:
        df = pd.read_csv(f'{pathdir}/{file}.txt', delimiter="\t", names=['source', 'type', 'target'])

        if dataset == 'rlf/lffam-cp':

            df['type'] = df['type'].map(rel_map)
            df['source'] = df['source'].map(ent_map)
            df['target'] = df['target'].map(ent_map)

            df = df.reindex(columns=['source', 'target','type'])

            df.loc[-1] = [len(df),"", ""]  # adding a row, add size of set
            df.index = df.index + 1  # shifting index
            df.sort_index(inplace=True)

            df.to_csv(f'{path_openke}/{file}2id.txt', index=False, header=False, sep=" ")

        elif dataset == 'rezojdm16k':
            df = df.reindex(columns=['source', 'target','type'])

            df.loc[-1] = [len(df),"", ""]  # adding a row, add size of set
            df.index = df.index + 1  # shifting index
            df.sort_index(inplace=True)

            df.to_csv(f'{path_openke}/{file}2id.txt', index=False, header=False, sep=" ")

    reversed_ents = reverse_order(entities)
    reversed_rels = reverse_order(relations)

    reversed_ents.to_csv(f'{path_openke}/entity2id.txt', index=False, header=False, sep="\t")
    reversed_rels.to_csv(f'{path_openke}/relation2id.txt', index=False, header=False, sep="\t")

    return entities, relations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-dataset', dest='dataset', default="rlf/lffam-cp", help='Dataset to use, default: rlf/lffam-cp')

    args = parser.parse_args()

    convert(dataset=args.dataset)
