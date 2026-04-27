#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import os
import shutil


def trim_center(seq, target_len):
    if len(seq) == target_len:
        return seq
    if len(seq) < target_len:
        raise ValueError(f"Sequence shorter than target_len: {len(seq)} < {target_len}")

    start = (len(seq) - target_len) // 2
    end = start + target_len
    return seq[start:end]
    
        
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="dataset directory")
    parser.add_argument("--train_src", type=str, required=True, help="")                    
    parser.add_argument("--test_src", type=str, required=True, help="") 
    parser.add_argument("--train_file", type=str, required=True, help="")                    
    parser.add_argument("--test_file", type=str, required=True, help="") 
    parser.add_argument("--seqwin", type=int, required=True, help="")
    args = parser.parse_args()

    DATA_PATH = args.path
    TARGET_LEN = args.seqwin
    train_src = args.train_src
    test_src = args.test_src
    train_file = args.train_file
    test_file = args.test_file

    # ===== train =====
    train_src = f"{DATA_PATH}/{train_src}"
    train_out = f"{DATA_PATH}/{train_file}"

    if not os.path.exists(train_src):
        raise FileNotFoundError(train_src)

    #df_train = pd.read_csv(train_src, header=None, index_col = None, names=["Chromosome", "gene_name", "label", "site_id", "seq" ])
    df_train = pd.read_csv(train_src, header=None, index_col = None, names=["seq", "label"])
    df_train["seq"] = df_train["seq"].astype(str).apply(
        lambda x: trim_center(x, TARGET_LEN)
    )
    df_train.to_csv(train_out, index=False, header=False)
    print(f"[OK] train output: {train_out}")

    # ===== test =====
    test_src = f"{DATA_PATH}/{test_src}"
    test_out = f"{DATA_PATH}/{test_file}"

    if not os.path.exists(test_src):
        raise FileNotFoundError(test_src)

    #df_test = pd.read_csv(test_src, header=None, index_col = None, names=["Chromosome", "gene_name", "label", "site_id", "seq" ])
    df_test = pd.read_csv(test_src, header=None, index_col = None, names=["seq", "label"])
    df_test["seq"] = df_test["seq"].astype(str).apply(
        lambda x: trim_center(x, TARGET_LEN)
    )
    df_test.to_csv(test_out, index=False, header=False)
    print(f"[OK] test output: {test_out}")

    print("\n=== train & test trimming finished ===")



