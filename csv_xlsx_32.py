#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import openpyxl as px
import pandas as pd
import argparse
import os

columns_measure = ['Threshold', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'MCC', 'F1', 'AUC', 'AUPRC']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine_method', type=str)
    parser.add_argument('--encode_method', type=str)
    parser.add_argument('--species', type=str)
    parser.add_argument('--results_path', type=str, required=True)
    parser.add_argument('--lens', type=str, required=True)
    parser.add_argument('--deep_method', type=str, required=True)
    parser.add_argument('--deep_encode', type=str, required=True)
    args = parser.parse_args()

    machine_method_item = args.machine_method.strip().split(',')
    encode_method_item = args.encode_method.strip().split(',')
    species = args.species
    base_path = args.results_path.rstrip('/')
    lens = args.lens.strip().split(',')
    deep_method = args.deep_method.strip().split(',')
    deep_encode = args.deep_encode.strip().split(',')

    for seq_len in lens:

        outfile_name = f"result_{species}_{seq_len}.xlsx"
        print(f"\n=== Processing {species} {seq_len} nt ===")

        # ========= ML models ========= #
        for machine_method in machine_method_item:
            val_measure = []
            test_measure = []
            for encode_method in encode_method_item:
                infile_path = f"{base_path}/{species}/{seq_len}/{machine_method}/{encode_method}"
                infile1 = f"{infile_path}/val_measures.csv"
                infile2 = f"{infile_path}/test_measures.csv"

                if not os.path.exists(infile1) or not os.path.exists(infile2):
                    print(f"⚠ Missing file: {infile1} or {infile2}")
                    continue

                val_measure.append(pd.read_csv(infile1, index_col=0).iloc[-1].values.tolist())
                test_measure.append(pd.read_csv(infile2, index_col=0).iloc[-1].values.tolist())

            pd_val = pd.DataFrame(data=val_measure, index=encode_method_item, columns=columns_measure)
            pd_test = pd.DataFrame(data=test_measure, index=encode_method_item, columns=columns_measure)

            pd_val_test = pd.concat([pd_val, pd_test], axis=0)

            mode_f = 'a' if os.path.exists(outfile_name) else 'w'
            with pd.ExcelWriter(outfile_name, engine="openpyxl", mode=mode_f) as writer:
                pd_val_test.to_excel(writer, sheet_name=machine_method)

        # ========= DL models ========= #
        for machine_method in deep_method:
            val_measure = []
            test_measure = []
            for encode_method in deep_encode:
                infile_path = f"{base_path}/{species}/{seq_len}/{machine_method}/{encode_method}"
                infile1 = f"{infile_path}/val_measures.csv"
                infile2 = f"{infile_path}/test_measures.csv"

                if not os.path.exists(infile1) or not os.path.exists(infile2):
                    print(f"⚠ Missing DL file: {infile1} or {infile2}")
                    continue

                val_measure.append(pd.read_csv(infile1, index_col=0).iloc[-1].values.tolist())
                test_measure.append(pd.read_csv(infile2, index_col=0).iloc[-1].values.tolist())

            pd_val = pd.DataFrame(data=val_measure, index=deep_encode, columns=columns_measure)
            pd_test = pd.DataFrame(data=test_measure, index=deep_encode, columns=columns_measure)
            pd_val_test = pd.concat([pd_val, pd_test], axis=0)

            mode_f = 'a'
            with pd.ExcelWriter(outfile_name, engine="openpyxl", mode=mode_f) as writer:
                pd_val_test.to_excel(writer, sheet_name=machine_method)

        # ========= Stacking (ranking & combine) ========= #
        rank_file = f"{base_path}/{species}/{seq_len}/ranking.csv"
        if os.path.exists(rank_file):
            pd_rank = pd.read_csv(rank_file)
            with pd.ExcelWriter(outfile_name, engine="openpyxl", mode='a') as writer:
                pd_rank.to_excel(writer, sheet_name='ranking')

        comb_file = f"{base_path}/{species}/{seq_len}/combine/top_measure.csv"
        if os.path.exists(comb_file):
            pd_comb = pd.read_csv(comb_file)
            with pd.ExcelWriter(outfile_name, engine="openpyxl", mode='a') as writer:
                pd_comb.to_excel(writer, sheet_name='stack')

            maxAUC = 0
            maxStack = 1
            for i in range(int(pd_comb.shape[0] / 2)):
                if pd_comb.loc[2*i, 'AUC'] > maxAUC:
                    maxAUC = pd_comb.loc[2*i, 'AUC']
                    maxStack = 2*i

            with pd.ExcelWriter(outfile_name, engine="openpyxl", mode='a') as writer:
                pd_comb[maxStack:maxStack+2].to_excel(writer, sheet_name='top', index=True,)

