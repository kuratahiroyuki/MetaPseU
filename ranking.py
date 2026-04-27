#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import os   


def ranking(measure_path,
            machine_method_item, encode_method_item,
            deep_method_item, deep_encode_item):

    unified_cols = [
        'Threshold', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy',
        'MCC', 'F1', 'AUC', 'AUPRC'
    ]

    val_measure = []

    def process_block(method_list, encode_list):
        nonlocal val_measure
        for m in method_list:
            for enc in encode_list:
                infile = f"{measure_path}/{m}/{enc}/val_measures.csv"

                if not os.path.exists(infile):
                    raise FileNotFoundError(f"Missing file: {infile}")

                df = pd.read_csv(infile, index_col=0)
                row = df.iloc[-1].reindex(unified_cols)

                val_measure.append([m, enc] + row.values.tolist())

    process_block(machine_method_item, encode_method_item)  # ML
    process_block(deep_method_item, deep_encode_item)        # DL

    df = pd.DataFrame(val_measure, columns=['Machine', 'Encode'] + unified_cols)
    df_sorted = df.sort_values('AUC', ascending=False)
    df_sorted.to_csv(f"{measure_path}/ranking.csv", index=False)

    return df_sorted[['Machine', 'Encode']].values.tolist()


    
if __name__=='__main__':

    DRS='DRS10_21'
    measure_path = "../data/result/%s" %DRS 
    machine_method_item = ['LGBM','RF','SVM','XGB']
    encode_method_item = ["binary","RCKmer","DNC","TNC","ENAC","CKSNAP","NCP","ANF","EIIP","PseEIIP"] 
    # ['RF','SVM','XGB','LGBM']
    # "binary","RCKmer","DNC","TNC","ENAC","CKSNAP","NCP","ANF","EIIP","PseEIIP" 

    combination = ranking(measure_path, machine_method_item, encode_method_item)  
    
    print(combination)








 


