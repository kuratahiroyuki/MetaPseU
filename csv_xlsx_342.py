#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import openpyxl as px
import pandas as pd
import argparse
import os

columns_measure= ['Threshold', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'MCC', 'F1', 'AUC', 'AUPRC']

if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine_method', type=str, help='term')
    parser.add_argument('--machine_encode', type=str, help='term')
    parser.add_argument('--deep_method', type=str, help='term')
    parser.add_argument('--deep_encode', type=str, help='term')
    parser.add_argument('--result_path', type=str, help='term')
    parser.add_argument('--outfile', type=str, help='filem')
    parser.add_argument('--species', type=str, help='filem')
    parser.add_argument('--seqwin', type=int, help='value')
    args = parser.parse_args()
    
    machine_method = args.machine_method.strip().split(',')
    machine_encode  = args.machine_encode.strip().split(',')
    deep_method = args.deep_method .strip().split(',')
    deep_encode = args.deep_encode.strip().split(',')
    result_path = args.result_path
    species = args.species
    seqwin = args.seqwin
    outfile_name = args.outfile

    infile_name = ["val_measures.csv", "test_measures.csv" ]

    for method in machine_method :
        val_measure=[]
        test_measure=[]
        for i, encode_method in enumerate(machine_encode):

          infile_path = "%s/%s/%s/%s/%s" %(result_path, species, seqwin, method, encode_method)
          infile1 = infile_path + '/' + infile_name[0] #val
          infile2 = infile_path + '/' + infile_name[1] #test

          val_measure.append(  (pd.read_csv(infile1, index_col=0).iloc[-1].values.tolist())) # means
          test_measure.append( (pd.read_csv(infile2, index_col=0).iloc[-1].values.tolist())) # means

        print(f'{val_measure}, {encode_method}')
        
        pd_val_measure  = pd.DataFrame(data=val_measure, index=machine_encode, columns=columns_measure)
        pd_test_measure = pd.DataFrame(data=test_measure, index=machine_encode, columns=columns_measure)

        print(pd_val_measure)
        print(pd_test_measure)

        pd_val_test = pd.concat([pd_val_measure, pd_test_measure], axis=0)

        if os.path.exists(outfile_name) == True:
            mode_f ='a'
            with pd.ExcelWriter(outfile_name, engine="openpyxl", mode = mode_f, if_sheet_exists="replace") as writer: 
              pd_val_test.to_excel(writer, sheet_name = method) #index=False, header=False
        else :
            mode_f ='w' 
            with pd.ExcelWriter(outfile_name, engine="openpyxl", mode = mode_f) as writer: 
              pd_val_test.to_excel(writer, sheet_name = method) #index=False, header=False

    for method in deep_method:
        val_measure=[]
        test_measure=[]
        for i, encode_method in enumerate(deep_encode):

          infile_path = "%s/%s/%s/%s/%s" %(result_path, species, seqwin, method, encode_method)
          infile1 = infile_path + '/' + infile_name[0] #val
          infile2 = infile_path + '/' + infile_name[1] #test

          val_measure.append(  (pd.read_csv(infile1, index_col=0).iloc[-1].values.tolist())) # means
          test_measure.append( (pd.read_csv(infile2, index_col=0).iloc[-1].values.tolist())) # means

        pd_val_measure  = pd.DataFrame(data=val_measure, index=deep_encode, columns=columns_measure)
        pd_test_measure = pd.DataFrame(data=test_measure, index=deep_encode, columns=columns_measure)

        print(pd_val_measure)
        print(pd_test_measure)

        pd_val_test = pd.concat([pd_val_measure, pd_test_measure], axis=0)

        if os.path.exists(outfile_name) == True:
            mode_f ='a'
            with pd.ExcelWriter(outfile_name, engine="openpyxl", mode = mode_f, if_sheet_exists="replace") as writer: 
              pd_val_test.to_excel(writer, sheet_name = method) #index=False, header=False
        else :
            mode_f ='w' 
            with pd.ExcelWriter(outfile_name, engine="openpyxl", mode = mode_f) as writer: 
              pd_val_test.to_excel(writer, sheet_name = method) #index=False, header=False


    rank_file = result_path + '/%s/%s/ranking.csv' %(species, seqwin)
    pd_rank = pd.read_csv(rank_file)
    with pd.ExcelWriter(outfile_name, engine="openpyxl", mode = 'a', if_sheet_exists="replace") as writer: 
        pd_rank.to_excel(writer, sheet_name = 'ranking')
        
    comb_file='%s/%s/%s/combine/top_measure.csv' %(result_path,species, seqwin)
    pd_comb = pd.read_csv(comb_file)
    with pd.ExcelWriter(outfile_name, engine="openpyxl", mode = 'a', if_sheet_exists="replace") as writer: 
        pd_comb.to_excel(writer, sheet_name = 'stack')

    rank_file = result_path + '/%s/%s/weight_ranking.csv' %(species, seqwin)
    pd_rank = pd.read_csv(rank_file)
    with pd.ExcelWriter(outfile_name, engine="openpyxl", mode = 'a', if_sheet_exists="replace") as writer: 
        pd_rank.to_excel(writer, sheet_name = 'weight_ranking')
        
    comb_file='%s/%s/%s/sel_combine/sel_measure.csv' %(result_path, species, seqwin)
    pd_comb = pd.read_csv(comb_file)
    with pd.ExcelWriter(outfile_name, engine="openpyxl", mode = 'a', if_sheet_exists="replace") as writer: 
        pd_comb.to_excel(writer, sheet_name = 'select')
 
    maxAUC=0
    maxStack=1
    for i in range(int(pd_comb.shape[0]/2)):
        if pd_comb.loc[2*i,'AUC'] > maxAUC :
            maxAUC=pd_comb.loc[2*i,'AUC']
            maxStack=2*i

    with pd.ExcelWriter(outfile_name, engine="openpyxl", mode = 'a', if_sheet_exists="replace") as writer: 
        pd_comb[maxStack:maxStack+2].to_excel(writer, sheet_name = 'top', index=True) 

        




 


