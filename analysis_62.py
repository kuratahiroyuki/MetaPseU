import numpy as np
import pandas as pd
import argparse
import prettytable as pt
from valid_metrices_p21 import *


def measure_evaluation(score_val, score_test, inpath, val_file, test_file, kfold, threshold=None):  
    for i in range(kfold):
    
        infile = inpath + '/' + str(i+1) + '/' + val_file
        result = np.loadtxt(infile, delimiter=',', skiprows=1)
        prob=result[:,1]
        label=result[:,2]
        
        th_, rec_, pre_, f1_, spe_, acc_, mcc_, auc_, pred_class, prauc_ = eval_metrics(prob, label) 
        valid_matrices = th_, rec_, pre_, f1_, spe_, acc_, mcc_, auc_, prauc_

        score_val.iloc[i,0]= th_
        score_val.iloc[i,1]= rec_
        score_val.iloc[i,2]= spe_
        score_val.iloc[i,3]= pre_   
        score_val.iloc[i,4]= acc_
        score_val.iloc[i,5]= mcc_
        score_val.iloc[i,6]= f1_ 
        score_val.iloc[i,7]= auc_                     
        score_val.iloc[i,8]= prauc_

        infile = inpath + '/' + str(i+1) + '/' + test_file
        result = np.loadtxt(infile, delimiter=',', skiprows=1)
        prob=result[:,1]
        label=result[:,2]
        
        th_, rec_, pre_, f1_, spe_, acc_, mcc_, auc_, pred_class, prauc_ = th_eval_metrics(th_, prob, label)
        test_matrices = th_, rec_, pre_, f1_, spe_, acc_, mcc_, auc_, prauc_
            
        score_test.iloc[i,0]= th_
        score_test.iloc[i,1]= rec_
        score_test.iloc[i,2]= spe_
        score_test.iloc[i,3]= pre_   
        score_test.iloc[i,4]= acc_
        score_test.iloc[i,5]= mcc_
        score_test.iloc[i,6]= f1_ 
        score_test.iloc[i,7]= auc_                     
        score_test.iloc[i,8]= prauc_
        
        #print_table(print_results(valid_matrices, test_matrices) )
        
    means = score_val.astype(float).mean(axis='index')
    means = pd.DataFrame(np.array(means).reshape(1,-1), index= ['means'], columns=columns_measure)
    score_val = pd.concat([score_val, means])                   

    means = score_test.astype(float).mean(axis='index')
    means = pd.DataFrame(np.array(means).reshape(1,-1), index= ['means'], columns=columns_measure)
    score_test = pd.concat([score_test, means]) 
    
    return score_val, score_test


if __name__=='__main__' :
    parser = argparse.ArgumentParser()

    parser.add_argument('--machine_method', type=str, required=True)
    parser.add_argument('--encode_method', type=str, required=True)
    parser.add_argument('--species', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--lens', type=str, required=True)  
    parser.add_argument('--deep_method', type=str, required=True)
    parser.add_argument('--deep_encode_method', type=str, required=True)
    

    args = parser.parse_args()

    machine_method_item = args.machine_method.strip().split(',')
    encode_method_item  = args.encode_method.strip().split(',')
    species = args.species
    data_path = args.data_path
    lens = args.lens.strip().split(',')
    deep_method_item = args.deep_method.strip().split(',')
    deep_encode_method_item  = args.deep_encode_method.strip().split(',')

    kfold=5

    test_file='test_roc.csv'
    val_file='val_roc.csv'
    val_measure ='val_measures.csv'
    test_measure ='test_measures.csv'

    index_fold =[i+1 for i in range(kfold)] 
    columns_measure= ['Threshold', 'Sensitivity', 'Specificity', 
                       'Precision', 'Accuracy', 'MCC', 'F1', 'AUC', 'AUPRC']

    for len in lens:
        for machin_method in machine_method_item:  
            for encode_method in encode_method_item:             

                inpath = data_path + '/' + species + '/' + len + "/" + machin_method + '/' + encode_method

                score_val  = pd.DataFrame(data=[], index=index_fold, columns=columns_measure)
                score_test = pd.DataFrame(data=[], index=index_fold, columns=columns_measure)

                score_val, score_test = measure_evaluation(
                    score_val, score_test, inpath, 
                    val_file, test_file, kfold, threshold=None
                )

                score_val.to_csv('%s/val_measures.csv' % inpath, header=True, index=True)        
                score_test.to_csv('%s/test_measures.csv' % inpath, header=True, index=True)

                print(score_val)
                print(score_test)
    
    for len in lens:
        for machin_method in deep_method_item:  
            for encode_method in deep_encode_method_item:             

                inpath = data_path + '/' + species + '/' + len + "/" + machin_method + '/' + encode_method

                score_val  = pd.DataFrame(data=[], index=index_fold, columns=columns_measure)
                score_test = pd.DataFrame(data=[], index=index_fold, columns=columns_measure)

                score_val, score_test = measure_evaluation(
                    score_val, score_test, inpath, 
                    val_file, test_file, kfold, threshold=None
                )

                score_val.to_csv('%s/val_measures.csv' % inpath, header=True, index=True)        
                score_test.to_csv('%s/test_measures.csv' % inpath, header=True, index=True)

                print(score_val)
                print(score_test)
