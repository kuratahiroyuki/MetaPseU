#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os

def import_txt(filename):
    all_data = []
    with open(filename) as f:
        reader = f.readlines()       
        for row in reader:
            all_data.append(row.replace("\n", "").split(','))          
    return pd.DataFrame(all_data, columns = ["seq", "label"])
 
def output_csv_pandas(filename, data):
    data.to_csv(filename, index = None)

#setting
path ='./dataset'
kfold = 5
type=["hg38","mm10","sacCer3"]

out_path=path

for i in type:
    training_data = import_txt(path + "/train_PseU_"+i+"_201_c0.7.txt")
    count=0
    skf = StratifiedKFold(n_splits = kfold, shuffle=True)
    for train_index, val_index in skf.split(training_data, training_data['label']):
        count += 1
        os.makedirs(out_path +"/train_PseU_"+i+"_201"+ "/cross_val/" + str(count), exist_ok = True)
        output_csv_pandas(out_path+"/train_PseU_"+i+"_201"+ "/cross_val/" + str(count) + "/cv_train_" + str(count) + ".csv", training_data.loc[train_index,:].reset_index(drop=True))
        output_csv_pandas(out_path+"/train_PseU_"+i+"_201"+ "/cross_val/" + str(count) + "/cv_val_" + str(count) + ".csv", training_data.loc[val_index,:].reset_index(drop=True))































































