# from ml_av2.py
import os
import time
import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost as xgb
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from gensim.models import word2vec
from sklearn.metrics import r2_score,  mean_squared_error, mean_absolute_error
from libml.encodingRNA_2 import Kmer, RCKmer, DNC, TNC, ENAC, binary, CKSNAP, NCP, ANF, EIIP, PseEIIP, PSTNPss
from libml import ACC
from libml import Pse
from libml import check_parameters

rna_dict = {'A':4,'C':5,'G':6,'U':7,'-':19} # for RNAFM

def emb_seq_w2v(seq_mat, w2v_model, num):
    num_sample = len(seq_mat)    
    seq_emb=[]  
    for j in range(num_sample):
      if j%1000==0:
        print(f'j: {j}')
      seq=seq_mat[j]
      seq_emb.append( np.array([np.array(w2v_model.wv[seq[i:i+num]]) for i in range(len(seq) - num + 1)]) )
    seq_emb = np.array(seq_emb)        
    seq_emb = seq_emb.reshape(num_sample,len(seq) - num + 1, -1)  #74446 x 41 x 4
    seq_emb = seq_emb.reshape(num_sample,1,-1).squeeze() #74446 x 41 x 4
    return seq_emb

def nuc_dict_construction():
   AA = 'ACGU'
   keys=[]
   vectors=[]
   for i, key in enumerate(AA) :
      base=np.zeros(4)
      keys.append(key)
      base[i]=1
      vectors.append(base)
   nuc_dict = dict(zip(keys, vectors))
   return nuc_dict
   
def emb_seq_BE(seq_mat, nuc_dict, num):
   num_sample = len(seq_mat) 
   
   seq_emb=[]
   for j in range(num_sample):
      seq = seq_mat[j]
      seq_emb.append(np.array([np.array([nuc_dict[seq[i + k]] for k in range(num)]).reshape([4 * num]) for i in range(len(seq) - num + 1)]))
   seq_emb = np.array(seq_emb)
   seq_emb = seq_emb.reshape(num_sample, len(seq) - num + 1, -1)  #74446 x 41 x 4
   seq_emb = seq_emb.reshape(num_sample, 1, -1).squeeze() 
   return seq_emb
   

def pad_input_csv(filename, seqwin, index_col = None):
    df1 = pd.read_csv(filename, delimiter=',',index_col = index_col)
    seq = df1.loc[:,'seq'].tolist()
    #data triming and padding
    for i in range(len(seq)):
       if len(seq[i]) > seqwin:
         seq[i]=seq[i][0:seqwin]
       seq[i] = seq[i].ljust(seqwin, '-')
    df1['seq'] = seq

    return df1


def pad_input_csv_rnafm(filename, seqwin, index_col = None):
    df1 = pd.read_csv(filename, delimiter=',',index_col = index_col)
    seqs = df1.loc[:,'seq'].tolist()
    #data triming and padding
    sequence = []
    rna_fm_token = []
    for seq in seqs:
        if len(seq) > seqwin:
            seq = seq[0:seqwin]
        sequence.append( seq.ljust(seqwin, '-') )        
        rna_fm_token.append([rna_dict[res] for res in seq])
      
    df1['seq'] = sequence
    df1['token'] = rna_fm_token

    return df1
    
      
def pickle_save(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def pickle_read(path):
    with open(path, "rb") as f:
        res = pickle.load(f)      
    return res
    
def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)

def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data    


#############################################################################################
if __name__ == '__main__' :
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--intrain', help='Path')
    parser.add_argument('--intest', help='Path')
    parser.add_argument('--outpath', help='Path')
    parser.add_argument('--machine', help='Path')
    parser.add_argument('--encode', help='Path')
    parser.add_argument('--fold', type=int, help='Path')
    parser.add_argument('--seqwin', type=int, help='Path')
    parser.add_argument('--type', help='Path')

    parser.add_argument('--lag', type=int, help='Path')
    parser.add_argument('--weight', type=float, help='Path')
    parser.add_argument('--lamadaValue', type=int, help='Path')
    parser.add_argument('--kmer', type=int, help='Path') 
       
    parser.add_argument('--w2vmodel', help='Path')
    parser.add_argument('--kmer_w2v', type=int, help='Path') 

    parser.add_argument('--index',
                        help="The indices file user choose.\n"
                             "Default indices:\n"
                             "DNA dinucleotide: Rise, Roll, Shift, Slide, Tilt, Twist.\n"
                             "DNA trinucleotide: Dnase I, Bendability (DNAse).\n"
                             "RNA: Rise, Roll, Shift, Slide, Tilt, Twist.\n")
    parser.add_argument("--udi", help="The user-defined indices file.")
    parser.add_argument('--all_index', action='store_true', help="Choose all physico-chemical indices, default: False.")

    args = parser.parse_args()
    path = args.intrain
    test_file = args.intest
    out_path_0 = args.outpath   
    machine_method_item = args.machine.strip().split(',')
    encode_method_item = args.encode.strip().split(',')
    kfold = args.fold
    seqwin = args.seqwin
    
    kmer = args.kmer
    kmer_w2v = args.kmer_w2v
   
    for machine_method in machine_method_item:
        for encode_method in encode_method_item: 
            args.encode=encode_method ###
            print(args.encode)
            if 'W2V' in encode_method :
                w2v_model = word2vec.Word2Vec.load(args.w2vmodel)
                os.makedirs(out_path_0 + '/' + machine_method + '/' + encode_method, exist_ok=True)
                out_path =  out_path_0 + '/' + machine_method + '/' + encode_method
            else:
                os.makedirs(out_path_0 + '/' + machine_method + '/' + encode_method, exist_ok=True)
                out_path =  out_path_0 + '/' + machine_method + '/' + encode_method
                                        
            for i in range(1, kfold+1):
                os.makedirs(out_path + "/" + str(i) + "/data_model", exist_ok=True)
                modelname= "machine_model.sav"
            
                if encode_method == 'RNAFM':
                    train_dataset = pad_input_csv_rnafm(path + "/" + str(i) + "/cv_train_" + str(i) + ".csv", seqwin, index_col = None) #'seq', 'label', 'token'
                    valid_dataset =  pad_input_csv_rnafm(path + "/" + str(i) + "/cv_val_" + str(i) + ".csv", seqwin, index_col = None)
                    test_dataset = pad_input_csv_rnafm(test_file, seqwin, index_col = None)

                else:
                    train_dataset = pad_input_csv(path + "/" + str(i) + "/cv_train_" + str(i) + ".csv", seqwin, index_col = None) #'seq', 'label'
                    valid_dataset = pad_input_csv(path + "/" + str(i) + "/cv_val_" + str(i) + ".csv", seqwin, index_col = None)
                    test_dataset = pad_input_csv(test_file, seqwin, index_col = None)      
          
                if 'W2V' in encode_method:
                    train_seq = train_dataset['seq'].tolist()
                    valid_seq = valid_dataset['seq'].tolist()
                    test_seq = test_dataset['seq'].tolist()
                    train_X = emb_seq_w2v(train_seq, w2v_model, kmer_w2v)
                    valid_X = emb_seq_w2v(valid_seq, w2v_model, kmer_w2v)       
                    test_X = emb_seq_w2v(test_seq, w2v_model, kmer_w2v)      
                elif encode_method == 'RNAFM':
                    train_token = train_dataset['token'].tolist()
                    valid_token = valid_dataset['token'].tolist()
                    test_token = test_dataset['token'].tolist()

                    train_X = rna_fm_encoding(train_token)
                    valid_X = rna_fm_encoding(valid_token)
                    test_X = rna_fm_encoding(test_token)
                    print(f'valid_X {valid_X}')
                else:
                    kw = {}
                    train_seq_label=train_dataset.values.tolist()
                    valid_seq_label  =valid_dataset.values.tolist()
                    test_seq_label =test_dataset.values.tolist()
                    #print(f'valid_seq_label {train_seq_label}')   
                    #print(f'valid_seq_label {len(train_seq_label)}')

                    if encode_method == 'Kmer':
                        train_X = np.array(Kmer(train_seq_label, **kw), dtype=float)
                        valid_X = np.array(Kmer(valid_seq_label, **kw), dtype=float)
                        test_X = np.array(Kmer(test_seq_label, **kw), dtype=float)
                        #print(f'test_X {test_X}')
                        
                    elif encode_method == 'RCKmer':
                        train_X = np.array(RCKmer(train_seq_label, **kw), dtype=float) #k=2
                        valid_X = np.array(RCKmer(valid_seq_label, **kw), dtype=float)
                        test_X = np.array(RCKmer(test_seq_label, **kw), dtype=float)

                    elif encode_method == 'DNC':
                        train_X = np.array(DNC(train_seq_label, **kw), dtype=float) 
                        valid_X = np.array(DNC(valid_seq_label, **kw), dtype=float)
                        test_X = np.array(DNC(test_seq_label, **kw), dtype=float)               

                    elif encode_method == 'TNC':
                        train_X = np.array(TNC(train_seq_label, **kw), dtype=float) 
                        valid_X = np.array(TNC(valid_seq_label, **kw), dtype=float)
                        test_X = np.array(TNC(test_seq_label, **kw), dtype=float) 
                         
                    elif encode_method == 'ENAC':
                        kw = {'order': 'ACGU'}
                        train_X = np.array(ENAC(train_seq_label, **kw), dtype=float) 
                        valid_X = np.array(ENAC(valid_seq_label, **kw), dtype=float)
                        test_X = np.array(ENAC(test_seq_label, **kw), dtype=float)  

                    elif encode_method == 'binary':
                        train_X = np.array(binary(train_seq_label, **kw), dtype=float) 
                        valid_X = np.array(binary(valid_seq_label, **kw), dtype=float)
                        test_X = np.array(binary(test_seq_label, **kw), dtype=float)  

                    elif encode_method == 'CKSNAP':
                        kw = {'order': 'ACGU'}
                        train_X = np.array(CKSNAP(train_seq_label, **kw), dtype=float) 
                        valid_X = np.array(CKSNAP(valid_seq_label, **kw), dtype=float)
                        test_X = np.array(CKSNAP(test_seq_label, **kw), dtype=float)  
                        
                    elif encode_method == 'NCP':
                        train_X = np.array(NCP(train_seq_label, **kw), dtype=float) 
                        valid_X = np.array(NCP(valid_seq_label, **kw), dtype=float)
                        test_X = np.array(NCP(test_seq_label, **kw), dtype=float)  

                    elif encode_method == 'ANF':
                        train_X = np.array(ANF(train_seq_label, **kw), dtype=float) 
                        valid_X = np.array(ANF(valid_seq_label, **kw), dtype=float)
                        test_X = np.array(ANF(test_seq_label, **kw), dtype=float)  
                        
                    elif encode_method == 'PSTNPss': 
                        train_X = np.array(PSTNPss(train_seq_label, **kw), dtype=float) 
                        valid_X = np.array(PSTNPss(valid_seq_label, **kw), dtype=float)
                        test_X = np.array(PSTNPss(test_seq_label, **kw), dtype=float)              
                        
                    elif encode_method == 'EIIP':
                        train_X = np.array(EIIP(train_seq_label, **kw), dtype=float) 
                        valid_X = np.array(EIIP(valid_seq_label, **kw), dtype=float)
                        test_X = np.array(EIIP(test_seq_label, **kw), dtype=float)                                                                           

                    elif encode_method == 'PseEIIP':
                        train_X = np.array(PseEIIP(train_seq_label, **kw), dtype=float) 
                        valid_X = np.array(PseEIIP(valid_seq_label, **kw), dtype=float)
                        test_X = np.array(PseEIIP(test_seq_label, **kw), dtype=float)
                        
                    elif encode_method == 'DAC' :
                        my_property_name, my_property_value, kmer = check_parameters.check_acc_arguments(args)
                        train_X = np.array(ACC.make_ac_vector(train_seq_label, my_property_name, my_property_value, args.lag, kmer), dtype=float) 
                        valid_X = np.array(ACC.make_ac_vector(valid_seq_label, my_property_name, my_property_value, args.lag, kmer), dtype=float)
                        test_X  = np.array(ACC.make_ac_vector(test_seq_label, my_property_name, my_property_value, args.lag, kmer), dtype=float)
                        
                    elif encode_method == 'DCC' :
                        my_property_name, my_property_value, kmer = check_parameters.check_acc_arguments(args)
                        encodings = ACC.make_cc_vector(train_seq_label, my_property_name, my_property_value, args.lag, kmer)
                        train_X = np.array(ACC.make_cc_vector(train_seq_label, my_property_name, my_property_value, args.lag, kmer), dtype=float) 
                        valid_X = np.array(ACC.make_cc_vector(valid_seq_label, my_property_name, my_property_value, args.lag, kmer), dtype=float)
                        test_X  = np.array(ACC.make_cc_vector(test_seq_label, my_property_name, my_property_value, args.lag, kmer), dtype=float)
                                               
                    elif encode_method == 'DACC':
                        my_property_name, my_property_value, kmer = check_parameters.check_acc_arguments(args)
                        #encodings = ACC.make_acc_vector(train_seq_label, my_property_name, my_property_value, args.lag, kmer)
                        train_X = np.array(ACC.make_acc_vector(train_seq_label, my_property_name, my_property_value, args.lag, kmer), dtype=float) 
                        valid_X = np.array(ACC.make_acc_vector(valid_seq_label, my_property_name, my_property_value, args.lag, kmer), dtype=float)
                        test_X  = np.array(ACC.make_acc_vector(test_seq_label, my_property_name, my_property_value, args.lag, kmer), dtype=float)
                        
                    elif encode_method == 'PseDNC':
                        my_property_name, my_property_value, lamada_value, weight, kmer = check_parameters.check_Pse_arguments(args, train_seq_label)
                        #encodings = Pse.make_PseDNC_vector(train_seq_label, my_property_name, my_property_value, lamada_value, weight)
                        train_X = np.array(Pse.make_PseDNC_vector(train_seq_label, my_property_name, my_property_value, lamada_value, weight), dtype=float) 
                        valid_X = np.array(Pse.make_PseDNC_vector(valid_seq_label, my_property_name, my_property_value, lamada_value, weight), dtype=float)
                        test_X  = np.array(Pse.make_PseDNC_vector(test_seq_label, my_property_name, my_property_value, lamada_value, weight), dtype=float)
                                      
                    elif encode_method == 'PseKNC':
                        my_property_name, my_property_value, lamada_value, weight, kmer = check_parameters.check_Pse_arguments(args, train_seq_label)
                        #encodings = Pse.make_PseKNC_vector(train_seq_label, my_property_name, my_property_value, lamada_value, weight, kmer)
                        train_X = np.array(Pse.make_PseKNC_vector(train_seq_label, my_property_name, my_property_value, lamada_value, weight, kmer), dtype=float) 
                        valid_X = np.array(Pse.make_PseKNC_vector(valid_seq_label, my_property_name, my_property_value, lamada_value, weight, kmer), dtype=float)
                        test_X  = np.array(Pse.make_PseKNC_vector(test_seq_label, my_property_name, my_property_value, lamada_value, weight, kmer), dtype=float)
                                                   
                    elif encode_method == 'PCPseDNC': #duplication PseDNC
                        my_property_name, my_property_value, lamada_value, weight, kmer = check_parameters.check_Pse_arguments(args, train_seq_label)
                        #encodings = Pse.make_PseDNC_vector(train_seq_label, my_property_name, my_property_value, lamada_value, weight)
                        train_X = np.array(Pse.make_PseDNC_vector(train_seq_label, my_property_name, my_property_value, lamada_value, weight), dtype=float) 
                        valid_X = np.array(Pse.make_PseDNC_vector(valid_seq_label, my_property_name, my_property_value, lamada_value, weight), dtype=float)
                        test_X  = np.array(Pse.make_PseDNC_vector(test_seq_label, my_property_name, my_property_value, lamada_value, weight), dtype=float)
                               
                    elif encode_method == 'SCPseDNC':
                        my_property_name, my_property_value, lamada_value, weight, kmer = check_parameters.check_Pse_arguments(args, train_seq_label)
                        #encodings = Pse.make_SCPseDNC_vector(train_seq_label, my_property_name, my_property_value, lamada_value, weight)
                        train_X = np.array(Pse.make_SCPseDNC_vector(train_seq_label, my_property_name, my_property_value, lamada_value, weight), dtype=float) 
                        valid_X = np.array(Pse.make_SCPseDNC_vector(valid_seq_label, my_property_name, my_property_value, lamada_value, weight), dtype=float)
                        test_X  = np.array(Pse.make_SCPseDNC_vector(test_seq_label, my_property_name, my_property_value, lamada_value, weight), dtype=float)
                                                                                                                                                                                                              
                    else :
                        pass
                        print('No encode method')
                        exit()
                        
                train_y = train_dataset['label'].to_numpy()    
                valid_y = valid_dataset['label'].to_numpy() 
                test_y = test_dataset['label'].to_numpy()

                train_result = np.zeros((len(train_y), 2))
                train_result[:, 1] = train_y                      
                cv_result = np.zeros((len(valid_y), 2))
                cv_result[:, 1] = valid_y
                test_result = np.zeros((len(test_y), 2))
                test_result[:,1] = test_y   #score:one of two, label
             
                if machine_method == 'RF':
                    model = RandomForestClassifier(max_depth=4, random_state=0, n_estimators=100)
                    clf = model.fit(train_X, train_y)
                    
                elif machine_method == 'NB':
                    model = GaussianNB() # 正規分布を仮定したベイズ分類
                    clf = model.fit(train_X, train_y)
                    
                elif machine_method == 'KN':
                    model = KNeighborsClassifier()
                    clf = model.fit(train_X, train_y)

                elif machine_method == 'LR':
                    model = LogisticRegression(random_state=0)
                    clf = model.fit(train_X, train_y)                    
                    
                elif machine_method == 'SVM':    
                    model = svm.SVC(probability=True)
                    clf = model.fit(train_X, train_y)
              
                elif machine_method == 'XGB':
                    xgb_train = xgb.DMatrix(train_X, train_y)
                    xgb_eval  = xgb.DMatrix(valid_X , valid_y)
                    params = {
                    "learning_rate": 0.01,
                    "max_depth": 3
                    }
                    clf = xgb.train(params, 
                    xgb_train, evals=[(xgb_train, "train"), (xgb_eval, "validation")], 
                    num_boost_round=100, early_stopping_rounds=20)
                 
                elif machine_method == 'LGBM':   
                    lgb_train = lgb.Dataset(train_X , train_y)
                    lgb_eval = lgb.Dataset(valid_X , valid_y, reference=lgb_train)
                    params = {         
                    'objective': 'binary',# 二値分類問題         
                    'metric': 'auc',# AUC の最大化を目指す          
                    'verbosity': -1,# Fatal の場合出力
                    'random_state': 123,
                    }
                    clf = lgb.train(
                        params,
                        lgb_train,
                        valid_sets=lgb_eval,
                        num_boost_round=1000,
                        callbacks=[
                            lgb.early_stopping(100),     # ← これが early_stopping_rounds の代わり
                            lgb.log_evaluation(50)       # 50イテレーションごとにログ出力
                        ]
                    )


                else:
                    print('No learning method')
                    exit()
                
                #pickle.dump(clf, open(out_path + "/" + str(i) + "/data_model/machine_model.asv",'wb'))  
              
              #CV
                if machine_method == 'LGBM':  
                    train_result[:, 0] = clf.predict(train_X, num_iteration=clf.best_iteration) 
                    score = clf.predict(valid_X, num_iteration=clf.best_iteration)
                    cv_result[:, 0] = score
                elif machine_method == 'XGB':  
                    train_result[:, 0] = clf.predict(xgb_train)
                    score = clf.predict(xgb_eval)
                    cv_result[:, 0] = score
                else :
                    score = clf.predict_proba(train_X)
                    train_result[:, 0] = score[:,1]       
                    score = clf.predict_proba(valid_X)
                    cv_result[:, 0] = score[:,1]
                  
                #print(cv_result)
              
                #independent test
                if test_dataset.shape[0] != 0:
                    if machine_method == 'LGBM':
                        test_result[:, 0] = clf.predict(test_X, num_iteration=clf.best_iteration)
                    elif machine_method == 'XGB': 
                        test_result[:, 0] = clf.predict(xgb.DMatrix(test_X))
                    else:
                        test_result[:, 0] = clf.predict_proba(test_X)[:,1]
              
            
                #print(test_result)
                train_output = pd.DataFrame(train_result,  columns=['prob', 'label'] )
                train_output.to_csv(out_path  + "/" + str(i) + "/train_roc.csv")  #prob, label   
                #CV  
                cv_output = pd.DataFrame(cv_result,  columns=['prob', 'label'] )
                cv_output.to_csv(out_path  + "/" + str(i) + "/val_roc.csv")  #prob, label

                #independent test
                test_output = pd.DataFrame(test_result,  columns=['prob', 'label'] )
                test_output.to_csv(out_path  + "/" + str(i) + "/test_roc.csv")  #prob, label
           
            print(f'machine: {machine_method}, encode: {encode_method}' )
            print('elapsed time', time.time() - start)

