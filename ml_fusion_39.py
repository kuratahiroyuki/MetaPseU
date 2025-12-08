import os
import sys
import pickle
import pandas as pd
import numpy as np
import time
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
sys.path.append(os.path.abspath(''))
sys.path.append(os.path.abspath('..'))
from valid_metrices_p21 import *
from ranking import ranking

item_column =['Thre','Rec','Spe','Pre','Acc','MCC','F1','AUC','PRAUC'] 

def combine_model(valid_data, test_data, data_path, out_dir, kfold, columns):  
    prediction_result_cv = []
    prediction_result_test = []
    
    train_y, train_X = valid_data[:,-1], valid_data[:,:-1]
    valid_y, valid_X = valid_data[:,-1], valid_data[:,:-1]
    test_y, test_X = test_data[:,-1], test_data[:,:-1]
    
    model = LogisticRegression(random_state=0)
    #model = RandomForestClassifier(max_depth=4, random_state=0, n_estimators=100)
    #model = GaussianNB()
    #model = KNeighborsClassifier()
    #model = svm.SVC(probability=True)
    
    rfc = model.fit(train_X, train_y)
    
    os.makedirs('%s/%s/%s' % (data_path, out_dir, kfold), exist_ok=True)
    
    """
    if os.path.isfile('%s/%s/%s/lr_model.asv' % (data_path, out_dir, kfold)) :
        pass
    else:
       os.makedirs('%s/%s/%s' % (data_path, out_dir, kfold), exist_ok=True)
    pickle.dump(rfc, open('%s/%s/%s/lr_model.asv' % (data_path, out_dir, kfold), 'wb'))
    """
    
    """
    if os.path.isfile('%s/%s/%s/lr_model.asv' % (data_path, out_dir, kfold)):
        rfc = pickle.load(open('%s/%s/%s/lr_model.asv' % (data_path, out_dir, kfold), 'rb'))
    else :
        print('No %s model' %out_dir)
    """
                 
    scores = rfc.predict_proba(valid_X)
    tmp_result = np.zeros((len(valid_y), 1+2))
    tmp_result[:, 0], tmp_result[:, 1:] = valid_y, scores    
    prediction_result_cv.append(tmp_result)
             
    if test_X.shape[0] != 0:
        scores_test = rfc.predict_proba(test_X)
        tmp_result_test = np.zeros((len(test_y), 1+2))
        tmp_result_test[:, 0], tmp_result_test[:, 1:] = test_y, scores_test    
        prediction_result_test.append(tmp_result_test)           
        #print(prediction_result_cv[0]) #[1  0.094, 0.9057 ]
    
    valid_probs = prediction_result_cv[0][:,2]
    valid_labels = prediction_result_cv[0][:,0]    
    #print(np.array([valid_probs, valid_labels]).T)
    
    test_probs = prediction_result_test[0][:,2]
    test_labels = prediction_result_test[0][:,0]   

    cv_output = pd.DataFrame(np.array([valid_probs, valid_labels]).T,  columns=['prob', 'label'] )
    cv_output.to_csv('%s/%s/%s/val_roc.csv' % (data_path, out_dir, kfold))  
    
    df_valid = pd.DataFrame(valid_data, columns = columns)
    df_valid.to_csv('%s/%s/%s/val_roc_com.csv' % (data_path, out_dir, kfold))
    df_test = pd.DataFrame(test_data, columns=columns)
    df_test.to_csv('%s/%s/%s/test_roc_com.csv' % (data_path, out_dir, kfold))
     
    test_output = pd.DataFrame(np.array([test_probs, test_labels]).T,  columns=['prob', 'label'] )
    test_output.to_csv('%s/%s/%s/test_roc.csv' % (data_path, out_dir, kfold))
    
    #print(f'validation: prob label {valid_probs} {valid_labels} ')
    
    # metrics calculation
    th_, rec_, pre_, f1_, spe_, acc_, mcc_, auc_, pred_class, prauc_ = eval_metrics(valid_probs, valid_labels) 
    valid_matrices = th_, rec_, spe_, pre_,  acc_, mcc_, f1_, auc_, prauc_
    th_, rec_, pre_, f1_, spe_, acc_, mcc_, auc_, pred_class, prauc_ = th_eval_metrics(th_, test_probs, test_labels)
    test_matrices = th_, rec_, spe_, pre_,  acc_, mcc_, f1_, auc_, prauc_

    print_results(valid_matrices, test_matrices) 
    #print(f'valid_matrices {valid_matrices}')  
    
    df = pd.DataFrame([valid_matrices, test_matrices], index=['valid','test'], columns=item_column)
    df2 = pd.DataFrame([test_matrices], index=['test'], columns=item_column)
    
    return df, df2


def train_test(kfold, data_path, out_dir, combination, columns):
    #feature combine for each fold
    valid_data =[]
    test_data =[]
    for comb in combination:
        machine = comb[0]
        fea = comb[1] #encoding
        for datype in ['val','test']:
                fea_file = data_path + '/%s/%s/%s/%s_roc.csv' %(machine, fea, str(kfold), datype)
                fea_data = pd.read_csv(fea_file)
                if datype =='val':
                    valid_data.append(fea_data['prob'].values.tolist())
                    valid_data.append(fea_data['label'].values.tolist())
                elif datype =='test':
                    test_data.append(fea_data['prob'].values.tolist())
                    test_data.append(fea_data['label'].values.tolist())
                    
    valid_data = np.array(valid_data).T
    test_data = np.array(test_data).T    
    #print(f'valid_data\n {valid_data}')
    print(f'valid_data\n {valid_data.shape}')
    print(f'test_data\n {test_data.shape}')
    # Redundant labels [label,prob,label, prob,....] are removed 
    valid_data = np.delete(valid_data, [i for i in range(1, 2*len(combination)-1, 2)], 1)
    test_data  = np.delete(test_data, [i for i in  range(1, 2*len(combination)-1, 2)], 1) 
    #print('\n')
    #print(f'test_data\n {test_data}')
    #print(f'test_data[:,-1]\n {test_data[:,-1]}')
    #print(f'test_data[:,:-1]\n {test_data[:,:-1]}') 
    #print(f'test_data[:,:-1]\n {test_data[:,:-1].shape}')      
    #print('non redundant valid_data.shape {}'.format(valid_data.shape) )
 
    # training and testing
    df, df2 = combine_model(valid_data, test_data, data_path, out_dir, kfold, columns)

    return df, df2
    

# score combine method based on logistic regression
if __name__ == '__main__':   

    parser = argparse.ArgumentParser()
    parser.add_argument('--machine_method', type=str, help='term')
    parser.add_argument('--encode_method', type=str, help='term')
    parser.add_argument('--top_list', type=str, help='term')
    parser.add_argument('--species', type=str, help='term')
    parser.add_argument('--results_path', type=str, required=True)
    parser.add_argument('--lens', type=str, required=True)
    parser.add_argument('--deep_method', type=str, required=True)
    parser.add_argument('--deep_encode', type=str, required=True)
    args = parser.parse_args()
    
    machine_method = args.machine_method.strip().split(',')
    encode_method  =  args.encode_method.strip().split(',')
    species = args.species
    top_list  =  args.top_list.strip().split(',')
    top_list = [int(i) for i in top_list]
    results_path = args.results_path.rstrip('/')
    lens = args.lens.strip().split(',')
    deep_method = args.deep_method.strip().split(',')
    deep_encode = args.deep_encode.strip().split(',')
    kfold=5
    

    for l in lens:
        data_path=results_path+"/"+species+"/"+l
        combination_rank = ranking(data_path, machine_method, encode_method,deep_method,deep_encode) 
            
        df_all = pd.DataFrame(columns=item_column)     
        for top_number in top_list:
            print(f'top_number: {top_number}')

            comb = []
            for i in range(top_number):
                comb.append(combination_rank[i]) 
            combination = comb

            columns = []      
            columns = [combination[i][0]+'-'+combination[i][1] for i in range(0,top_number)] + ['label']
            top_combination = [ combination[i] for i in range(0,top_number)]
            out_dir ='combine/top%s'%top_number
                
            print(top_combination )
            df_valid = pd.DataFrame(columns=item_column) 
            df_test = pd.DataFrame(columns=item_column)     
            for k in range(1, kfold+1):
                df, df2 = train_test(k, data_path, out_dir, top_combination, columns)      
                df_valid.loc[str(k) + "_valid"] = df.loc['valid']
                df_test.loc[str(k) + "_test"] = df.loc['test']
            
            df_cat = pd.DataFrame(columns=item_column)
            df_cat.loc["mean_train"] = df_valid.mean()
            df_cat.loc["sd_train"] = df_valid.std()
            df_cat.loc["mean_test"] = df_test.mean()
            df_cat.loc["sd_test"] = df_test.std()
                
            df_cat.to_csv('%s/%s/average_measure.csv' %(data_path, out_dir)) 
            print(df_cat )
                        
            df_all.loc[str(top_number) + '_valid'] = df_valid.mean()
            df_all.loc[str(top_number) + '_test'] = df_test.mean()
                
        df_all.to_csv('%s/combine/top_measure.csv' %(data_path))         
        
        
    



    
    
    
       
