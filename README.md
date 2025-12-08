# Meta-PseU Execution Guide (README)

## Overall Workflow

1.Preparation of cross-validation (CV) datasets and independent test datasets
2.Construction of Word2Vec (W2V) models for deep learning
3.Training and testing of deep learning models (TX, CNN, bLSTM)
4.Training and testing of machine learning models
5.Analysis of all results and implementation of stacking


## Preparation of CV and Independent Test Datasets  

The training dataset is divided into k-fold cross-validation subsets.
>python train_division_1.py

## Independent test data are generated in FASTA format.

>python test_fasta.py

## Construction of Word2Vec (W2V) models with K-mer

Pred_DL/w2v
>sh w2v_const.sh

## Training and Testing of Deep Learning Models

deep_learning: TX, CNN, bLSTM(BiLSTM)
encoding: W2V, binary (OH)
pred_DL/network
>sh train_test_dl.sh (calling train_test_8.py fro TX, CNN, bLSTM or ml_av2.py for RF)


## Training and Testing of Machine Learning Models

machine_learning: LGBM,XGB,RF,SVM,NB,KN,LR #LGBM,XGB,RF,SVM,NB,KN,LR
encoding: DNC,TNC,CKSNAP,RCKmer,PseEIIP,binary,ENAC,ANF,NCP,EIIP,PSTNPss,DAC,DCC,DACC,PseKNC,PseDNC,SCPseDNC
pred_ML
>sh train_test_ml.sh  
(calling  ml_train_test_63.py)  

## Analysis and Stacking
>sh stack_1.sh 
(calling  analysis_62.py  ml_fusion_39.py  csv_xlsx_32.py )

