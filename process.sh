#!/bin/bash

species=$1
seqwin=$2
cutoff=$3

main_path=`pwd`
echo ${main_path}

result_path=${main_path}/results
mkdir -p ${result_path}

# === Common parameters ===

machine_method=LGBM,XGB,RF,SVM,NB,KN,LR # 
machine_encode=DNC,TNC,CKSNAP,RCKmer,binary,ENAC,ANF,NCP,EIIP,PseEIIP,PseKNC,PseDNC,SCPseDNC,DAC,DCC,DACC

deep_method=CNN,TX,bLSTM

deep_encode=W2V_4_64_30_30_1,BE
#deep_encode=W2V_1_64_128_40_1,W2V_2_64_128_40_1,W2V_3_64_128_40_1,W2V_4_64_128_40_1,BE

kfold=5
top_list=1,2,4,8,16,32,48,64,80,96,108,118

# === ML ===

cd pred_ML
# convnetional encodings
lag=5
weight=0.1
lamada=2
kmer=1

echo "=== START batch processing ==="
echo "=== ML Processing ${species} with seqwin=${seqwin} ==="

train_path=${main_path}/dataset/${species}_${seqwin}/cross_val
test_csv=${main_path}/dataset/${species}_${seqwin}/independent_test.csv

score_path=${main_path}/results/${species}/${seqwin}

<<cout
cout

python3 ml_train_test_81.py \
    --intrain ${train_path} \
    --intest ${test_csv} \
    --outpath ${score_path} \
    --machine ${machine_method} \
    --encode ${machine_encode} \
    --type RNA \
    --seqwin ${seqwin} \
    --fold ${kfold} \
    --lag ${lag} \
    --weight ${weight} \
    --lamadaValue ${lamada} \
    --kmer ${kmer}


cd ..


# === DL ===

cd pred_DL/network

size=64
sg=1

for deep_model in ${deep_method//,/ }
do    
echo "=== DL Processing ${deep_model} | ${species} with seqwin= ${seqwin} ==="

epochs=30;  window=30; w2v_model=${main_path}/pred_DL/w2vmodel/${species}_201/w2v_4_${size}_${epochs}_${window}_${sg}.pt     
#epochs=128; window=40; w2v_model=${main_path}/pred_DL/w2vmodel/rna_w2v_kmer_${size}_${epochs}_${window}_${sg}.pt

<<cout        
cout
python3 dl_train_test_32.py \
    --intrain ${train_path} \
    --intest ${test_csv} \
    --outpath ${score_path} \
    --losstype "balanced" \
    --deeplearn ${deep_model} \
    --deep_encode ${deep_encode} \
    --fold ${kfold} \
    --w2vmodel ${w2v_model} \
    --seqwin ${seqwin} \
    --kmer ${kmer} \
    --size ${size} \
    --epochs ${epochs} \
    --window ${window} \
    --sg ${sg}

done

cd ..
cd ..


# === Meta-classifier ===

outfile=${main_path}/results/result_${species}_${seqwin}_${cutoff}.xlsx
echo "=== Meta Processing ${species} with seqwin=${seqwin} ==="

<<cout
cout

echo evaluation
python analysis_623.py --machine_method ${machine_method} --machine_encode ${machine_encode}  --deep_method ${deep_method} --deep_encode ${deep_encode} --species ${species} --seqwin ${seqwin} --result_path ${result_path} 

echo fusion
python ml_fusion_54.py --machine_method ${machine_method} --machine_encode ${machine_encode}  --deep_method ${deep_method} --deep_encode ${deep_encode} --species ${species} --seqwin ${seqwin} --result_path ${result_path}  --top_list ${top_list} 

echo selection
python ml_select_14.py --species ${species} --seqwin ${seqwin} --result_path ${result_path}  --top_list ${top_list}

echo output
python csv_xlsx_342.py  --machine_method ${machine_method} --machine_encode ${machine_encode} --deep_method ${deep_method} --deep_encode ${deep_encode} --species ${species} --seqwin ${seqwin} --result_path ${result_path} --outfile ${outfile}

  

