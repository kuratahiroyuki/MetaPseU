#!/bin/bash

cd ..
cd ..
main_path=`pwd`
cd pred_DL
cd network

echo ${main_path}

# === Common parameters ===
species_list="hg38 mm10 sacCer3" 
seqwin_list="41 81 121 161 201"

species_list="sacCer3" 
seqwin_list="41"

deep_method=CNN,TX,bLSTM
#deep_encode=W2V_4_64_30_30_1,BE
#deep_encode=W2V_4_64_128_40_1,BE

deep_method=CNN
deep_encode=W2V_1_64_128_40_1 #,W2V_2_64_128_40_1  #,W2V_3_64_128_40_1,W2V_4_64_128_40_1

kfold=5
kmer=1 #temporary value
size=64
#epochs=30
#window=30
sg=1

epochs=128
window=40

# === Species list ===
for seqwin in $seqwin_list
do
echo "============================"
echo "Data set Length: ${seqwin}"
echo "============================"

    for species in $species_list
    do
    echo "============================"
    echo "Processing species: ${species}"
    echo "============================"

    train_path=${main_path}/dataset/${species}_${seqwin}/cross_val
    test_file=${main_path}/dataset/${species}_${seqwin}/independent_test.csv
    result_path=${main_path}/result/${species}/${seqwin}

    #w2v_model=${main_path}/pred_DL/w2vmodel/${species}_201/w2v_kmer_${size}_${epochs}_${window}_${sg}.pt
    w2v_model=${main_path}/pred_DL/w2vmodel/rna_w2v_kmer_${size}_${epochs}_${window}_${sg}.pt
    
        for deep_model in ${deep_method//,/ }
        do
        echo "--- Model: ${deep_model} ---"
        echo "Running ${species} | ${deep_model}"

<<cout
cout
       
        python3 dl_train_test_32.py \
            --intrain ${train_path} \
            --intest ${test_file} \
            --outpath ${result_path} \
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
    done
done

cd ..
cd ..

