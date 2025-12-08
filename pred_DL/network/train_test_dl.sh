#!/bin/bash

cd ..
cd ..
main_path=`pwd`
echo ${main_path}
cd pred_DL
cd network

# === Common parameters ===
kfold=5
size=64
epochs=30
sg=1
window=30
encode_method=BE #W2V BE

# === Base directories ===
base_dataset=${main_path}/dataset
base_result=${main_path}/results
base_w2v=${main_path}/pred_DL/w2vmodel

# === Species list ===
for seqwin in 201 
do
echo "============================"
echo "Data set Length: ${seqwin}"
echo "============================"

    for species in hg38      #mm10 sacCer3
    do
    echo "============================"
    echo "Processing species: ${species}"
    echo "============================"

    train_path=${base_dataset}/train_PseU_${species}_${seqwin}/cross_val
    test_file=${base_dataset}/train_PseU_${species}_${seqwin}/independent_test.csv
    result_path=${base_result}/${species}/${seqwin}
    w2v_path=${base_w2v}/${species}_201

    for deep_method in CNN TX bLSTM
    do
        echo "--- Model: ${deep_method} ---"

        for kmer in 4
        do
        w2v_model=${w2v_path}/w2v_${kmer}_${size}_${epochs}_${window}_${sg}.pt


        echo "Running ${species} | ${deep_method} | kmer=${kmer}"
        python3 train_test_8.py \
            --intrain ${train_path} \
            --intest ${test_file} \
            --outpath ${result_path} \
            --losstype "balanced" \
            --deeplearn ${deep_method} \
            --encode ${encode_method} \
            --fold ${kfold} \
            --w2vmodel ${w2v_model} \
            --seqwin ${seqwin} \
            --kmer ${kmer} \
            --size ${size} \
            --epochs ${epochs} \
            --sg ${sg} \
            --window ${window}
        done
    done
    done
done

