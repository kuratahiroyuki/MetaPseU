#!/bin/bash
set -e


cutoff=0.7
full_len=201
DATA_PATH="./dataset"
KFOLD=5
SPECIES="hg38 mm10 sacCer3"
SEQWINS="41 81 121 161 201"

SEQWINS=41

for sp in $SPECIES
do
    train_src=train_PseU_${sp}_${full_len}_c${cutoff}.txt
    test_src=test_PseU_${sp}_${full_len}_c${cutoff}.txt
    
    echo "==========================="
    echo "Processing species = ${sp}"
    echo "==========================="
    
    for seqwin in $SEQWINS
    do
      echo "==========================="
      echo "Processing seqwin = ${seqwin}"
      echo "==========================="

      train_file=train_PseU_${sp}_${seqwin}_c${cutoff}.txt
      test_file=test_PseU_${sp}_${seqwin}_c${cutoff}.txt
     
      python3 trim_center_31.py --path ${DATA_PATH} --seqwin ${seqwin} --train_src ${train_src} --test_src ${test_src} --train_file ${train_file} --test_file ${test_file}

      python3 train_division_31.py --path ${DATA_PATH} --kfold ${KFOLD} --species ${sp} --seqwin ${seqwin} --train_file ${train_file}

      python3 test_fasta_31.py --path ${DATA_PATH} --species ${sp} --seqwin ${seqwin} --test_file ${test_file}

    done
done
