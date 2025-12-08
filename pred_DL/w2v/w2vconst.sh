#!/bin/sh

w2v_path=../w2vmodel
infile=rna_dataset.json

size=64 
epochs=128 
sg=1 
window=40
              

for kmer in 1 2 3 4
do
echo ${kmer}
python w2v_const.py --infile ${infile} --w2v ${w2v_path} --kmer ${kmer} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window}
done


