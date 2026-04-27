#!/bin/bash
set -e

for name in PseU_hg38_41_c0.7
do

name1=train_${name}
name2=test_${name}

infile1=dataset/${name1}.txt
outfile1=${name1}.fasta

infile2=dataset/${name2}.txt
outfile2=${name2}.fasta

python csv2fasta.py --infile ${infile1} --outfile ${outfile1}
python csv2fasta.py --infile ${infile2} --outfile ${outfile2}

done
