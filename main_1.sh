#!/bin/bash

for seqwin in 41 #81 121 161 201
do
    echo "seqwin = ${seqwin}"
    bash process.sh hg38 ${seqwin} 0.7
done

exit 1

for seqwin in 41 81 121 161 201
do
    bash process.sh mm10 ${seqwin} 0.7
done

for seqwin in 41 81 121 161 201
do
    bash process.sh sacCer3 ${seqwin} 0.7
done

