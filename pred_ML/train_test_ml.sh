#!/bin/bash

machine_method=LGBM,XGB,RF,SVM,NB,KN,LR #LGBM,XGB,RF,SVM,NB,KN,LR
encode_method=DNC,TNC,CKSNAP,RCKmer,PseEIIP,binary,ENAC,ANF,NCP,EIIP,PSTNPss,DAC,DCC,DACC,PseKNC,PseDNC,SCPseDNC
kfold=5

# convnetional encodings
lag=5
weight=0.1
lamada=2
kmer=1
echo "=== START batch processing ==="

for sp in hg38 ; do       #mm10 sacCer3
    echo "----------------------------------"
    echo " Species: ${sp}"
    echo "----------------------------------"

    for seqwin in 201; do
        echo "=== Processing ${sp} with seqwin=${seqwin} ==="

        train_path=../dataset/train_PseU_${sp}_${seqwin}/cross_val
        test_csv=../dataset/train_PseU_${sp}_${seqwin}/independent_test.csv
        result_path=../results/${sp}/${seqwin}

        python3 ml_train_test_63.py \
            --intrain ${train_path} \
            --intest ${test_csv} \
            --outpath ${result_path} \
            --machine ${machine_method} \
            --encode ${encode_method} \
            --type RNA \
            --seqwin ${seqwin} \
            --fold ${kfold} \
            --lag ${lag} \
            --weight ${weight} \
            --lamadaValue ${lamada} \
            --kmer ${kmer}

    done
done
