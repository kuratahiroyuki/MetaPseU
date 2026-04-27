#!/bin/bash

machine_method=LGBM,XGB,RF,NB,KN,LR #LGBM,XGB,RF,SVM,NB,KN,LR
encode_method=DNC,TNC,CKSNAP,RCKmer,PseEIIP,binary,ENAC,ANF,NCP,EIIP,PSTNPss,DAC,DCC,DACC,PseKNC,PseDNC,SCPseDNC
kfold=5

# convnetional encodings
lag=5
weight=0.1
lamada=2
kmer=1
echo "=== START batch processing ==="

for species in sacCer3; do       #hg38 mm10 sacCer3
    echo "----------------------------------"
    echo " Species: ${species}"
    echo "----------------------------------"

    for seqwin in 41; do  #41 81 121 161 201
        echo "=== Processing ${species} with seqwin=${seqwin} ==="

        train_path=../dataset/${species}_${seqwin}/cross_val
        test_csv=../dataset/${species}_${seqwin}/independent_test.csv
        result_path=../result/${species}/${seqwin}

        python3 ml_train_test_81.py \
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
