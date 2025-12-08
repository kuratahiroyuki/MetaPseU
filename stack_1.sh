#!/bin/bash

machine_method=LGBM,XGB,RF,SVM,NB,KN,LR
encode_method=DNC,TNC,CKSNAP,RCKmer,binary,ENAC,ANF,NCP,EIIP,PseEIIP,PSTNPss,PseKNC,PseDNC,SCPseDNC,DAC,DCC,DACC
top_list=1,2,4,8,16,32,48,64,80,96,112,125
lens=201
data_path=./results
deep_method=CNN,TX,bLSTM
deep_encode=W2V_4_64_30_30_1,BE

for species in hg38 #mm10 sacCer3
do
python3 analysis_62.py --machine_method ${machine_method} --encode_method ${encode_method} --species ${species} --data_path ${data_path} --lens ${lens} --deep_method "${deep_method}" --deep_encode_method "${deep_encode}"

python3 ml_fusion_39.py --machine_method ${machine_method} --encode_method ${encode_method} --species ${species} --top_list ${top_list}  --results_path "${data_path}"  --lens "${lens}" --deep_method "${deep_method}" --deep_encode "${deep_encode}"
    
python3 csv_xlsx_32.py --machine_method ${machine_method} --encode_method ${encode_method} --species ${species} --results_path "${data_path}" --lens "${lens}" --deep_method "${deep_method}" --deep_encode "${deep_encode}"

done





