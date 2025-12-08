import pandas as pd
import random
from sklearn.model_selection import train_test_split

data_path = "./dataset"
types=["hg38","mm10","sacCer3"]

for type in types:
   test_txt = data_path + "/test_PseU_"+type+"_201_c0.7.txt" #ENNAVIA-D indepedent dataset
   test_fasta = data_path + "/train_PseU_"+type+"_201/independent_test.fa"
   test_csv = data_path   + "/train_PseU_"+type+"_201/independent_test.csv"

   test = pd.read_csv(test_txt , header=None)
   test = test.rename(columns={0:'seq',1:'label'})
   print(test)


   with open(test_fasta, 'w') as fout:
      for i in range(test.shape[0]):
         if test.iloc[i,1] == 1:
            fout.write('>pep_%s|1|label\n'%i)
            fout.write(test.iloc[i,0])
            fout.write('\n')
         else:
            fout.write('>pep_%s|0|label\n'%i)
            fout.write(test.iloc[i,0])
            fout.write('\n')

   test.to_csv(test_csv, index=None)
    
