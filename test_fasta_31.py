import pandas as pd
import os
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--species", type=str, required=True, help="")  
    parser.add_argument("--seqwin", type=int, required=True)
    parser.add_argument("--test_file", type=str, required=True, help="")  
    
    args = parser.parse_args()
    test_f = args.test_file
    data_path = args.path
    species = args.species
    seqwin = args.seqwin

    test_txt = f"{data_path}/{test_f}"
    
    out_dir = f"{data_path}/{species}_{seqwin}"
    os.makedirs(out_dir, exist_ok=True)

    test_fasta = f"{out_dir}/independent_test.fa"
    test_csv   = f"{out_dir}/independent_test.csv"

    #df_test = pd.read_csv(test_txt, header=None, index_col = None, names=["Chromosome", "gene_name", "label", "site_id", "seq" ])    
    df_test = pd.read_csv(test_txt, header=None, index_col = None, names=["seq", "label"])    
    df_test['label'] = df_test['label'].astype(int)

    with open(test_fasta, 'w') as fout:
        for i in range(df_test.shape[0]):
            label = df_test.loc[i, "label"]
            fout.write(f'>pep_{i}|{label}|label\n')
            fout.write(df_test.loc[i, "seq"])
            fout.write('\n')

    df_test[["seq","label"]].to_csv(test_csv, index=None)
        
        
