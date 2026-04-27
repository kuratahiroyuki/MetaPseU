import pandas as pd
import os
import argparse
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold


def output_csv_pandas(filename, data):
    data.to_csv(filename, index=None, header=True)


if __name__=="__main__":
    # ===== 引数 =====
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--train_file", type=str, required=True, help="")  
    parser.add_argument("--species", type=str, required=True, help="")  
    parser.add_argument("--seqwin", type=int, required=True)
    args = parser.parse_args()

    data_path = args.path
    kfold = args.kfold
    train_f = args.train_file
    species = args.species 
    seqwin = args.seqwin 
    
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)

    train_file = f"{data_path}/{train_f}"
    #train_df = pd.read_csv(train_file, header=None, index_col = None, names=["Chromosome", "gene_name", "label", "site_id", "seq" ])      
    train_df = pd.read_csv(train_file, header=None, index_col = None, names=["seq", "label"])   
    print(train_df)

    X = train_df
    y = train_df["label"]
         
         
    for idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold = idx +1
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"Fold {fold}")
        print("Train size:", len(train_idx))
        print("Val size:", len(val_idx))
   
        out_dir = f"{data_path}/{species}_{seqwin}/cross_val/{fold}"
        os.makedirs(out_dir, exist_ok=True)

        output_csv_pandas(
            f"{out_dir}/cv_train_{fold}.csv",
            train_df.loc[train_idx, ["seq","label"]].reset_index(drop=True)
        )
        output_csv_pandas(
            f"{out_dir}/cv_val_{fold}.csv",
            train_df.loc[val_idx, ["seq","label"]].reset_index(drop=True)
        )            
            

