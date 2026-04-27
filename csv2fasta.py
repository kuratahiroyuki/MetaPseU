import pandas as pd
import argparse

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument( '--infile', type=str, help='file')
    parser.add_argument( '--outfile', type=str, help='file')

    infile = parser.parse_args().infile
    outfile = parser.parse_args().outfile

    #chr1,intergenic,1,chr1:629454:+,GAUAAAAGAGUUACUUUGAUA

    df = pd.read_csv(infile, delimiter=",", names=["chr","location","label","id","seq",], index_col = None)
    print(df)
    #df = df[df['seq'].str.upper().str.fullmatch('[AUTGC]+')]
    seqs = df["seq"].values.tolist()
    labels = df["label"].values.tolist()

    with open(outfile,"w") as f:
        for i, (seq, label) in enumerate(zip(seqs,labels)):
            f.write(f">seq{i+1}|{label}\n")
            f.write(f"{seq}\n")


"""
>seq1|0
CCUUUUUCCUUCUCCUUCCUU
"""
