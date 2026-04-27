# MetaPseU

MetaPseU is a bioinformatics pipeline for predicting RNA pseudouridine (PseU) sites. The workflow combines conventional machine-learning models, deep-learning models, and a meta-classifier.

## Repository structure

```text
MetaPseU/
├── dataset/                 # Example training/test datasets
├── pred_ML/                 # Machine-learning models and RNA encodings
├── pred_DL/                 # Deep-learning models and Word2Vec resources
├── results/                 # Output directory
├── data_const.sh            # Dataset construction script
├── main_1.sh                # Main training/testing workflow
├── process.sh               # Per-species/per-window processing pipeline
└── csv2fasta.py             # CSV-to-FASTA conversion utility
```

## Installation

Create and activate a Python environment, then install the required packages.

```bash
conda create -n metapseu python=3.10 -y
conda activate metapseu
pip install -r requirements.txt
```

## Quick start

### 1. Prepare datasets

```bash
bash data_const.sh
```

### 2. Build Word2Vec models

```bash
cd pred_DL/w2v
bash w2vconst.sh
cd ../..
```

### 3. Train and test the models

```bash
bash main_1.sh
```

This workflow trains machine-learning models, deep-learning models, and the meta-classifier.

## Model settings

### Deep-learning models

Implemented deep-learning models:

- CNN
- Transformer encoder (TX)
- Bidirectional LSTM (bLSTM/BiLSTM)

Supported encodings:

- Word2Vec (W2V)
- Binary/one-hot encoding (BE)

Deep-learning training is performed in `pred_DL/network` by `dl_train_test_32.py`.

### Machine-learning models

Implemented machine-learning models:

- LightGBM (LGBM)
- XGBoost (XGB)
- Random Forest (RF)
- Support Vector Machine (SVM)
- Naive Bayes (NB)
- k-Nearest Neighbors (KN)
- Logistic Regression (LR)

Supported RNA sequence encodings include DNC, TNC, CKSNAP, RCKmer, PseEIIP, binary, ENAC, ANF, NCP, EIIP, DAC, DCC, DACC, PseKNC, PseDNC, and PCPseDNC.

Machine-learning training is performed in `pred_ML` by `ml_train_test_81.py`.

## Convert CSV to FASTA

Input CSV format:

```text
chromosome,location,label,id,sequence
```

Example command:

```bash
python csv2fasta.py --infile input.csv --outfile output.fasta
```

## Outputs

Results are written to the `results/` directory. Final summary files are exported as Excel files named like:

```text
results/result_<species>_<sequence_window>_<cutoff>.xlsx
```

## Notes for GitHub release

- Python bytecode files and cache directories are excluded by `.gitignore`.
- Large model files (`*.pt`) are currently included in this cleaned archive because they were present in the uploaded tool. For a public GitHub repository, consider moving large pretrained models to GitHub Releases, Zenodo, Figshare, or another model-hosting location.
- The current command-line workflow uses shell scripts. Run commands from the repository root unless otherwise noted.
