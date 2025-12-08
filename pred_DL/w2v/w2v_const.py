#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import pandas as pd
import json
from gensim.models import word2vec


def sep_word(data, num):
    res = []

    for i, seq in enumerate(data):
        res.append([seq[j: j+ num] for j in range(len(seq) - num + 1)])
        
    return res
        

if __name__=='__main__':
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--w2v', help='path')
    parser.add_argument('-i1', '--infile', help='file')
    parser.add_argument('-k', '--kmer', type=int, help='value')
    parser.add_argument('-s', '--size', type=int, help='value')
    parser.add_argument('-e', '--epochs', type=int, help='value')
    parser.add_argument('-sg', '--sg', type=int, help='value')
    parser.add_argument('-window', '--window', type=int, help='value')

    infile = parser.parse_args().infile
    w2v_path = parser.parse_args().w2v
    kmer = parser.parse_args().kmer #4
    size = parser.parse_args().size #100
    epochs = parser.parse_args().epochs #4
    sg = parser.parse_args().sg # default 1
    window = parser.parse_args().window #100

    #sequence preparation
    json_open = open(infile,'r')
    data = json.load(json_open)
    print(len(data))
    
    sequences = []
    for value in data:
        #print(value['sequence'])
        sequences.append(value['sequence'])
    #print(sequences)

    # word2vec training
    words = sep_word(sequences, kmer)
    #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(words, vector_size = size, min_count = 1, window = window - kmer + 1, epochs = epochs, sg = sg) 
    model.save(w2v_path + "/rna_w2v_" + str(kmer) + '_' + str(size) + '_' + str(epochs) + '_' + str(window) + '_' + str(sg)  + ".pt")

    print('elapsed time:', time.time()-start)








