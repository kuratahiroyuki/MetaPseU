#!/usr/bin/env python
#_*_coding:utf-8_*_

import re
import sys

### T=>U
myDiIndex = {
    'AA': 0, 'AC': 1, 'AG': 2, 'AU': 3,
    'CA': 4, 'CC': 5, 'CG': 6, 'CU': 7,
    'GA': 8, 'GC': 9, 'GG': 10, 'GU': 11,
    'UA': 12, 'UC': 13, 'UG': 14, 'UU': 15
}
myTriIndex = {
    'AAA': 0, 'AAC': 1, 'AAG': 2, 'AAU': 3,
    'ACA': 4, 'ACC': 5, 'ACG': 6, 'ACU': 7,
    'AGA': 8, 'AGC': 9, 'AGG': 10, 'AGU': 11,
    'AUA': 12, 'AUC': 13, 'AUG': 14, 'AUU': 15,
    'CAA': 16, 'CAC': 17, 'CAG': 18, 'CAU': 19,
    'CCA': 20, 'CCC': 21, 'CCG': 22, 'CCU': 23,
    'CGA': 24, 'CGC': 25, 'CGG': 26, 'CGU': 27,
    'CUA': 28, 'CUC': 29, 'CUG': 30, 'CUU': 31,
    'GAA': 32, 'GAC': 33, 'GAG': 34, 'GAU': 35,
    'GCA': 36, 'GCC': 37, 'GCG': 38, 'GCU': 39,
    'GGA': 40, 'GGC': 41, 'GGG': 42, 'GGU': 43,
    'GUA': 44, 'GUC': 45, 'GUG': 46, 'GUU': 47,
    'UAA': 48, 'UAC': 49, 'UAG': 50, 'UAU': 51,
    'UCA': 52, 'UCC': 53, 'UCG': 54, 'UCU': 55,
    'UGA': 56, 'UGC': 57, 'UGG': 58, 'UGU': 59,
    'UUA': 60, 'UUC': 61, 'UUG': 62, 'UUU': 63
}

def generatePropertyPairs(myPropertyName):
    pairs = []
    for i in range(len(myPropertyName)):
        for j in range(i+1, len(myPropertyName)):
            pairs.append([myPropertyName[i], myPropertyName[j]])
            pairs.append([myPropertyName[j], myPropertyName[i]])

    return pairs

def make_ac_vector(fastas, myPropertyName, myPropertyValue, lag, kmer):
    encodings = []
    myIndex = myDiIndex if kmer == 2 else myTriIndex
    header = ['#', 'label']
    for p in myPropertyName:
        for l in range(1, lag + 1):
            header.append('%s.lag%d' %(p, l))
    #encodings.append(header) ###
    
    for i in fastas:
        #name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        #code = [name, label]
        sequence, label = i[0], i[1] ###
        code = [] ###
        for p in myPropertyName:
            meanValue = 0
            #for j in range(len(sequence) - kmer):
            for j in range(len(sequence) - kmer + 1):
                meanValue = meanValue + float(myPropertyValue[p][myIndex[sequence[j: j+kmer]]])
            #meanValue = meanValue / (len(sequence) - kmer)
            meanValue = meanValue / (len(sequence) - kmer + 1)
            
            for l in range(1, lag + 1):
                acValue = 0
                for j in range(len(sequence) - kmer - l + 1):
                    #acValue = acValue + (float(myPropertyValue[p][myIndex[sequence[j: j+kmer]]]) - meanValue) * (float(myPropertyValue[p][myIndex[sequence[j+l:j+l+kmer]]]))
                    acValue = acValue + (float(myPropertyValue[p][myIndex[sequence[j: j + kmer]]]) - meanValue) * (
                    float(myPropertyValue[p][myIndex[sequence[j + l:j + l + kmer]]]) - meanValue)
                acValue = acValue / (len(sequence) - kmer - l + 1)
                #print(acValue)
                code.append(acValue)
        encodings.append(code)
    return encodings


def make_cc_vector(fastas, myPropertyName, myPropertyValue, lag, kmer):
    encodings = []
    myIndex = myDiIndex if kmer == 2 else myTriIndex
    if len(myPropertyName) < 2:
        print('Error: two or more property are needed for cross covariance (i.e. DCC and TCC) descriptors')
        sys.exit(1)
    propertyPairs = generatePropertyPairs(myPropertyName)
    header = ['#', 'label'] + [n[0] + '-' + n[1] + '-lag.' + str(l) for n in propertyPairs for l in range(1, lag + 1)]
    #encodings.append(header) ###

    for i in fastas:
        #name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        #code = [name, label]
        sequence, label = i[0], i[1] ###
        code = [] ###
        for pair in propertyPairs:
            meanP1 = 0
            meanP2 = 0
            #for j in range(len(sequence) - kmer):
            for j in range(len(sequence) - kmer + 1):
                meanP1 = meanP1 + float(myPropertyValue[pair[0]][myIndex[sequence[j: j+kmer]]])
                meanP2 = meanP2 + float(myPropertyValue[pair[1]][myIndex[sequence[j: j+kmer]]])
            #meanP1 = meanP1 / (len(sequence) - kmer)
            #meanP2 = meanP2 / (len(sequence) - kmer)
            meanP1 = meanP1 / (len(sequence) - kmer + 1)
            meanP2 = meanP2 / (len(sequence) - kmer + 1)

            for l in range(1, lag + 1):
                ccValue = 0
                for j in range(len(sequence) - kmer - l + 1):
                    ccValue = ccValue + (float(myPropertyValue[pair[0]][myIndex[sequence[j: j + kmer]]]) - meanP1) * (
                        float(myPropertyValue[pair[1]][myIndex[sequence[j + l:j + l + kmer]]]) - meanP2)
                ccValue = ccValue / (len(sequence) - kmer - l + 1)
                code.append(ccValue)
        encodings.append(code)
    return encodings


def make_acc_vector(fastas, myPropertyName, myPropertyValue, lag, kmer):
    encodings = []
    myIndex = myDiIndex if kmer == 2 else myTriIndex
    if len(myPropertyName) < 2:
        print('Error: two or more property are needed for cross covariance (i.e. DCC and TCC) descriptors')
        sys.exit(1)

    header = ['#', 'label']
    for p in myPropertyName:
        for l in range(1, lag + 1):
            header.append('%s.lag%d' %(p, l))
    propertyPairs = generatePropertyPairs(myPropertyName)
    header = header + [n[0] + '-' + n[1] + '-lag.' + str(l) for n in propertyPairs for l in range(1, lag + 1)]
    #encodings.append(header) ###

    for i in fastas:
        #name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        #code = [name, label]
        sequence, label = i[0], i[1] ###
        code = [] ###        
        
        ## Auto covariance
        for p in myPropertyName:
            meanValue = 0
            # for j in range(len(sequence) - kmer):
            for j in range(len(sequence) - kmer + 1):
                meanValue = meanValue + float(myPropertyValue[p][myIndex[sequence[j: j + kmer]]])
            # meanValue = meanValue / (len(sequence) - kmer)
            meanValue = meanValue / (len(sequence) - kmer + 1)

            for l in range(1, lag + 1):
                acValue = 0
                for j in range(len(sequence) - kmer - l + 1):
                    # acValue = acValue + (float(myPropertyValue[p][myIndex[sequence[j: j+kmer]]]) - meanValue) * (float(myPropertyValue[p][myIndex[sequence[j+l:j+l+kmer]]]))
                    acValue = acValue + (float(myPropertyValue[p][myIndex[sequence[j: j + kmer]]]) - meanValue) * (
                        float(myPropertyValue[p][myIndex[sequence[j + l:j + l + kmer]]]) - meanValue)
                acValue = acValue / (len(sequence) - kmer - l + 1)
                # print(acValue)
                code.append(acValue)

        ## Cross covariance
        for pair in propertyPairs:
            meanP1 = 0
            meanP2 = 0
            #for j in range(len(sequence) - kmer):
            for j in range(len(sequence) - kmer + 1):
                meanP1 = meanP1 + float(myPropertyValue[pair[0]][myIndex[sequence[j: j+kmer]]])
                meanP2 = meanP2 + float(myPropertyValue[pair[1]][myIndex[sequence[j: j+kmer]]])
            #meanP1 = meanP1 / (len(sequence) - kmer)
            #meanP2 = meanP2 / (len(sequence) - kmer)
            meanP1 = meanP1 / (len(sequence) - kmer + 1)
            meanP2 = meanP2 / (len(sequence) - kmer + 1)

            for l in range(1, lag + 1):
                ccValue = 0
                for j in range(len(sequence) - kmer - l + 1):
                    ccValue = ccValue + (float(myPropertyValue[pair[0]][myIndex[sequence[j: j + kmer]]]) - meanP1) * (
                        float(myPropertyValue[pair[1]][myIndex[sequence[j + l:j + l + kmer]]]) - meanP2)
                ccValue = ccValue / (len(sequence) - kmer - l + 1)
                code.append(ccValue)
        encodings.append(code)
    return encodings
