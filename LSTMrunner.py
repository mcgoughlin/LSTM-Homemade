# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 12:48:27 2019

@author: Billy
"""
import numpy as np
import LSTM_Classes
import LSTM_functions
import pandas as pd

df = pd.read_csv('monthly_csv.csv')[['score','index']].dropna().reset_index()

trans_data = df['score'].values.tolist()

#Normalizes data and retrieves the scale factor for the normalization such that any data magnitudes can be input
normalized_data,scale_factor = LSTM_functions.normalize(trans_data)

#split ratio determines the percentage of material that is used to train
split_ratio = 0.95
split = int(np.round((1-split_ratio)*len(normalized_data)))
train_data = LSTM_functions.reverse_qsort(normalized_data[split:len(normalized_data)])
test_data = LSTM_functions.reverse_qsort(normalized_data[0:split])

cell = LSTM_Classes.LSTM_cell()

timesteps_pb = 15
start_point = 0


data, labels = LSTM_functions.split_sequence(train_data, timesteps_pb)
test_data, test_labels = LSTM_functions.split_sequence(test_data, timesteps_pb)

test_data = test_data.tolist()
test_labels = test_labels.tolist()
data = data.tolist()
labels = labels.tolist()
#Calculates epochs
epochs = len(data)

#Compute the first batch
if epochs > 0:
    inpt = data[0]
    label_ = labels[0]
    cell.LSTM(inpt,label =label_, Learning = True, new = True)

loop = 1
#Compute the rest of the batches
while loop< epochs: 
    print(loop)
    inpt = data[loop]
    label_ = labels[loop]
    cell.LSTM(inpt, label=label_, Learning=True)
    print(cell.forget_weights, cell.input_weights, cell.candidate_weights, cell.output_weights)
    loop+=1
