# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 17:33:21 2019

@author: Billy
"""
import math
import mpmath
import numpy as np

def ArraySizer(array):

    try:
        column_count = len(array[0])
        row_count = len(array)
    except(TypeError):
        row_count = 1
        column_count = len(array)
    
    return [row_count,column_count]

def Sigmoid(array):
    
    output = []
    if ArraySizer(array)[0] > 1:
        for row in array:
            row_vals = []
            for value in row:
                try:
                    row_vals.append(1/(1+math.exp(-(value))))
                except(OverflowError):
                    if value > 0:
                        row_vals.append(1)
                    else:
                        row_vals.append(0)
            output.append(row_vals)
        return output
    else:
        row_vals = []
        for value in array:
            try:
                row_vals.append(1/(1+math.exp(-(value))))
            except(OverflowError):
                if value > 0:
                    row_vals.append(1)
                else:
                    row_vals.append(0)
        return row_vals

def tanh(array):
    
    output = []
    if ArraySizer(array)[0]>1:
        for row in array:
            row_vals = []
            for value in row:
                try:
                    row_vals.append((math.exp(2*value)-1)/(1+math.exp(2*value)))
                except(OverflowError):
                    if value > 0:
                        row_vals.append(1)
                    else:
                        row_vals.append(-1)
            output.append(row_vals)
        return output
    else:
        row_vals = []
        for value in array:
            try:
                row_vals.append(1/(1+math.exp(-(value))))
            except(OverflowError):
                if value > 0:
                    row_vals.append(1)
                else:
                    row_vals.append(0)
        return row_vals
    
def sech(array):
    
    output = []
    if ArraySizer(array)[0]>1:
        for row in array:
            row_vals = []
            for value in row:
                try:
                    row_vals.append(float(mpmath.sech(value)))
                except(OverflowError):
                    if value > 0:
                        row_vals.append(1)
                    else:
                        row_vals.append(-1)
            output.append(row_vals)
        return output
    else:
        row_vals = []
        for value in array:
            try:
                row_vals.append(float(mpmath.sech(value)))
            except(OverflowError):
                if value > 0:
                    row_vals.append(1)
                else:
                    row_vals.append(0)
        return row_vals
    
def PointwiseMultiplication(V1,V2):
    
    output = []
    if ArraySizer(V1)==ArraySizer(V2):
        
        if ArraySizer(V1)[0]>1:
            for i in range(len(V1)):
                row_vals = []
                for j in range(len(V1[0])):
                    row_vals.append(V1[i][j]*V2[i][j])
                output.append(row_vals)
            return output
        else:
            row_vals = []
            for j in range(len(V1)):
                row_vals.append(V1[j]*V2[j])
            return row_vals
    else:
        print("Matrices are not same size, incompatible for pointwise multiplication")
        
def PointwiseAddition(V1,V2):
    
    output = []
    if ArraySizer(V1)==ArraySizer(V2):
        
        if ArraySizer(V1)[0]>1:
            for i in range(len(V1)):
                row_vals = []
                for j in range(len(V1[0])):
                    row_vals.append(V1[i][j]+V2[i][j])
                output.append(row_vals)
            return output
        else:
            row_vals = []
            for j in range(len(V1)):
                row_vals.append(V1[j]+V2[j])
            return row_vals
    else:
        print("Matrices are not same size, incompatible for pointwise addition")


def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def reverse_qsort(array=[12,4,5,6,7,3,1,15]):
    """Sort the array by using quicksort."""

    less = []
    equal = []
    greater = []

    if len(array) > 1:
        pivot = array[0]
        for x in array:
            if x < pivot:
                less.append(x)
            elif x == pivot:
                equal.append(x)
            elif x > pivot:
                greater.append(x)
        # Don't forget to return something!
        return reverse_qsort(greater)+equal+reverse_qsort(less) # Just use the + operator to join lists
    # Note that you want equal ^^^^^ not pivot
    else:  # You need to handle the part at the end of the recursion - when you only have one element in your array, just return the array.
        return array

def qsort(array=[12,4,5,6,7,3,1,15]):
    """Sort the array by using quicksort."""

    less = []
    equal = []
    greater = []

    if len(array) > 1:
        pivot = array[0]
        for x in array:
            if x < pivot:
                less.append(x)
            elif x == pivot:
                equal.append(x)
            elif x > pivot:
                greater.append(x)
        # Don't forget to return something!
        return reverse_qsort(less)+equal+reverse_qsort(greater) # Just use the + operator to join lists
    # Note that you want equal ^^^^^ not pivot
    else:  # You need to handle the part at the end of the recursion - when you only have one element in your array, just return the array.
        return array
  
def normalize(array):
    max_val = max(array)
    min_val = min(array)
    
    output = []
    
    for val in array:
        output.append((val-min_val)/(max_val-min_val))
        
    argmax = np.argmax(array)
    scale_factor = array[argmax]/output[argmax]
    
    return output,scale_factor