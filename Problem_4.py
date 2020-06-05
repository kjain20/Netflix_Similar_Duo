#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 18:39:52 2018
"""

import numpy as np
import scipy
import random
from datetime import datetime
from collections import Counter
import csv
import itertools

n = 1000 # NUMBER OF HASH FUNCTIONS

start = datetime.now()

input_array = scipy.sparse.load_npz('Part_1_Sparse_Matrix.npz').toarray()
user_index = np.load('User_Index.npy').item()
movie_to_index = np.load('Movie_Index.npy').item()

mult = []
add = []
mod = 10007

for i in range(n):
    mult.append(random.randint(1,mod))
    add.append(random.randint(1,mod))
    
mult = np.array(mult)
mult = np.reshape(mult,(1,len(mult)))
add = np.array(add)
add = np.reshape(add,(1,len(add)))

M_i_list = []
M_i = np.add(np.zeros(input_array.shape[0]),input_array.shape[0] + mod)[:n]  # INITIAL VALUE

sig_mat = []
for i in range(input_array.shape[1]):
    one_ind = np.where(input_array[:,i]==1)[0]
    one_ind = np.reshape(one_ind,(len(one_ind),1))
    m = one_ind.shape[0]
    adder = np.dot(np.ones((m,1)),add)
    hash_values = np.mod(np.add(np.dot(one_ind,mult),adder),mod)
    hash_values = np.min(hash_values,axis=0)
    sig_mat.append(hash_values.reshape((len(hash_values),1)))

sig_mat = np.concatenate(sig_mat,axis=1)



r = 10
b = 100

bands = []
for i in range(b):
    temp = []
    for j in range(r):
        temp.append(i*r + j)
    bands.append(np.array(temp))
bands = np.array(bands)

mult = []
add = []
mod = 10000169

for i in range(r):
    mult.append(random.randint(1,mod))
    add.append(random.randint(1,mod))

add = np.array(add).reshape((len(add),1))
mult = np.array(mult).reshape((len(mult),1))

original_array = scipy.sparse.load_npz('Part_1_Sparse_Matrix.npz').tocsr()

input_array = sig_mat
user_index = np.load('User_Index.npy').item()
index_to_user = {user_index[a]:a for a in user_index.keys()}
movie_to_index = np.load('Movie_Index.npy').item()
similar_pairs = []
for i in range(b):
    input_array_band = input_array[bands[i]]
    hash_buckets = np.sum(np.mod(np.add(np.multiply(input_array_band,mult),add),mod),axis=0)
    hash_dict = Counter(hash_buckets)
    considered_hash_values = [a for a in hash_dict.keys() if hash_dict[a] > 1]
    hash_indices = []
    for element in considered_hash_values:
        indices = np.where(hash_buckets==element)[0]
        similar_pairs += list(itertools.combinations(indices,2))
similar_pairs = list(set(similar_pairs))

indices = original_array.nonzero()
coordinates = [(a1,a2) for a2,a1 in zip(indices[0],indices[1])]
coordinates = sorted(coordinates, key=lambda x: x[0])
index_ones = []
for element in coordinates:
    if len(index_ones) == element[0]:
        index_ones.append([element[1]])
    else:
        index_ones[element[0]].append(element[1])


true_positives = []
for element in similar_pairs:
    A = index_ones[element[0]]
    B = index_ones[element[1]]
    A_int_B = len(np.intersect1d(A,B))*1.0
    A_uni_B = len(np.union1d(A,B))*1.0
    jacc_sim = (A_int_B/A_uni_B)
    if jacc_sim > 0.65:
        true_positives.append([index_to_user[element[0]],index_to_user[element[1]]])

with open('similarPairs.csv','w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(true_positives)

end = datetime.now()
print(end-start)