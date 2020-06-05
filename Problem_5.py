#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 17:08:52 2018
"""
import numpy as np
from datetime import datetime
import scipy
import scipy.sparse
import random
from collections import Counter
import itertools

def nearest_neighbour(user_vector):
    # user_vector dimensions: (4499,1)
    input_array = scipy.sparse.load_npz('Part_1_Sparse_Matrix.npz').toarray()
    n = 1000 
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
    
    one_ind_user = np.where(user_vector==1)[0]
    one_ind_user = np.reshape(one_ind_user,(len(one_ind_user),1))
    m = one_ind_user.shape[0]
    adder = np.dot(np.ones((m,1)),add)
    hash_values = np.mod(np.add(np.dot(one_ind_user,mult),adder),mod)
    user_sig_mat = np.min(hash_values,axis=0)
    
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
    
    similar_pairs = []
    
    for i in range(b):
        print(str(i) + " - " + str(datetime.now()))
        input_array_band = sig_mat[bands[i]]
        hash_buckets = np.sum(np.mod(np.add(np.multiply(input_array_band,mult),add),mod),axis=0)
        user_band = user_sig_mat[bands[i]]
        user_hash = np.sum(np.mod(np.add(np.multiply(user_band,mult.reshape(len(mult))),add.reshape(len(add))),mod))
        similar_pairs += list(np.where(hash_buckets==user_hash)[0])
    
    if len(similar_pairs) < 0 :
        return "No similar pairs with jacc similarity > 0.65"
    
    similar_pairs = np.unique(similar_pairs)
    
    original_array = scipy.sparse.load_npz('Part_1_Sparse_Matrix.npz').tocsr()
    
    indices = original_array.nonzero()
    coordinates = [(a1,a2) for a2,a1 in zip(indices[0],indices[1])]
    coordinates = sorted(coordinates, key=lambda x: x[0])
    index_ones = []
    for element in coordinates:
        if len(index_ones) == element[0]:
            index_ones.append([element[1]])
        else:
            index_ones[element[0]].append(element[1])
    
    sim_score = []
    
    A = list(np.where(user_vector)[0])
    for element in similar_pairs:
        B = index_ones[element]
        A_int_B = len(np.intersect1d(A,B))*1.0
        A_uni_B = len(np.union1d(A,B))*1.0
        jacc_sim = (A_int_B/A_uni_B)
        sim_score.append(jacc_sim)
    
    best_user = similar_pairs[np.argmax(sim_score)]
    
    best_user = "The closest user is user: " + str(best_user)
    return best_user
        

user_vector = scipy.sparse.load_npz('Part_1_Sparse_Matrix.npz').tocsr()[:,1].toarray()

start = datetime.now()
print(nearest_neighbour(user_vector))
end = datetime.now()
print(end-start)