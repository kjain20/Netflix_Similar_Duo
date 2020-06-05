#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 16:06:59 2018
"""

import numpy as np
import scipy
import scipy.sparse
import random
from datetime import datetime
import matplotlib.pyplot as plt


start = datetime.now()

input_array = scipy.sparse.load_npz('Part_1_Sparse_Matrix.npz').tocsr()
user_index = np.load('User_Index.npy').item()
#movie_to_index = np.load('Movie_Index.npy').item()

random_pairs = []
for i in range(10000):
    random_pairs.append((random.randint(0,len(user_index)-1),random.randint(0,len(user_index)-1)))
    
indices = input_array.nonzero()
coordinates = [(a1,a2) for a2,a1 in zip(indices[0],indices[1])]
coordinates = sorted(coordinates, key=lambda x: x[0])
index_ones = []
for element in coordinates:
    if len(index_ones) == element[0]:
        index_ones.append([element[1]])
    else:
        index_ones[element[0]].append(element[1])

jacc_dist_list = []
for element in random_pairs:
    A = index_ones[element[0]]
    B = index_ones[element[1]]
    A_int_B = len(np.intersect1d(A,B))*1.0
    A_uni_B = len(np.union1d(A,B))*1.0
    jacc_dist = 1 - (A_int_B/A_uni_B)
    jacc_dist_list.append(jacc_dist)

plt.hist(jacc_dist_list, bins=25)
plt.xlabel('Jaccard Distance')
plt.show()

end = datetime.now()

print(end-start)