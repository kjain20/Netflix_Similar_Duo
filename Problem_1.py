#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 13:50:37 2018
"""

from datetime import datetime
from scipy import sparse
from collections import Counter
import numpy as np
import scipy


start = datetime.now()

with open("../Netflix_data.txt") as f:
    data = f.read().split("\n")[:-1]

user_id_list = []
movie_id_list = [] 
for line in data:
    if len(line.split(",")) != 1 and int(line.split(",")[1]) >= 3:
        user_id_list.append(int(line.split(",")[0]))
    elif len(line.split(",")) == 1:
        movie_id_list.append(int(line[:-1]))   

movie_id_list = np.unique(movie_id_list)
user_rating_count_dict = dict(Counter(user_id_list))
considered_users = [a for a in user_rating_count_dict.keys() if user_rating_count_dict[a] <= 20 and user_rating_count_dict[a] >= 1]         

movie_index = {k: v for v, k in enumerate(movie_id_list)}
user_index = {k: v for v, k in enumerate(considered_users)}


row_ind = []
col_ind = []
values = []
for line in data:
    flag = 0
    if len(line.split(",")) == 1:
        movie_id = int(line[:-1])
        flag = 1
    else:
        user_id = int(line.split(",")[0])
        rating = int(line.split(",")[1])
        if rating >= 3:
            rating = 1
        else:
            rating = 0
    if flag == 0:
        try:
            col_ind.append(user_index[user_id])
            row_ind.append(movie_index[movie_id])
            values.append(rating)
        except KeyError:
            pass    # USER NOT IN CONSIDERED_USERS

sparse_mat = sparse.coo_matrix((values, (row_ind, col_ind)))
scipy.sparse.save_npz("Part_1_Sparse_Matrix.npz",sparse_mat)
np.save('User_Index.npy', user_index)


end = datetime.now()

print(end-start)



