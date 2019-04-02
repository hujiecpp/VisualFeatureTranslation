import numpy as np
import os
import scipy.io as sio
import natsort
from delf import feature_io
# from match_images import *
# import argparse
import pdb

datasets = ['Oxford5k', 'Paris6k', 'Holidays', 'Landmarks']

for dataset in datasets:

    query_dir = './result/' + dataset + '/query'
    query_list = natsort.natsorted(os.listdir(query_dir))
    result_dir = './delf/' + dataset + '/query'

    for q in query_list:
        # a = feature_io.ReadFromFile(query_dir + '/' + q)
        # print(len(a))
        # print(a[0].shape)
        # print(a[1].shape)
        # print(a[2].shape)
        # print(a[3].shape)
        query_l, _, query_d, _, _ = feature_io.ReadFromFile(query_dir + '/' + q)
        mat_name = result_dir + '/' + q
        sio.savemat(mat_name, {'delf' : query_d})
        print(mat_name)
        # break

    base_dir = './result/' + dataset + '/base'
    base_list = natsort.natsorted(os.listdir(base_dir))
    result_dir = './delf/' + dataset + '/base'

    for b in base_list:
        # a = feature_io.ReadFromFile(base_dir + '/' + b)
        # print(len(a))
        # print(a[0].shape)
        # print(a[1].shape)
        # print(a[2].shape)
        # print(a[3].shape)
        # print(b)
        base_l, _, base_d, _, _ = feature_io.ReadFromFile(base_dir + '/' + b)
        mat_name = result_dir + '/' + b
        sio.savemat(mat_name, {'delf' : base_d})
        print(mat_name)
        # break
    # break