import torch
import numpy as np
import scipy.io as sio
import os
from os import listdir
from os.path import join
import torch.utils.data as data
from scipy.io import loadmat

### Load data set
# read data from data set files
class dataFromFolder(data.Dataset):
    # init
    def __init__(self, data_dir, ori, dst):
        super(dataFromFolder, self).__init__()
        self.desc_ori_path = join(data_dir, ori)
        self.desc_dst_path = join(data_dir, dst)
        self.file_names = [x for x in listdir(self.desc_ori_path)]
        self.ori = ori
        self.dst = dst

    # load descriptor pair
    def __getitem__(self, index):
        file_name = self.file_names[index]
        desc_ori_tmp = sio.loadmat(join(self.desc_ori_path, file_name))
        desc_dst_tmp = sio.loadmat(join(self.desc_dst_path, file_name))

        desc_ori = torch.from_numpy(desc_ori_tmp[self.ori])
        desc_dst = torch.from_numpy(desc_dst_tmp[self.dst])

        return desc_ori.squeeze(), desc_dst.squeeze(), file_name # torch

    # return data set num
    def __len__(self):
        return len(self.file_names)

def computeGroundTruth_1(base_dataset_dir = './data/test/Oxford5k', query_dataset_dir = './data/test/Oxford5k', base_d2d = 'vgg16', query_d2d = 'vgg16', descriptor_name = 'vgg16', descriptor_dim = 4096):
    base_dir = base_dataset_dir + '/base/' + base_d2d
    query_dir = query_dataset_dir + '/query/' + query_d2d

    base_files = os.listdir(base_dir)
    bases = np.zeros((len(base_files), descriptor_dim))
    names = []
    top_k = 0
    for file in base_files:
        desc_tmp = sio.loadmat(base_dir + '/' + file)
        desc = desc_tmp[descriptor_name].reshape(-1)
        bases[top_k, :] = desc
        tmp = file.split('.')
        name = tmp[0]
        # print(name)
        names.append(name)
        top_k = top_k + 1
    # print(top_k)
    ##
    query_files = os.listdir(query_dir)
    id = 0
    mAP = 0.0
    for file in query_files:
        desc_tmp = sio.loadmat(query_dir + '/' + file)
        query = file.split('.')[0]
        resultFileName = query_dataset_dir + '/resdat/' + query + '_' + query_d2d + '.txt'
        print(resultFileName)
        resultFile = open(resultFileName, 'w')
        desc = desc_tmp[descriptor_name].reshape(-1)
        # L2
        # dist = np.sum((desc - bases) ** 2, 1)
        # cos
        dist = -np.dot(bases, desc)

        index = np.argsort(dist)
        for i in range(top_k):
            re = names[index[i]] + '\n'
            resultFile.write(re)
            # print(re)
            # print(dist[index[i]])
        id = id + 1
        t = os.popen(query_dataset_dir + '/compute_ap ' + query_dataset_dir + '/gnd/' + query + ' ' + resultFileName)
        ap = float(t.read())
        mAP = mAP + ap
        # print("Runing {}: the ap is {}.".format(id, ap))
        resultFile.close()

    print("The mAP {} : {}.".format(base_d2d, mAP / id))
    return mAP / id

def computeGroundTruth_2(base_dataset_dir = './data/test/Oxford5k', query_dataset_dir = './data/test/Oxford5k', base_d2d = 'vgg16', query_d2d = 'vgg16', descriptor_name = 'vgg16', descriptor_dim = 4096):
    base_dir = base_dataset_dir + '/base/' + base_d2d
    query_dir = query_dataset_dir + '/query/' + query_d2d

    base_files = os.listdir(base_dir)
    bases = np.zeros((len(base_files), descriptor_dim))
    names = []
    top_k = 0

    for file in base_files:
        desc_tmp = sio.loadmat(base_dir + '/' + file)
        desc = desc_tmp[descriptor_name].reshape(-1)
        bases[top_k, :] = desc
        tmp = file.split('.')
        name = tmp[0] + '.' + tmp[1]
        # print(name)
        names.append(name)
        top_k = top_k + 1

    query_files = os.listdir(query_dir)
    id = 0

    resultFileName = query_dataset_dir + '/resdat/' + query_d2d + '.dat'
    resultFile = open(resultFileName, 'w')
    print(resultFileName)

    for file in query_files:
        desc_tmp = sio.loadmat(query_dir + '/' + file)
        tmp = file.split('.')

        re = tmp[0] + '.' + tmp[1] + ' '
        resultFile.write(re)

        desc = desc_tmp[descriptor_name].reshape(-1)
        # L2
        # dist = np.sum((desc - bases) ** 2, 1)
        # cos
        dist = -np.dot(bases, desc)

        index = np.argsort(dist)
        for i in range(top_k):
            re = str(i) + ' ' + names[index[i]]
            if i == top_k - 1:
                re = re + '\n'
            else:
                re = re + ' '
            resultFile.write(re)
        id = id + 1
        # print("Runing {}.".format(id))
        # break
    resultFile.close()
    # print('python ' + query_dataset_dir + '/holidays_map.py ' + resultFileName + ' ' + query_dataset_dir + '/holidays_images.dat')
    t = os.popen('python2 ' + query_dataset_dir + '/holidays_map.py ' + resultFileName + ' ' + query_dataset_dir + '/holidays_images.dat')
    # print(t.read())
    mAP = float(t.read())
    print("The mAP {} : {}.".format(base_d2d, mAP))
    return mAP
