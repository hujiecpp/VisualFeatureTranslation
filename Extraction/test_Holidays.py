import os
import argparse
import numpy as np
import scipy.io as sio

# python test_Holidays.py --feature_name x --feature_dim x --galary_path x --query_path x
parser = argparse.ArgumentParser(description = 'Test Holidays mAP.')
parser.add_argument('--feature_name', required = True, help = 'sift_fv')
parser.add_argument('--feature_dim', type = int, default = 2048, help = 'Dimension of feature')
parser.add_argument('--galary_path', required = True, help = './Holidays/galary/')
parser.add_argument('--query_path', required = True, help = './Holidays/query/')
opt = parser.parse_args()
print(opt)

# Test Setting
feature_name = opt.feature_name
feature_dim = opt.feature_dim
galary_path = opt.galary_path + '/' + feature_name
query_path = opt.query_path + '/' + feature_name

# Loading Galary
galary_files = os.listdir(galary_path)
galary = np.zeros((len(galary_files), feature_dim))
names = []
for i, file in enumerate(galary_files):
    feat_tmp = sio.loadmat(galary_path + '/' + file)
    feat = feat_tmp[feature_name].reshape(-1)
    galary[i, :] = feat
    tmp = file.split('.')
    name = tmp[0] + '.' + tmp[1]
    # print(name)
    names.append(name)

# Loading Queries
query_files = os.listdir(query_path)
## Result File Name
result_path = './Holidays_results/'
if not os.path.exists(result_path):
    os.mkdir(result_path)

result_file_name = result_path + feature_name + '_result.dat'
print(result_file_name)
resultFile = open(result_file_name, 'w')
## Searching and Saving the Result
for file in query_files:
    feat_tmp = sio.loadmat(query_path + '/' + file)
    ### Remove .mat
    tmp = file.split('.')
    re = tmp[0] + '.' + tmp[1] + ' '
    resultFile.write(re)

    feat = feat_tmp[feature_name]
    dist = np.sum((feat - galary) ** 2, 1)
    index = np.argsort(dist)
    sz = len(names)
    for i in range(sz):
        re = str(i) + ' ' + names[index[i]]
        if i == sz - 1:
            re = re + '\n'
        else:
            re = re + ' '
        resultFile.write(re)
resultFile.close()

# Evaluating
t = os.popen('python2 holidays_map.py ' + result_file_name + ' ./holidays_images.dat')
# print(t.read())
mAP = float(t.read())
print("The mAP of {}: {}.\n".format(feature_name, mAP))
