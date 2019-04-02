import os
import argparse
import torch
import scipy.io as sio
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import *
from models import *
import functools
import numpy as np

def eculideanDis(a, b):
    return np.sqrt(np.sum((a - b) ** 2, 1))

def test_Holidays(ori_name, dst_name, d2d, ori_dim, dst_dim):
    print('----> Loading data......')
    Holidays_base_set = dataFromFolder('./data/test/Holidays/base', ori = ori_name, dst = dst_name)
    Holidays_base_loader = DataLoader(dataset = Holidays_base_set, num_workers = 4, batch_size = 1, shuffle = False)

    Holidays_query_set = dataFromFolder('./data/test/Holidays/query', ori = ori_name, dst = dst_name)
    Holidays_query_loader = DataLoader(dataset = Holidays_query_set, num_workers = 4, batch_size = 1, shuffle = False)

    ngpu = 1

    ori = Variable(torch.FloatTensor(1, ori_dim)).cuda()
    dst = Variable(torch.FloatTensor(1, dst_dim)).cuda()

    Es = Encoder(ngpu, ori_dim).cuda()
    Es_model_dir = "./checkpoint/Holidays/{}/Es.pth".format(d2d)
    print(Es_model_dir)
    Es_state_dict = torch.load(Es_model_dir)
    Es.load_state_dict(Es_state_dict)

    Et = Encoder(ngpu, dst_dim).cuda()
    Et_model_dir = "./checkpoint/Holidays/{}/Et.pth".format(d2d)
    print(Et_model_dir)
    Et_state_dict = torch.load(Et_model_dir)
    Et.load_state_dict(Et_state_dict)

    D = Decoder(ngpu, dst_dim).cuda()
    D_model_dir = "./checkpoint/Holidays/{}/D.pth".format(d2d)
    print(D_model_dir)
    D_state_dict = torch.load(D_model_dir)
    D.load_state_dict(D_state_dict)

    Es.eval()
    Et.eval()
    D.eval()

    cnt = 0.0
    total_translation_error = 0.0
    total_reconstruction_error = 0.0

    # base
    for iteration, batch in enumerate(Holidays_base_loader, 1):
        ori_cpu, dst_cpu, name = batch[0], batch[1], batch[2]
        ori.data.resize_(ori_cpu.size()).copy_(ori_cpu)
        dst.data.resize_(dst_cpu.size()).copy_(dst_cpu)

        # translation
        ori_latent = Es(ori)
        dst_rec = D(ori_latent)

        dst_rec = dst_rec.cpu().data.numpy()

        if not os.path.exists("./result/Holidays/base"):
            os.mkdir("./result/Holidays/base")
        if not os.path.exists("./result/Holidays/base/{}".format(d2d)):
            os.mkdir("./result/Holidays/base/{}".format(d2d))
        sio.savemat("./result/Holidays/base/{}/{}".format(d2d, name[0]), {dst_name : dst_rec})

        cnt = cnt + 1
        # translation error
        translation_error = eculideanDis(dst_rec, dst_cpu.numpy())

        # reconstruction
        ori_latent = Et(dst)
        dst_rec = D(ori_latent)

        dst_rec = dst_rec.cpu().data.numpy()

        # reconstruction error
        reconstruction_error = eculideanDis(dst_rec, dst_cpu.numpy())

        total_translation_error = total_translation_error + translation_error
        total_reconstruction_error = total_reconstruction_error + reconstruction_error


    # Holidays_now_mAP = computeGroundTruth_2(base_dataset_dir = './result/Holidays', 
                                            # query_dataset_dir = './data/test/Holidays',
                                            # base_d2d = d2d, query_d2d = dst_name,
                                            # descriptor_name = dst_name, descriptor_dim = dst_dim)
    dis_difference = 1.0 / cnt * (total_translation_error - total_reconstruction_error)
    return dis_difference#, Holidays_now_mAP

def test_Oxford5k(ori_name, dst_name, d2d, ori_dim, dst_dim):
    print('----> Loading data......')
    Oxford5k_base_set = dataFromFolder('./data/test/Oxford5k/base', ori = ori_name, dst = dst_name)
    Oxford5k_base_loader = DataLoader(dataset = Oxford5k_base_set, num_workers = 4, batch_size = 1, shuffle = False)

    Oxford5k_query_set = dataFromFolder('./data/test/Oxford5k/query', ori = ori_name, dst = dst_name)
    Oxford5k_query_loader = DataLoader(dataset = Oxford5k_query_set, num_workers = 4, batch_size = 1, shuffle = False)

    ngpu = 1

    ori = Variable(torch.FloatTensor(1, ori_dim)).cuda()
    dst = Variable(torch.FloatTensor(1, dst_dim)).cuda()

    Es = Encoder(ngpu, ori_dim).cuda()
    Es_model_dir = "./checkpoint/Oxford5k/{}/Es.pth".format(d2d)
    Es_state_dict = torch.load(Es_model_dir)
    Es.load_state_dict(Es_state_dict)

    Et = Encoder(ngpu, dst_dim).cuda()
    Et_model_dir = "./checkpoint/Oxford5k/{}/Et.pth".format(d2d)
    Et_state_dict = torch.load(Et_model_dir)
    Et.load_state_dict(Et_state_dict)

    D = Decoder(ngpu, dst_dim).cuda()
    D_model_dir = "./checkpoint/Oxford5k/{}/D.pth".format(d2d)
    D_state_dict = torch.load(D_model_dir)
    D.load_state_dict(D_state_dict)

    Es.eval()
    Et.eval()
    D.eval()

    cnt = 0.0
    total_translation_error = 0.0
    total_reconstruction_error = 0.0

    # base
    for iteration, batch in enumerate(Oxford5k_base_loader, 1):
        ori_cpu, dst_cpu, name = batch[0], batch[1], batch[2]
        ori.data.resize_(ori_cpu.size()).copy_(ori_cpu)
        dst.data.resize_(dst_cpu.size()).copy_(dst_cpu)

        # translation
        ori_latent = Es(ori)
        dst_rec = D(ori_latent)

        dst_rec = dst_rec.cpu().data.numpy()

        if not os.path.exists("./result/Oxford5k/base"):
            os.mkdir("./result/Oxford5k/base")
        if not os.path.exists("./result/Oxford5k/base/{}".format(d2d)):
            os.mkdir("./result/Oxford5k/base/{}".format(d2d))
        sio.savemat("./result/Oxford5k/base/{}/{}".format(d2d, name[0]), {dst_name : dst_rec})

        cnt = cnt + 1
        # translation error
        translation_error = eculideanDis(dst_rec, dst_cpu.numpy())

        # reconstruction
        ori_latent = Et(dst)
        dst_rec = D(ori_latent)

        dst_rec = dst_rec.cpu().data.numpy()

        # reconstruction error
        reconstruction_error = eculideanDis(dst_rec, dst_cpu.numpy())

        total_translation_error = total_translation_error + translation_error
        total_reconstruction_error = total_reconstruction_error + reconstruction_error


    # Oxford5k_now_mAP = computeGroundTruth_1(base_dataset_dir = './result/Oxford5k', 
    #                                         query_dataset_dir = './data/test/Oxford5k',
    #                                         base_d2d = d2d, query_d2d = dst_name,
    #                                         descriptor_name = dst_name, descriptor_dim = dst_dim)
    dis_difference = 1.0 / cnt * (total_translation_error - total_reconstruction_error)
    return dis_difference#, Oxford5k_now_mAP

def test_Paris6k(ori_name, dst_name, d2d, ori_dim, dst_dim):
    print('----> Loading data......')
    Paris6k_base_set = dataFromFolder('./data/test/Paris6k/base', ori = ori_name, dst = dst_name)
    Paris6k_base_loader = DataLoader(dataset = Paris6k_base_set, num_workers = 4, batch_size = 1, shuffle = False)

    Paris6k_query_set = dataFromFolder('./data/test/Paris6k/query', ori = ori_name, dst = dst_name)
    Paris6k_query_loader = DataLoader(dataset = Paris6k_query_set, num_workers = 4, batch_size = 1, shuffle = False)

    ngpu = 1

    ori = Variable(torch.FloatTensor(1, ori_dim)).cuda()
    dst = Variable(torch.FloatTensor(1, dst_dim)).cuda()

    Es = Encoder(ngpu, ori_dim).cuda()
    Es_model_dir = "./checkpoint/Paris6k/{}/Es.pth".format(d2d)
    Es_state_dict = torch.load(Es_model_dir)
    Es.load_state_dict(Es_state_dict)

    Et = Encoder(ngpu, dst_dim).cuda()
    Et_model_dir = "./checkpoint/Paris6k/{}/Et.pth".format(d2d)
    Et_state_dict = torch.load(Et_model_dir)
    Et.load_state_dict(Et_state_dict)

    D = Decoder(ngpu, dst_dim).cuda()
    D_model_dir = "./checkpoint/Paris6k/{}/D.pth".format(d2d)
    D_state_dict = torch.load(D_model_dir)
    D.load_state_dict(D_state_dict)

    Es.eval()
    Et.eval()
    D.eval()

    cnt = 0.0
    total_translation_error = 0.0
    total_reconstruction_error = 0.0

    # base
    for iteration, batch in enumerate(Paris6k_base_loader, 1):
        ori_cpu, dst_cpu, name = batch[0], batch[1], batch[2]
        ori.data.resize_(ori_cpu.size()).copy_(ori_cpu)
        dst.data.resize_(dst_cpu.size()).copy_(dst_cpu)

        # translation
        ori_latent = Es(ori)
        dst_rec = D(ori_latent)

        dst_rec = dst_rec.cpu().data.numpy()

        if not os.path.exists("./result/Paris6k/base"):
            os.mkdir("./result/Paris6k/base")
        if not os.path.exists("./result/Paris6k/base/{}".format(d2d)):
            os.mkdir("./result/Paris6k/base/{}".format(d2d))
        sio.savemat("./result/Paris6k/base/{}/{}".format(d2d, name[0]), {dst_name : dst_rec})

        cnt = cnt + 1
        # translation error
        translation_error = eculideanDis(dst_rec, dst_cpu.numpy())

        # reconstruction
        ori_latent = Et(dst)
        dst_rec = D(ori_latent)

        dst_rec = dst_rec.cpu().data.numpy()

        # reconstruction error
        reconstruction_error = eculideanDis(dst_rec, dst_cpu.numpy())

        total_translation_error = total_translation_error + translation_error
        total_reconstruction_error = total_reconstruction_error + reconstruction_error


    # Paris6k_now_mAP = computeGroundTruth_1(base_dataset_dir = './result/Paris6k', 
    #                                         query_dataset_dir = './data/test/Paris6k',
    #                                         base_d2d = d2d, query_d2d = dst_name,
    #                                         descriptor_name = dst_name, descriptor_dim = dst_dim)
    dis_difference = 1.0 / cnt * (total_translation_error - total_reconstruction_error)
    return dis_difference#, Paris6k_now_mAP

if __name__ == "__main__":
    #"hesaffsiftfv", "hesaffsiftvlad",
    descs = ["delffv", "delfvlad",  
                     "resnetcrow", "resnetgem", "resnetmac", "resnetrgem", "resnetrmac", "resnetspoc",
                     "vggcrow", "vgggem", "vggmac", "vggrgem", "vggrmac", "vggspoc",
                     "siftfv", "siftvlad"]
    dims = {"delffv":2048, "delfvlad":2048, "hesaffsiftfv":2048, "hesaffsiftvlad":2048, "siftfv":2048, "siftvlad":2048,
                     "resnetcrow":2048, "resnetgem":2048, "resnetmac":2048, "resnetrgem":2048, "resnetrmac":2048, "resnetspoc":2048,
                     "vggcrow":512, "vgggem":512, "vggmac":512, "vggrgem":512, "vggrmac":512, "vggspoc":512}

    # mAP_filename = './result/mAP_result_Paris6k.txt'
    # mAP_file = open(mAP_filename, 'a')

    dis_filename = './result/dis_result.txt'
    dis_file = open(dis_filename, 'a')

    re = "Holidays Result:\n"
    # mAP_file.write(re)
    dis_file.write(re)
    for ori in descs:
        re = ori + ': '
        # mAP_file.write(re)
        for dst in descs:
            d2d = ori + '_' + dst
            ori_dim = dims[ori]
            dst_dim = dims[dst]
            dis = test_Holidays(ori, dst, d2d, ori_dim, dst_dim)

            # re = "{:.2f} ".format(mAP * 100)
            # mAP_file.write(re)
            # print(re)

            re = "{:.8f} ".format(dis[0])
            dis_file.write(re)
            print(re)

        # mAP_file.write('\n')
        dis_file.write('\n')

    re = "Oxford5k Result:\n"
    # mAP_file.write(re)
    dis_file.write(re)
    for ori in descs:
        re = ori + ': '
        # mAP_file.write(re)
        for dst in descs:
            d2d = ori + '_' + dst
            ori_dim = dims[ori]
            dst_dim = dims[dst]
            dis = test_Oxford5k(ori, dst, d2d, ori_dim, dst_dim)

            # re = "{:.2f} ".format(mAP * 100)
            # mAP_file.write(re)
            # print(re)

            re = "{:.8f} ".format(dis[0])
            dis_file.write(re)
            print(re)

        # mAP_file.write('\n')
        dis_file.write('\n')

    re = "Paris6k Result:\n"
    # mAP_file.write(re)
    dis_file.write(re)
    for ori in descs:
        re = ori + ': '
        # mAP_file.write(re)
        for dst in descs:
            d2d = ori + '_' + dst
            ori_dim = dims[ori]
            dst_dim = dims[dst]
            dis = test_Paris6k(ori, dst, d2d, ori_dim, dst_dim)

            # re = "{:.2f} ".format(mAP * 100)
            # mAP_file.write(re)
            # print(re)

            re = "{:.8f} ".format(dis[0])
            dis_file.write(re)
            print(re)

        # mAP_file.write('\n')
        dis_file.write('\n')


    # mAP_file.close()
    dis_file.close()
