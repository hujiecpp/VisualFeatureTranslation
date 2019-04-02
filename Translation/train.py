import os
import argparse
import torch
import scipy.io as sio
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import *
from utils import *
import functools
import numpy as np

### Training parament setting
parser = argparse.ArgumentParser(description = 'D2D translation implementation')
parser.add_argument('--ori', required = True, help = 'fv')
parser.add_argument('--ori_dimension', type = int, default = 2048, help = 'dimension of the ori descriptor')
parser.add_argument('--dst', required = True, help = 'fv')
parser.add_argument('--dst_dimension', type = int, default = 2048, help = 'dimension of the ori descriptor')
parser.add_argument('--nEpochs', type = int, default = 10, help = 'number of epochs to train the model')
parser.add_argument('--lr', type = float, default = 0.000001, help = 'Learning Rate. Default = 0.002')
parser.add_argument('--beta1', type = float, default = 0.5, help = 'beta1 for adam. default = 0.5')
parser.add_argument('--cuda', action = 'store_true', help = 'use cuda?')
parser.add_argument('--threads', type = int, default = 8, help = 'number of threads for data loader to use')
parser.add_argument('--seed', type = int, default = 2333, help = 'random seed to use. Default=233')
opt = parser.parse_args()
opt.d2d = opt.ori + '_' + opt.dst
print(opt)

# ### batch size
batch_size = 5 #10 50 / 10 / best:5

### cuda setting
if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
### uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms
cudnn.benchmark = True

### random seed
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

### Load data
print('----> Loading data......')
train_set = dataFromFolder('./data/train/Landmarks/base', ori = opt.ori, dst = opt.dst)
training_data_loader = DataLoader(dataset = train_set, num_workers = opt.threads, batch_size = batch_size, shuffle = True)

### Init models
print('----> Initialize models......')
ngpu = 1
ori_dim = opt.ori_dimension
dst_dim = opt.dst_dimension

Es = Encoder(ngpu, ori_dim)
Es.apply(weightsInit)

Et = Encoder(ngpu, dst_dim)
Et.apply(weightsInit)

D = Decoder(ngpu, dst_dim)
D.apply(weightsInit)

### Init setting
loss_MSE = nn.MSELoss()

ori = Variable(torch.FloatTensor(batch_size, ori_dim))
dst = Variable(torch.FloatTensor(batch_size, dst_dim))

if opt.cuda:
    Es = Es.cuda()
    Et = Et.cuda()

    D = D.cuda()
    loss_MSE = loss_MSE.cuda()
    ori = ori.cuda()
    dst = dst.cuda()

### optimizer
optimizer_Es = optim.Adam(Es.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay = 1e-5)#1
optimizer_Et = optim.Adam(Et.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay = 1e-5)
optimizer_D = optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay = 1e-5)

def train(epoch):
    Es.train()
    Et.train()

    D.train()
    sz = len(training_data_loader)
    for iteration, batch in enumerate(training_data_loader, 1):
        ori_cpu, dst_cpu = batch[0], batch[1]
        ori.data.resize_(ori_cpu.size()).copy_(ori_cpu)
        dst.data.resize_(dst_cpu.size()).copy_(dst_cpu)

        ori_latent = Es(ori)
        dst_rec = D(ori_latent)

        dst_latent = Et(dst)
        dst_self_rec = D(dst_latent)

        err_g_ori = loss_MSE(dst_rec, dst)
        err_g_dst = loss_MSE(dst_self_rec, dst)
        err_g =  err_g_ori + err_g_dst
        err_g.backward()
        out_ori = err_g_ori.data.mean()
        out_dst = err_g_dst.data.mean()
        optimizer_Es.step()
        optimizer_Et.step()
        optimizer_D.step()
        if iteration % 1000 == 0:
            print("===> Epoch[{}]({}/{}), loss_ori: [{:.10f}], loss_dst: [{:.10f}]".format(epoch, iteration, sz, out_ori, out_dst))

def test_Oxford5k(epoch, Oxford5k_gnd_mAP):
    Es.eval()
    D.eval()
    # base
    for iteration, batch in enumerate(Oxford5k_base_loader, 1):
        ori_cpu, dst_cpu, name = batch[0], batch[1], batch[2]
        ori.data.resize_(ori_cpu.size()).copy_(ori_cpu)
        dst.data.resize_(dst_cpu.size()).copy_(dst_cpu)

        ori_latent = Es(ori)
        dst_rec = D(ori_latent)

        dst_rec = dst_rec.cpu().data.numpy()
        ori_latent = ori_latent.cpu().data.numpy()

        if not os.path.exists("./result"):
            os.mkdir("./result")
        if not os.path.exists("./result/Oxford5k"):
            os.mkdir("./result/Oxford5k")
        if not os.path.exists("./result/Oxford5k/base"):
            os.mkdir("./result/Oxford5k/base")
        if not os.path.exists("./result/Oxford5k/base/{}".format(opt.d2d)):
            os.mkdir("./result/Oxford5k/base/{}".format(opt.d2d))
        sio.savemat("./result/Oxford5k/base/{}/{}".format(opt.d2d, name[0]), {opt.dst : dst_rec})

    Oxford5k_now_mAP = computeGroundTruth_1(
                        base_dataset_dir = './result/Oxford5k', 
                                         query_dataset_dir = './data/test/Oxford5k',
                                        base_d2d = opt.d2d, query_d2d = opt.dst, 
                                        descriptor_name = opt.dst, descriptor_dim = dst_dim)
    resultFileName = './result/Oxford5k/' + opt.d2d + '.txt'
    resultFile = open(resultFileName, 'a')
    re = "{} (epoch {}), mAP: {:.4f} / {:.4f}.\n".format(opt.d2d, epoch, Oxford5k_gnd_mAP, Oxford5k_now_mAP)
    resultFile.write(re)
    resultFile.close()

    return Oxford5k_now_mAP

def test_Paris6k(epoch, Paris6k_gnd_mAP):
    Es.eval()
    D.eval()
    # base
    for iteration, batch in enumerate(Paris6k_base_loader, 1):
        ori_cpu, dst_cpu, name = batch[0], batch[1], batch[2]
        ori.data.resize_(ori_cpu.size()).copy_(ori_cpu)
        dst.data.resize_(dst_cpu.size()).copy_(dst_cpu)

        ori_latent = Es(ori)
        dst_rec = D(ori_latent)

        dst_rec = dst_rec.cpu().data.numpy()
        ori_latent = ori_latent.cpu().data.numpy()

        if not os.path.exists("./result"):
            os.mkdir("./result")
        if not os.path.exists("./result/Paris6k"):
            os.mkdir("./result/Paris6k")
        if not os.path.exists("./result/Paris6k/base"):
            os.mkdir("./result/Paris6k/base")
        if not os.path.exists("./result/Paris6k/base/{}".format(opt.d2d)):
            os.mkdir("./result/Paris6k/base/{}".format(opt.d2d))
        sio.savemat("./result/Paris6k/base/{}/{}".format(opt.d2d, name[0]), {opt.dst : dst_rec})

    Paris6k_now_mAP = computeGroundTruth_1(
                        base_dataset_dir = './result/Paris6k', 
                                         query_dataset_dir = './data/test/Paris6k',
                                        base_d2d = opt.d2d, query_d2d = opt.dst, 
                                        descriptor_name = opt.dst, descriptor_dim = dst_dim)
    resultFileName = './result/Paris6k/' + opt.d2d + '.txt'
    resultFile = open(resultFileName, 'a')
    re = "{} (epoch {}), mAP: {:.4f} / {:.4f}.\n".format(opt.d2d, epoch, Paris6k_gnd_mAP, Paris6k_now_mAP)
    resultFile.write(re)
    resultFile.close()

    return Paris6k_now_mAP

def test_Holidays(epoch, Holidays_gnd_mAP):
    Es.eval()
    D.eval()
    # base
    for iteration, batch in enumerate(Holidays_base_loader, 1):
        ori_cpu, dst_cpu, name = batch[0], batch[1], batch[2]
        ori.data.resize_(ori_cpu.size()).copy_(ori_cpu)
        dst.data.resize_(dst_cpu.size()).copy_(dst_cpu)

        ori_latent = Es(ori)
        dst_rec = D(ori_latent)

        dst_rec = dst_rec.cpu().data.numpy()
        ori_latent = ori_latent.cpu().data.numpy()

        if not os.path.exists("./result"):
            os.mkdir("./result")
        if not os.path.exists("./result/Holidays"):
            os.mkdir("./result/Holidays")
        if not os.path.exists("./result/Holidays/base"):
            os.mkdir("./result/Holidays/base")
        if not os.path.exists("./result/Holidays/base/{}".format(opt.d2d)):
            os.mkdir("./result/Holidays/base/{}".format(opt.d2d))
        sio.savemat("./result/Holidays/base/{}/{}".format(opt.d2d, name[0]), {opt.dst : dst_rec})

    Holidays_now_mAP = computeGroundTruth_2(
                        base_dataset_dir = './result/Holidays', 
                                         query_dataset_dir = './data/test/Holidays',
                                        base_d2d = opt.d2d, query_d2d = opt.dst, 
                                        descriptor_name = opt.dst, descriptor_dim = dst_dim)
    resultFileName = './result/Holidays/' + opt.d2d + '.txt'
    resultFile = open(resultFileName, 'a')
    re = "{} (epoch {}), mAP: {:.4f} / {:.4f}.\n".format(opt.d2d, epoch, Holidays_gnd_mAP, Holidays_now_mAP)
    resultFile.write(re)
    resultFile.close()

    return Holidays_now_mAP

def checkpoint(dataset):
    Es_path = "checkpoint/{}/{}/Es.pth".format(dataset, opt.d2d)
    Et_path = "checkpoint/{}/{}/Et.pth".format(dataset, opt.d2d)
    D_path = "checkpoint/{}/{}/D.pth".format(dataset, opt.d2d)

    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")

    if not os.path.exists("checkpoint/{}".format(dataset)):
        os.mkdir("checkpoint/{}".format(dataset))

    if not os.path.exists("checkpoint/{}/{}".format(dataset, opt.d2d)):
        os.mkdir("checkpoint/{}/{}".format(dataset, opt.d2d))
    
    torch.save(Es.state_dict(), Es_path)
    torch.save(Et.state_dict(), Et_path)
    torch.save(D.state_dict(), D_path)

    # print("Checkpoint {} saved.".format(epoch))

if __name__ == "__main__":
    # "hesaffsiftfv":0.5691, "hesaffsiftvlad":0.5827,
    Oxford5k_mAPs = {"delffv":0.7338, "delfvlad":0.7531, "siftfv":0.3625, "siftvlad":0.4049,
                     "resnetcrow":0.6173, "resnetgem":0.8447, "resnetmac":0.6082, "resnetrgem":0.8460, "resnetrmac":0.6846, "resnetspoc":0.6236,
                     "vggcrow":0.6838, "vgggem":0.8271, "vggmac":0.6097, "vggrgem":0.8230, "vggrmac":0.7084, "vggspoc":0.6643}
    # "hesaffsiftfv":0.5509, "hesaffsiftvlad":0.5295,
    Paris6k_mAPs = {"delffv":0.8306, "delfvlad":0.8254, "siftfv":0.3691, "siftvlad":0.4149,
                     "resnetcrow":0.7546, "resnetgem":0.9187, "resnetmac":0.7774, "resnetrgem":0.9190, "resnetrmac":0.8300, "resnetspoc":0.7675,
                     "vggcrow":0.7979, "vgggem":0.8685, "vggmac":0.7265, "vggrgem":0.8733, "vggrmac":0.8354, "vggspoc":0.7847}
    # "hesaffsiftfv":0.6466, "hesaffsiftvlad":0.6405,
    Holidays_mAPs = {"delffv":0.8342, "delfvlad":0.8461, "siftfv":0.6177, "siftvlad":0.6392,
                     "resnetcrow":0.8638, "resnetgem":0.8908, "resnetmac":0.8853, "resnetrgem":0.8932, "resnetrmac":0.8908, "resnetspoc":0.8657,
                     "vggcrow":0.8317, "vgggem":0.8457, "vggmac":0.7418, "vggrgem":0.8506, "vggrmac":0.8350, "vggspoc":0.8338}

    Oxford5k_gnd_mAP = Oxford5k_mAPs[opt.ori]#computeGroundTruth_1('./data/test/Oxford5k', './data/test/Oxford5k', opt.dst, opt.dst, opt.dst, dst_dim)#
    Paris6k_gnd_mAP = Paris6k_mAPs[opt.ori]#computeGroundTruth_1('./data/test/Paris6k', './data/test/Paris6k', opt.dst, opt.dst, opt.dst, dst_dim)#
    Holidays_gnd_mAP = Holidays_mAPs[opt.ori]#computeGroundTruth_2('./data/test/Holidays', './data/test/Holidays', opt.dst, opt.dst, opt.dst, dst_dim)#

    max_Oxford5k_mAP = 0.0
    max_Paris6k_mAP = 0.0
    max_Holidays_mAP = 0.0

    for epoch in range(1, opt.nEpochs + 1):
        train(epoch)

        if epoch == opt.nEpochs:
            checkpoint('Oxford5k')
            checkpoint('Paris6k')
            checkpoint('Holidays')
