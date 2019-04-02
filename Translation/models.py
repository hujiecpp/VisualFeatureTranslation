import torch
import torch.nn as nn
from torch.nn import init
import functools

def weightsInit(layer):
    classname = layer.__class__.__name__
    if classname.find('Linear') != -1:
        init.xavier_normal(layer.weight.data, gain=1)

class Encoder(nn.Module):
    def __init__(self, ngpu, dim):
        super(Encoder, self).__init__()
        self.ngpu = ngpu

        if dim == 2048:
            parameters = [2048, 2048, 2048, 510]
        if dim == 512:
            parameters = [512, 512, 510]

        seq = []
        for i in range(len(parameters) - 1):
            #
            seq = seq + [nn.Linear(parameters[i], parameters[i + 1])] + [nn.ELU()]
        self.model = nn.Sequential(*seq)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.model, input, range(self.ngpu))
        else:
            output = self.model(input)
        return output

class Decoder(nn.Module):
    def __init__(self, ngpu, dim):
        super(Decoder, self).__init__()
        self.ngpu = ngpu

        if dim == 2048:
            parameters = [510, 2048, 2048, 2048]
        if dim == 512:
            parameters = [510, 512, 512]

        seq = []
        for i in range(len(parameters) - 1):
            seq = seq + [nn.Linear(parameters[i], parameters[i + 1])]
            if i == len(parameters) - 2:
                seq = seq #+ [nn.Tanh()]
            else:
                # + [nn.BatchNorm1d(parameters[i + 1])]+ [nn.Dropout(0.5)]
                seq = seq + [nn.ELU()]
        self.model = nn.Sequential(*seq)
    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.model, input, range(self.ngpu))
        else:
            output = self.model(input)

        output = nn.functional.normalize(output)

        return output
