import torch
from torch import save
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

import shutil

cudaAvailable = False
if torch.cuda.is_available():
    cudaAvailable = True
Tensor = torch.cuda.FloatTensor if cudaAvailable else torch.Tensor

def toZeroThreshold(x, t=0.1):
	zeros = Tensor(x.shape).fill_(0.0)
	return torch.where(x > t, x, zeros)

def weights_init(layer):
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(layer.weight.data)

class Plotter:
    def __init__(self, attributes = [('trainloss', 1)]):      
        self.attributes = attributes

        for dictionary in attributes:
            attr = dictionary[0]
            freq = dictionary[1]
            setattr(self, attr, []) 
            setattr(self, attr+'_freq', freq )
    def log(self, attr, value):
        getattr(self, attr).append(value)

    def savelog(self, filename):
        pass

    def plot(self, ylabel, attributes = None, ymax = None, filename = 'plot.png'):
        
        plt.style.use('ggplot')

        if ymax is not None:
            plt.ylim(ymax=ymax)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)

        # if kwargs is not None:
        #     for key, value in kwargs:
        #         getattr(plt, key)(value)
        
        if attributes is None:
            attributes = [attr[0] for attr in self.attributes]

        for attr in attributes:
            Xs = getattr(self, attr+'_freq') * np.arange(1, len(getattr(self, attr))+1)
            Ys = getattr(self, attr)
            # print(Xs)
            # print(Ys)
            plt.plot( Xs, Ys, label=attr)
        
        plt.legend()
        plt.savefig(filename)
        plt.close()

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    save(state, filename)