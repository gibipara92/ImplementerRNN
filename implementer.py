from __future__ import print_function
from __future__ import division
import matplotlib
matplotlib.use('agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import tqdm
import argparse
from torch.autograd import Variable, Function
from torchvision import datasets, transforms
from IPython import embed
from torchvision.utils import save_image
from pandas import *
from copy import deepcopy
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')
import seaborn as sns
from scipy.misc import imsave
import math
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch MNIST RGAN')
# Choose dataset
parser.add_argument('--dataset', default='mnist', help='cifar10 | lsun | imagenet | folder | lfw', type=str)
# Specify size of images in dataset
parser.add_argument('--imsize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--epochs', type=int, default=100, help='the height / width of the input image to network')
parser.add_argument('--p_dim', type=int, default=20, help='the program size to network')
parser.add_argument('--path_img',   default='~/Downloads/implmenter_ims', help='Path', type=str)
args = parser.parse_args()
args.cuda = True


def generate_translations_dataset():
    '''
    Generate list of meaningful convolutional filters
    Features are all combinations of:
    Translation (up to 3 in any direction)
    Inversion (either inversion or no inversion)
    Box blurring for boxes of sizes (1x1) (3x3) (5x5) and (7x7)
    Any filter generated from within this domain space can be fully specified by the above parameters
    '''
    features = []
    for i in range(7):
        for j in range(7):
            for inv in range(2):
                for blur in [0,1,2,3]:
                    filter = np.zeros((7,7))
                    for k in range(i-blur,i+blur+1):
                        for l in range(j-blur,j+blur+1):
                            if k < 0 or l < 0 or k > 6 or l > 6:
                                continue
                            else:
                                filter[k][l] = 1. / (((2 * blur) + 1) ** 2)
                    if inv == 0:
                        filter *= -1
                    features.append(torch.from_numpy(np.array(filter)))
    return features

class Dataset(object):
    def __init__(self, batch_size):
        dataset = datasets.MNIST(root='/home/ubuntu/PycharmProjects/IndependentMechanisms/mnist', train=True, download=True,
                                 transform=transforms.Compose([
                                     # transforms.Scale(args.imsize-4),
                                     transforms.Pad(padding=2),
                                     transforms.ToTensor(),
                                     transforms.Lambda(lambda x: (x * 2.0) - 1.0),
                                 ]))
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=False,
                                                      shuffle=True, num_workers=int(2))
        dataset_test = datasets.MNIST(root='/home/ubuntu/PycharmProjects/IndependentMechanisms/mnist', train=False, download=True,
                                      transform=transforms.Compose([
                                          # transforms.Scale(args.imsize-4),
                                          transforms.Pad(padding=2),
                                          transforms.ToTensor(),
                                          transforms.Lambda(lambda x: (x * 2.0) - 1.0),
                                      ]))
        self.test_size = 1000
        self.dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=self.test_size, drop_last=False,
                                                 shuffle=False, num_workers=int(2))
        # self.pseudo_targets = Variable(torch.cat(self.compute_pseudo_targets()).view(10,1,args.imsize,args.imsize)).cuda()
        self.load_all_dataset()

    def load_all_dataset(self):
        self.all_X_or, self.all_targets = [], []
        print('Loading in memory the whole dataset')
        for it_epc, (X_or, target) in tqdm(enumerate(self.dataloader, 0)):
            self.all_targets.append(target)
            self.all_X_or.append(X_or)
        self.all_X_or = torch.cat(self.all_X_or)
        self.all_targets = torch.cat(self.all_targets)

class Hypernet(nn.Module):
    def __init__(self, p_dim, input_dim, output_dim, dataloader=None):
        self.train_loss = []
        self.accuracy = []
        super(Hypernet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.p_dim = p_dim
        self.hid_dim = 64  # p_dim * 2
        self.lstm_size = 128 # args.imsize ** 2
        sigma = 0.01
        self.lstm1 = torch.nn.LSTMCell(input_size=1, hidden_size=self.lstm_size)
        self.lstm2 = torch.nn.LSTMCell(input_size=self.lstm_size, hidden_size=self.lstm_size)
        self.fc1 = nn.Parameter(torch.randn(self.lstm_size, 1) * sigma)


        if dataloader is None:
            self.dataloader = Dataset(batch_size=256)
        else:
            self.dataloader = dataloader

    def implement_W(self, p):
        p = p.view(1, args.p_dim, 1)
        outputs = []
        h_t1 = Variable(torch.zeros(p.size(0), self.lstm_size).float(), requires_grad=False).cuda()
        c_t1 = Variable(torch.zeros(p.size(0), self.lstm_size).float(), requires_grad=False).cuda()
        h_t2 = Variable(torch.zeros(p.size(0), self.lstm_size).float(), requires_grad=False).cuda()
        c_t2 = Variable(torch.zeros(p.size(0), self.lstm_size).float(), requires_grad=False).cuda()

        for i, input_t in enumerate(p.chunk(p.size(1), dim=1)):
            h_t1, c_t1 = self.lstm1(input_t, (h_t1, c_t1))
        c1 = 0
        while c1 < self.lstm_size:
            for i, input_t in enumerate(p.chunk(p.size(1), dim=1)):
                h_t1, c_t1 = self.lstm(input_t, (h_t1, c_t1))
                outputs += [torch.mm(h_t, self.fc1)]
                c1 += 1
            h1 = torch.stack(outputs, 1).squeeze()
        for i, input_t in enumerate(h1.chunk(h1.size(1), dim=1)):
            h_t2, c_t2 = self.lstm2(input_t, (h_t2, c_t2))
        # Counter increments until weight vector of generated filter is complete
        while c2 < self.output_dim:
            for i, input_t in enumerate(p.chunk(p.size(1), dim=1)):
                h_t2, c_t2 = self.lstm(input_t, (h_t2, c_t2))
                outputs += [torch.mm(h_t, self.fc1)]
            h2 = torch.stack(outputs, 1).squeeze()
        return h2.view(self.input_dim, self.output_dim), None


    def forward(self, p, input_data):
        conv_mat = self.implement_W(p)
        output = F.conv2d(input=input_data, weight=conv_mat, bias=None, padding=3)
        # output = F.linear(input=output, weight=weight2)
        return output
        # return weight

def train_epoch(H, programs, optimizers, epoch, dataloader, train_H, signature=False, task_id=''):
    H.train()
    new_loss = []
    for batch_idx, (data, _) in enumerate(dataloader):
        for program_id, p in enumerate(programs):
            data = data.view(256, 32, 32)
            target = F.conv2d(input=data, weight=functions[program_id], bias=None, padding=3)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            H.optimizer.zero_grad()
            optimizers[program_id].zero__grad()
            output = H.forward(p, data.view(-1, args.imsize, args.imsize))
            #loss = F.MSELoss(output, target)
            if train_H and not args.no_train_H and epoch > 0:
                H.optimizer.step()
            if not args.no_train_p and epoch > 0:
                p_optimize.step()
            # Add timestep regularization after I figure it out
            if batch_idx > 25 and args.debug:
                break
            if batch_idx % 200 == 0:
                print('...')
            if epoch == 0:
                print('Skip 0th epoch')
                break
            new_loss.append(loss.data.cpu()[0])
    mean_loss = np.mean(new_loss)
    H.train_loss.append(mean_loss)
    H.accuracy.append(100.0 * correct / ((batch_idx + 1) * args.batch_size))
    print('* Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Train accuracy: {}/{} ({:.0f}%) \n'.format(
        epoch, len(dataloader), len(dataloader),
        100. * batch_idx / len(dataloader), mean_loss,
        correct, (batch_idx + 1) * args.batch_size, 100.0 * correct / ((batch_idx + 1) * args.batch_size)))
    return


def train_net(H, programs, optimizers, epochs, dataloader, train_H, task_id='', old_p=[]):
    for e in range(epochs):
        if not e % 1:
            if e > 0:
                plot_all(p_progress, task_id=task_id)
            #conv_weight = H.implement_W(p)
            #save_image(1 - conv_weight.data.cpu().view(-1, 1, 7, 7),
            #           args.path_img + '/task%s_weights_epc%06d.png' % (task_id, e), nrow=10, normalize=True, scale_each=True)
        train_epoch(H, programs, optimizers, epoch=e, dataloader=dataloader, train_H=train_H, task_id=task_id)

        # stop if very good already
        if H.accuracy[-1] > 99.0:
            print('# # # Good enough, stop training')
            break

    return

H = Hypernet(p_dim=args.p_dim, input_dim=args.imsize * args.imsize, output_dim=49).cuda()
H.optimizer = optim.Adam(H.parameters(), lr=0.001)

programs = []
optimizers = []

for i in range(392):
    programs.append(nn.Parameter(Variable(torch.randn((args.p_dim, 1)).cuda(), requires_grad=True).data))

for i in range(392):
    optimizers.append(optim.Adam([programs[i]], weight_decay=0.0, lr=0.01))

functions = generate_translations_dataset()

train_net(H, programs, optimizers, 1000, H.dataloader.dataloader, True)