from __future__ import print_function
from __future__ import division
import matplotlib
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
import random
import numpy as np
import torch.utils.data as utils
import joblib
from torch.optim.lr_scheduler import StepLR
import matplotlib.animation as animation
from tensorboardX import SummaryWriter
from copy import deepcopy
from shutil import copyfile
from datetime import datetime
import scipy
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import PIL
from torchvision.transforms import ToTensor
parser = argparse.ArgumentParser(description='PyTorch MNIST RGAN')
# Choose dataset
parser.add_argument('--dataset', default='/home/ubuntu/PycharmProjects/IndependentMechanisms/mnist',
                    help='path to dataset', type=str)
parser.add_argument('--meta_folder', default='/home/ubuntu/implementer_data/meta/',
                    help='path to save models/programs', type=str)
# Specify size of images in dataset
parser.add_argument('--imsize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs during training')
parser.add_argument('--test_epochs', type=int, default=5000, help='Number of epochs during testing')
parser.add_argument('--p_dim', type=int, default=20, help='the program size to network')
parser.add_argument('--lstm_size', type=int, default=128, help='Size of LSTM layers')
parser.add_argument('--mnist_size', type=int, default=200, help='Number of examples in test set')
parser.add_argument('--reg_lambda', type=float, default=0.000, help='Coefficient of regularization term')
parser.add_argument('--noise', type=float, default=0.2, help='Amount of noise to add to programs')
parser.add_argument('--H_lr', type=float, default=0.001, help='Learning rate for implementer')
parser.add_argument('--p_lr', type=float, default=0.1, help='Learning rate for programs')
parser.add_argument('--test_copies', type=int, default=50, help='Number of random seeds to try for test data')
parser.add_argument('--display_number', type=int, default=10, help='Number of random seeds to try for test data')
parser.add_argument('--path_img',   default='~/Downloads/implmenter_ims', help='Path', type=str)
parser.add_argument('--activation_function', default='tanh', help='tanh|sin|LeakyReLU', type=str)
parser.add_argument('--no_train_H', action='store_true', default=False, help='Do not train the HyperNet')
parser.add_argument('--no_train_p', action='store_true', default=False, help='Do not train the program')

args = parser.parse_args()

args2 = deepcopy(args)
del args2.meta_folder
del args2.path_img
del args2.dataset
del args2.no_train_H
del args2.no_train_p

now = datetime.now()
args2.time = now.strftime("%Y%m%d-%H%M%S")

if args.activation_function == 'tanh':
    args.activation_function = F.tanh
elif args.activation_function == 'sin':
    args.activation_function = torch.sin
elif args.activation_function == 'LeakyReLU':
    args.activation_function = F.leaky_relu
else:
    assert False


try:
    os.makedirs(args.meta_folder)
except OSError:
    pass

try:
    output_folder = args.meta_folder + '/' + str(args2)
    os.makedirs(output_folder)
except OSError:
    pass

copyfile(os.path.realpath(__file__), output_folder + "/code.py")
writer = SummaryWriter(args.meta_folder + '/runs/' + str(args2))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

def generate_sinusoid_batch(batch_size, input_idx=None):
    '''Generates a batch of sinusoids (1000 points between -5 and 5,
    with A between 0.1 and 5 and phase between 0 and pi'''
    amp = np.random.uniform(0.1, 5.0, [batch_size])
    phase = np.random.uniform(0, np.pi, [batch_size])
    outputs = np.zeros([batch_size, 1000])
    init_inputs = np.zeros([batch_size, 1000])
    for func in range(batch_size):
        init_inputs[func] = np.random.uniform(-5.0, 5.0, [1000])
        outputs[func] = amp[func] * np.sin(init_inputs[func]-phase[func])
    return init_inputs, outputs, amp, phase


class Hypernet(nn.Module):
    '''Initialize Hypernet'''
    def __init__(self, p_dim, output_dim, dataloader=None):
        self.train_loss = []
        self.mse_loss = []
        self.reg_loss = []
        self.accuracy = []
        super(Hypernet, self).__init__()
        self.output_dim = output_dim
        self.p_dim = p_dim
        self.lstm_size = args.lstm_size # args.imsize ** 2
        sigma = 0.01
        self.lstm1 = torch.nn.LSTMCell(input_size=1, hidden_size=self.lstm_size)
        self.lstm2 = torch.nn.LSTMCell(input_size=self.lstm_size, hidden_size=self.lstm_size)
        # We output weights 40 at a time
        self.fc1 = nn.Parameter(torch.randn(self.lstm_size, 40) * sigma)



    def implement_W(self, p, redundant_train=False):
        '''This is the same it has always been - generates weights from programs'''
        p = p.view(-1, p.shape[-2])
        # Add noise to programs if redundant_train is True
        if redundant_train:
            noise = torch.normal(mean=torch.zeros(p.shape), std=args.noise)
            p += noise.cuda()
        p = torch.tanh(p)
        #p = p.view(-1, args.p_dim)
        batch = p.shape[0]
        outputs = []
        h_t1 = Variable(torch.zeros(batch, self.lstm_size).float(), requires_grad=False).to(device)
        c_t1 = Variable(torch.zeros(batch, self.lstm_size).float(), requires_grad=False).to(device)
        h_t2 = Variable(torch.zeros(batch, self.lstm_size).float(), requires_grad=False).to(device)
        c_t2 = Variable(torch.zeros(batch, self.lstm_size).float(), requires_grad=False).to(device)
        #h1 = Variable(torch.zeros(p.size(0), self.lstm_size).float(), requires_grad=False).to(device)
        # Iterate through program for first LSTM layer
        for i, input_t in enumerate(p.chunk(p.shape[1], dim=1)):
            h_t1, c_t1 = self.lstm1(input_t, (h_t1, c_t1))
        c2 = 0
        # Counter increments until weight vector of generated filter is complete
        # Second LSTM layer
        while c2 < self.output_dim:
            for i, input_t in enumerate(range(self.output_dim)):
                h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
                outputs += [torch.mm(h_t2, self.fc1)]
                c2 += 40
                if c2 == self.output_dim:
                    break
            h2 = torch.stack(outputs, 1).squeeze()
        # We output 40 * 42 = 1680 weights for our two hidden layer network
        return h2.view(-1, 40, 45)

    def forward(self, p, input_data):
        # First, generate weights from program (adding noise to the programs)
        weights = self.implement_W(p, redundant_train=True)
        result = []
        # then, for each data point, apply the neural network described by the weights to the input_data
        for i in range(dataset_size):
            output = F.elu(F.linear(torch.Tensor(torch.Tensor(input_data[i]).view(-1, 1)), weight=torch.Tensor(weights[i][:, 0].cpu()).view(-1,1)) + weights[i][:, 1].cpu().repeat(500, 1))
            output = F.elu(F.linear(torch.Tensor(output), weight=torch.Tensor(weights[i][:, 2:42].cpu())) + weights[i][:, 42].cpu().repeat(500, 1))
            output = F.linear(torch.Tensor(output), weight=torch.Tensor(weights[i][:, 43].cpu()).view(1, -1)) + weights[i][:, 44][0].cpu().repeat(500, 1)
            result.append(output.view(-1))
        return torch.stack(result)

def train_epoch(H, programs, optimizers, epoch, train_H):
    H.train()
    new_loss = []
    batch = range(dataset_size)
    init_inputs = np.zeros([dataset_size, 500])
    target = np.zeros([dataset_size, 500])
    # Generate a batch of dataset_size points (input into the neural network) for the single function in dataset
    # dataset was generated by a call of generate_sinusoid_batch(1)
    for ei, i in enumerate(batch):
        init_inputs[ei] = sorted(np.random.uniform(-5.0, 5.0, [500]))
        # Compute targets using phase and amplitude
        target[ei] = dataset[2][i] * np.sin(init_inputs[ei]-dataset[3][i])
    init_inputs = torch.Tensor(init_inputs)
    target = torch.Tensor(target)
    H.optimizer.zero_grad()
    for i in list(np.array(batch).astype(int)):
        optimizers[i].zero_grad()
    # Compute the ourput of the neural network specified by the program applied to the generate input_data
    # since we only have 1 program in batch, the program is just programs[0]
    output = H.forward(torch.stack([programs[i] for i in batch]), init_inputs).view(-1, 500)
    mse_loss = F.mse_loss(output, target)
    loss = mse_loss# + reg_loss
    loss.backward()
    # Plot output and ground truth
    if epoch % 1000 == 0 and epoch != 0:
        plt.plot(np.array(init_inputs[0]), np.array(target[0]))
        plt.plot(np.array(init_inputs[0]), np.array(output.cpu().detach().numpy()[0]))
        plt.show()
    # level of noise becomes lower as training goes on
    if train_H and not args.no_train_H:
        H.optimizer.step()
    if not args.no_train_p:
        for i in list(np.array(batch).astype(int)):
            optimizers[i].step()
    # Logging
    new_loss.append(loss.data.cpu().item())
    mean_loss = np.mean(new_loss)
    H.train_loss.append(mean_loss)
    plt.show()
    if epoch % 10 == 0:
        print("Epoch " + str(epoch) + ":")
        print("Mean Loss for this epoch: ", mean_loss)
    return

def train_net(H, programs, optimizers, epochs, train_H):
    '''Main function, logs loss, updates schedulers, and iterates train_epoch 'epochs' times'''
    try:
        for e in range(epochs):
            scheduler.step()
            for sch in schedulers:
                sch.step()
            if e != 0 and not args.no_train_H:
                writer.add_scalar('data/total_loss', H.train_loss[-1], e)
            # Update programs and the implementer with one step for the one function in 'programs'
            train_epoch(H, programs, optimizers, epoch=e, train_H=train_H)
            if e % 2000 == 0 and not args.no_train_H and e != 0:
                torch.save(H, output_folder + "/sine_model.pyt")
                torch.save(programs, output_folder + "/sine_train_programs.pyt")
                torch.save(dataset, output_folder + "/generated_dataset.pyt")
            if e % 10 == 0:
                post_program_mat = np.zeros((len(programs), args.p_dim))
                for i, p in enumerate(programs):
                    post_program_mat[i, :] = np.array(torch.tanh(p.cpu().detach()).numpy()).reshape((args.p_dim))
                    post_program_mat[i, :] = np.array(torch.tanh(p.cpu().detach()).numpy()).reshape((args.p_dim))
            # joblib.dump((pre_program_mat, post_program_mat), "/home/ubuntu/implementer_data/programs.mat")
                if e % 100 == 0 and not args.no_train_H:
                    writer.add_histogram("data/weight_distribution",
                                     torch.tanh(torch.stack(programs).clone()).cpu().data.numpy().ravel(), e)
    except KeyboardInterrupt:
        pass

    return

# Generate dataset of one function
dataset_size = 100
dataset = generate_sinusoid_batch(dataset_size)

# Generate implementer and assign an optimizer and learning rate scheduler for it
H = Hypernet(p_dim=args.p_dim, output_dim=1800).to(device)

H.optimizer = optim.RMSprop(H.parameters(), lr=args.H_lr)

scheduler = StepLR(H.optimizer, step_size=max(1000, args.epochs) // 2, gamma=0.1)

programs = []
optimizers = []
schedulers = []

# Generate programs and assign an optimizer and scheduler for each
for i in range(dataset_size):
    programs.append(nn.Parameter(Variable(torch.randn((args.p_dim, 1)).to(device), requires_grad=True).data))
    optimizer_temp = optim.RMSprop([programs[i]], lr=args.p_lr)
    optimizers.append(optimizer_temp)
    #schedulers.append(CyclicLR(optimizer_temp, base_lr=0.00005, max_lr=0.01, step_size=args.epochs // 100, mode='triangular2'))
    schedulers.append(StepLR(optimizer_temp, step_size=max(1000, args.epochs) // 2, gamma=0.1))

# Set random seed
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

# Run main function
#H = torch.load("Good progra")
train_net(H, programs, optimizers, args.epochs, True)

dataset_size = 10
dataset = generate_sinusoid_batch(dataset_size)

args.no_train_H = True


programs = []
optimizers = []
schedulers = []

# Generate programs and assign an optimizer and scheduler for each
for i in range(dataset_size):
    programs.append(nn.Parameter(Variable(torch.randn((args.p_dim, 1)).to(device), requires_grad=True).data))
    optimizer_temp = optim.RMSprop([programs[i]], lr=args.p_lr)
    optimizers.append(optimizer_temp)
    #schedulers.append(CyclicLR(optimizer_temp, base_lr=0.00005, max_lr=0.01, step_size=args.epochs // 100, mode='triangular2'))
    schedulers.append(StepLR(optimizer_temp, step_size=max(1000, args.epochs) // 2, gamma=0.1))

train_net(H, programs, optimizers, args.epochs, True)
