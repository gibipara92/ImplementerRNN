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
from cls import CyclicLR
from copy import deepcopy

writer = SummaryWriter()

parser = argparse.ArgumentParser(description='PyTorch MNIST RGAN')
# Choose dataset
parser.add_argument('--dataset', default='mnist', help='cifar10 | lsun | imagenet | folder | lfw', type=str)
# Specify size of images in dataset
parser.add_argument('--imsize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--epochs', type=int, default=100, help='the height / width of the input image to network')
parser.add_argument('--p_dim', type=int, default=20, help='the program size to network')
parser.add_argument('--path_img',   default='~/Downloads/implmenter_ims', help='Path', type=str)
parser.add_argument('--no_train_H', action='store_true', default=False, help='Do not train the HyperNet')
parser.add_argument('--no_train_p', action='store_true', default=False, help='Do not train the program')
parser.add_argument('--debug', action='store_true', default=False, help='Use dropout in E')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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
    params = []
    for i in range(7):
        for j in range(7):
            for inv in range(2):
                for blur in [0, 1, 2, 3]:
                    filter = np.zeros((7, 7))
                    for k in range(i - blur, i + blur + 1):
                        for l in range(j - blur, j + blur + 1):
                            if k < 0 or l < 0 or k > 6 or l > 6:
                                continue
                            else:
                                filter[k][l] = 1.
                    if inv == 1:
                        filter *= -1
                    features.append(torch.from_numpy(np.array(filter)).view(1, 7, 7).float())
                    params.append((i, j, inv, blur))
    return features, params, [features[i] for i in test_idx]

class Dataset(object):
    def __init__(self, batch_size):
        function_dataset = utils.TensorDataset(torch.Tensor(range(392)), torch.stack([torch.Tensor(i) for i in generate_translations_dataset()]))
        self.function_dataloader = utils.DataLoader(function_dataset, batch_size=batch_size, shuffle=True)
        self.dataset = datasets.MNIST(root='/home/ubuntu/PycharmProjects/IndependentMechanisms/mnist', train=True, download=True,
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
        self.dataloader_test = torch.utils.data.DataLoader(dataset_test, drop_last=False,
                                                 shuffle=False, num_workers=int(2))
        # self.pseudo_targets = Variable(torch.cat(self.compute_pseudo_targets()).view(10,1,args.imsize,args.imsize)).to(device)
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
        self.lstm_size = 128 # args.imsize ** 2
        sigma = 0.01
        self.lstm1 = torch.nn.LSTMCell(input_size=1, hidden_size=self.lstm_size)
        self.lstm2 = torch.nn.LSTMCell(input_size=self.lstm_size, hidden_size=self.lstm_size)
        self.fc1 = nn.Parameter(torch.randn(self.lstm_size, 1) * sigma)

        if dataloader is None:
            self.dataloader = Dataset(batch_size=64)
        else:
            self.dataloader = dataloader

    def implement_W(self, p, redundant_train=False):
        p = p.view(-1, p.shape[-2])
        if redundant_train:
            noise = torch.normal(mean=torch.zeros(p.shape), std=0.2)
            p += noise.cuda()
        p = torch.sigmoid(p)
        #p = p.view(-1, args.p_dim)
        batch = p.shape[0]
        outputs = []
        h_t1 = Variable(torch.zeros(batch, self.lstm_size).float(), requires_grad=False).to(device)
        c_t1 = Variable(torch.zeros(batch, self.lstm_size).float(), requires_grad=False).to(device)
        h_t2 = Variable(torch.zeros(batch, self.lstm_size).float(), requires_grad=False).to(device)
        c_t2 = Variable(torch.zeros(batch, self.lstm_size).float(), requires_grad=False).to(device)
        #h1 = Variable(torch.zeros(p.size(0), self.lstm_size).float(), requires_grad=False).to(device)

        for i, input_t in enumerate(p.chunk(p.shape[1], dim=1)):
            h_t1, c_t1 = self.lstm1(input_t, (h_t1, c_t1))
        c2 = 0
        # Counter increments until weight vector of generated filter is complete
        while c2 < self.output_dim:
            for i, input_t in enumerate(range(49)):
                h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
                outputs += [torch.mm(h_t2, self.fc1)]
                c2 += 1
                if c2 == self.output_dim:
                    break
            h2 = torch.stack(outputs, 1).squeeze()
        return h2.view(-1, 1, 7, 7)


    def forward(self, p, input_data):
        conv_mat = self.implement_W(p)
        # output = F.conv2d(input=input_data, weight=conv_mat, bias=None, padding=3)
        # output = F.linear(input=output, weight=weight2)
        return conv_mat
        #return output
        # return weight

def train_epoch(H, programs, optimizers, add_noise, epoch, dataloader, train_H, signature=False, task_id=''):
    H.train()
    new_loss = []
    for batch_idx, (idxs, target) in enumerate(dataloader):
        #digits, _ = next(iter(test_set))
        data = [programs[i] for i in list(np.array(idxs).astype(int))]
        data, target = torch.stack(data).to(device), target.to(device)
        H.optimizer.zero_grad()
        for i in list(np.array(idxs).astype(int)):
            optimizers[i].zero_grad()
        output = H.implement_W(data, redundant_train=add_noise)
        #output = H.forward(programs[idxs], data.view(-1, 1, args.imsize, args.imsize))
        loss = F.mse_loss(output, target)
        #print(loss)
        loss.backward()
        #ETA = .000
        #grad_clip = 5.0

        # level of noise becomes lower as training goes on
        if train_H and not args.no_train_H:
            H.optimizer.step()
        if not args.no_train_p:
            for i in list(np.array(idxs).astype(int)):
                #noise = sigma * Variable(torch.randn(programs[i].shape).cuda())
                #programs[i].grad *= len(programs)
                #programs[i].grad += noise
                #programs[i].grad = programs[i].grad.clamp(min=-grad_clip, max=grad_clip)
                optimizers[i].step()
        # Add timestep regularization after I figure it out
        new_loss.append(loss.data.cpu().item())
   #     for i, p in enumerate(programs):
   #         programs[i] = data[i]
    mean_loss = np.mean(new_loss)
    H.train_loss.append(mean_loss)
    print("Mean Loss for this epoch: ", mean_loss)
    #H.accuracy.append(100.0 * correct / ((batch_idx + 1) * args.batch_size))
    #print('* Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Train accuracy: {}/{} ({:.0f}%) \n'.format(
    #    epoch, len(dataloader), len(dataloader),
    #    100. * batch_idx / len(dataloader), mean_loss,
    #    correct, (batch_idx + 1) * args.batch_size, 100.0 * correct / ((batch_idx + 1) * args.batch_size)))
    return


def train_net(H, programs, optimizers, epochs, dataloader, train_H, task_id='', old_p=[], add_noise=True):
    try:
        for e in range(epochs):
            scheduler.step()
            for sch in schedulers:
                sch.step()
            if e != 0:
                writer.add_scalar('data/loss', H.train_loss[-1], e)
                writer.add_scalar('data/dims', args.p_dim, e)
                #writer.add_scalar('data/lr', H.optimizer.param_groups[0]['lr'], e)
            print("Epoch " + str(e) + ":")
         #   if not e % 1:
         #       if e > 0:

                #plot_all(p_progress, task_id=task_id)
                #conv_weight = H.implement_W(p)
                #save_image(1 - conv_weight.data.cpu().view(-1, 1, 7, 7),
                #           args.path_img + '/task%s_weights_epc%06d.png' % (task_id, e), nrow=10, normalize=True, scale_each=True)
            if e % 5000 == 0:
                torch.save(H, "/home/ubuntu/implementer_data/model" + str(e) + "_" + str(args.p_dim) + ".pyt")
                torch.save(programs, "/home/ubuntu/implementer_data/programs" + str(e) + "_" + str(args.p_dim) + ".pyt")

            train_epoch(H, programs, optimizers, add_noise, epoch=e, dataloader=dataloader, train_H=train_H, task_id=task_id)
            if e % 10 == 0:
                post_program_mat = np.zeros((len(programs), args.p_dim))
                for i, p in enumerate(programs):
                    post_program_mat[i, :] = np.array(torch.sigmoid(p.cpu().detach()).numpy()).reshape((args.p_dim))

            # joblib.dump((pre_program_mat, post_program_mat), "/home/ubuntu/implementer_data/programs.mat")
                im = plt.imshow(post_program_mat.T)
                ims.append([im])

    except KeyboardInterrupt:
        pass


        # stop if very good already
        #if H.accuracy[-1] > 99.0:
        #    print('# # # Good enough, stop training')
        #    break

    return

ims = []
batch_size = 64
total_dataset_size = 392
test_size = 14
train_size = total_dataset_size - test_size
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
train_idx = random.sample(range(total_dataset_size),total_dataset_size - test_size)
test_idx = list(set(range(total_dataset_size)).difference(set(train_idx)))
dataset = generate_translations_dataset()
#mnist_dataset = Dataset(4)
#imgs = mnist_dataset.dataloader_test


#for i , (imgs, _) in  enumerate(mnist_dataloader):
#    break
function_dataset = utils.TensorDataset(torch.Tensor(range(total_dataset_size)), torch.stack([torch.Tensor(i) for i in generate_translations_dataset()[0]]))
function_dataloader = utils.DataLoader(function_dataset, batch_size=total_dataset_size, shuffle=True, drop_last=True)

#dataset = []
#for im in imgs:
#    for ej, mat in enumerate(function_dataloader):
#        result = F.conv2d(input=im, weight=mat, bias=None, padding=3)
#        dataset.append((result, ej, target))


H = Hypernet(p_dim=args.p_dim, input_dim=args.imsize * args.imsize, output_dim=49, dataloader=function_dataloader).to(device)
#H = torch.load("//home/ubuntu/implementer_data/good_model_length_20.pyt")
#H = torch.load("/home/ubuntu/implementer_data/model10000.pyt")
H.optimizer = optim.RMSprop(H.parameters(), lr=0.001)

#scheduler = CyclicLR(H.optimizer, base_lr=0.000005, max_lr=0.001, step_size=args.epochs // 10, mode='triangular2')
scheduler = StepLR(H.optimizer, step_size=max(1000,args.epochs) // 3, gamma=0.1)

programs = []
optimizers = []
schedulers = []

rand_ints = random.sample(range(train_size), 20)

#programs = torch.load("/home/ubuntu/implementer_data/Great_programs.pyt")
for i in range(392):
    programs.append(nn.Parameter(Variable(torch.randn((args.p_dim, 1)).to(device), requires_grad=True).data))
    optimizer_temp = optim.RMSprop([programs[i]], lr=0.1)
    optimizers.append(optimizer_temp)
    #schedulers.append(CyclicLR(optimizer_temp, base_lr=0.00005, max_lr=0.01, step_size=args.epochs // 100, mode='triangular2'))
    schedulers.append(StepLR(optimizer_temp, step_size=max(1000, args.epochs) // 3, gamma=0.1))

#fig = plt.figure()


train_net(H, programs, optimizers, args.epochs, function_dataloader, True)


#ani = animation.ArtistAnimation(fig, ims, interval=25, blit=True, repeat_delay=0)

#ani.save('dynamic_images.mp4')

#plt.show()

plt.figure()
plt.semilogy(H.train_loss[1:])
plt.title("Loss over time")

plt.show()

features1, params1, test1 = generate_translations_dataset()



def group_by(programs, number_of_groups, index):
    mats = []
    count = np.zeros(number_of_groups).astype(int)
    size_of_group = train_size / number_of_groups
    for i in range(number_of_groups):
        mats.append(np.array([]))
    for i in range(train_size):
        for j in range(number_of_groups):
            if params1[i][index] == j:
                if count[j] == 0:
                    mats[j] =  np.array(torch.sigmoid(programs[i].cpu().detach()).numpy()).reshape((args.p_dim))
                else:
                    mats[j] = np.vstack([mats[j], np.array(torch.sigmoid(programs[i].cpu().detach()).numpy()).reshape((args.p_dim))])
                count[j] += 1
    return np.concatenate(mats)

Xtrans = group_by(programs, 7, 0)
Ytrans = group_by(programs, 7, 1)
inv = group_by(programs, 2, 2)
blurs = group_by(programs, 4, 3)

plt.figure(figsize=(100,100))
plt.subplot(4,1,1)
plt.imshow(blurs.T)
plt.gca().set_title("Blur")
plt.xticks(list(np.linspace(0, train_size, 5).astype(int)[:-1]))
plt.subplot(4,1,2)
plt.gca().set_title("Inverse")
plt.imshow(inv.T)
plt.xticks(list(np.linspace(0, train_size, 3).astype(int)[:-1]))
plt.subplot(4,1,3)
plt.gca().set_title("X Translation")
plt.imshow(Xtrans.T)
plt.xticks(list(np.linspace(0, train_size, 8).astype(int)[:-1]))
plt.subplot(4,1,4)
plt.gca().set_title("Y Translation")
plt.imshow(Ytrans.T)
plt.xticks(list(np.linspace(0, train_size, 8).astype(int)[:-1]))
plt.show()

rand_ints = random.sample(range(train_size), 20)

plt.figure()
for i in range(10):
    plt.subplot(2,10,i+1)
    plt.imshow(np.array(H.implement_W(programs[rand_ints[i]]).cpu().detach().numpy()).reshape(7, 7))
    plt.subplot(2, 10, 10 + i + 1)
    plt.imshow(np.array(dataset[0][rand_ints[i]]).reshape(7, 7))

plt.show()

plt.figure()

#for ij, j in enumerate(range(0, len(dataset[2]), 41)):
#    plt.subplot(10, 21, (21 * ij) + 1)
#    plt.imshow(np.array(H.implement_W(programs[j][:]).cpu().detach().numpy()).reshape(7, 7))
#    for i in range(20):
#        p = deepcopy(programs[j])
#        plt.subplot(10,21,(21 * ij) + i + 2)
#        p[i] += 10000.0
#        plt.imshow(np.array(H.implement_W(p).cpu().detach().numpy()).reshape(7, 7))
#plt.show()



#for ij, j in enumerate(range(0, len(dataset[2]), 41)):
#    plt.subplot(10, 20, (20 * ij) + 1)
#    plt.imshow(np.array(H.implement_W(programs[j][:]).cpu().detach().numpy()).reshape(7, 7))
#    for i in range(1,args.p_dim):
#        p = deepcopy(programs[j])
#        plt.subplot(10, args.p_dim, (args.p_dim * ij) + (i - 1) + 2)
#        p = p[:i]
#        plt.imshow(np.array(H.implement_W(p).cpu().detach().numpy()).reshape(7, 7))
#plt.show()

args.no_train_H = True
args.epochs = 5000

function_dataset = utils.TensorDataset(torch.Tensor(range(test_size * 50)), torch.stack([torch.Tensor(i) for i in  50 * generate_translations_dataset()[2]]))
function_dataloader = utils.DataLoader(function_dataset, batch_size=(test_size * 50), shuffle=True, drop_last=True)

programs = []
optimizers = []
schedulers = []

#H = torch.load("/home/ubuntu/implementer_data/model20000_20.pyt")


#programs = torch.load("/home/ubuntu/implementer_data/Great_programs.pyt")
for i in range(test_size * 50):
    programs.append(nn.Parameter(Variable(torch.randn((args.p_dim, 1)).to(device), requires_grad=False).data))
    optimizer_temp = optim.RMSprop([programs[i]], lr=0.1)
    optimizers.append(optimizer_temp)
    #schedulers.append(CyclicLR(optimizer_temp, base_lr=0.00005, max_lr=0.01, step_size=args.epochs // 100, mode='triangular2'))
    schedulers.append(StepLR(optimizer_temp, step_size=max(1000, args.epochs // 2), gamma=0.1))
#H = torch.load("/home/ubuntu/implementer_data/model10000_20.pyt")
train_net(H, programs, optimizers, args.epochs, function_dataloader, True, add_noise=False)

# plt.figure()
# post_program_mat = np.zeros((len(programs), args.p_dim))
# for i in range(14):
#     post_program_mat[14*i:(14*i)+14,:] = np.array(torch.sigmoid(torch.stack(programs[i::14]).cpu().detach()).numpy()).reshape((50,args.p_dim))
# plt.imshow(post_program_mat.T)
# plt.show()

plt.figure()
post_program_mat = []
for i in range(14):
    post_program_mat.append(torch.stack(programs[i::14]))
post_program_mat = torch.cat(post_program_mat).cpu().detach().numpy()
plt.imshow(post_program_mat.reshape(700, args.p_dim).T)
plt.show()


plt.figure()
for ei, i in enumerate(test_idx):
    plt.subplot(3,14,ei+1)
    plt.imshow(np.array(dataset[2][ei]).reshape(7, 7))
    min_loss = 1000000000
    min_arg = -1
    losses = []
    for j in range(50):
        output = H.implement_W(programs[14*j + ei], redundant_train=True)
        loss = F.mse_loss(output, torch.Tensor(generate_translations_dataset()[2][ei]).cuda().view(1, 1, 7, 7))
        if loss < min_loss:
            min_loss = loss
            min_arg = j
        losses.append(loss.item())
    plt.subplot(3, 14, 14 + ei + 1)
    plt.imshow(np.array(H.implement_W(programs[(14 * min_arg) + ei]).cpu().detach().numpy()).reshape(7, 7))
    plt.subplot(3, 14, 28 + ei + 1)
    plt.scatter(np.zeros(50), losses)
plt.show()
