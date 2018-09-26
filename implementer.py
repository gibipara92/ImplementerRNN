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
sns.set_style("whitegrid", {'grid.color': '.8', 'grid.linestyle': u':', 'legend.frameon': True})

# Define file arguments
parser = argparse.ArgumentParser(description='PyTorch MNIST RGAN')
# Choose dataset
parser.add_argument('--dataset', default='mnist', help='cifar10 | lsun | imagenet | folder | lfw', type=str)
# Specify size of images in dataset
parser.add_argument('--imsize', type=int, default=32, help='the height / width of the input image to network')
# specify number of digits to train on - remaining digits will be tested
parser.add_argument('--split_point', type=int, default=5, help='How many digits to train on')
parser.add_argument('--epochs', type=int, default=100, help='the height / width of the input image to network')
parser.add_argument('--one_wrong_example', action='store_true', default=False, help='Use dropout in E')
parser.add_argument('--task2_reverse_task1', action='store_true', default=False, help='Make task2 equal to task1 but invert the labels')
parser.add_argument('--repeat_task1', action='store_true', default=False, help='Repeat task 1 exactly')
parser.add_argument('--RNN', action='store_true', default=False, help='Use dropout in E')
parser.add_argument('--CNN', action='store_true', default=False, help='Use dropout in E')
parser.add_argument('--signed', action='store_true', default=False, help='Use dropout in E')
parser.add_argument('--train_on_signed_noise', action='store_true', default=False, help='Trained on signed noise')
parser.add_argument('--batch_norm', action='store_true', default=False, help='Use batchnorm')
parser.add_argument('--batch_norm', action='store_true', default=False, help='Use batchnorm')
parser.add_argument('--dropout', action='store_true', default=False, help='Use dropout')
parser.add_argument('--signature', action='store_true', default=False, help='Trained with signed digits')
parser.add_argument('--fc', action='store_true', default=False, help='Use fully connected network')
parser.add_argument('--debug', action='store_true', default=False, help='Use dropout in E')
parser.add_argument('--no_train_H', action='store_true', default=False, help='Do not train the HyperNet')
parser.add_argument('--no_train_p', action='store_true', default=False, help='Do not train the program')
parser.add_argument('--five_outputs', action='store_true', default=False, help='Do not train the program')
parser.add_argument('--batch_size', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--p_dim', type=int, default=10, help='the program size to network')
parser.add_argument('--repeat_experiment', default=1, help='How many times to repeat the exp', type=int)
parser.add_argument('--max_iterations', default=3000, help='Maximum number of iterations', type=int)
parser.add_argument('--path_img',   default='/is/cluster/gparascandolo/generative-models/GAN/R_out/hypernet/test_0to4/',
                    help='Path', type=str)
args = parser.parse_args()
args.cuda = True

if not os.path.exists(args.path_img + ''):
    os.makedirs(args.path_img + '')

class Dataset(object):
    def __init__(self, batch_size):
        dataset = datasets.MNIST(root='/is/cluster/gparascandolo/mnist/', train=True, download=True,
                            transform=transforms.Compose([
                                # transforms.Scale(args.imsize-4),
                                transforms.Pad(padding=2),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: (x * 2.0) - 1.0),
                            ]))
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=False,
                                                 shuffle=True, num_workers=int(2))
        dataset_test = datasets.MNIST(root='/is/cluster/gparascandolo/mnist/', train=False, download=True,
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
        self.split_0to4()

    def compute_pseudo_targets(self):
        targets = []
        for i in range(10):
            found = False
            for it, (X, target) in enumerate(self.dataloader, 0):
                for ind, t in enumerate(target):
                    if t == i:
                        targets.append(X[ind])
                        found = True
                        break
                if found:
                    break
        return targets

    def sample(self):
        for it, (X, target) in enumerate(self.dataloader, 0):
            return X

    def add_signature(self, data, target, random_sign=False, erase_digits=False, noise=False, other_digit=False):
        if other_digit:
            data = data[torch.randperm(data.size()[0])]
        if erase_digits:
            data *= 0.0
        if random_sign:
            t = np.random.randint(0,10,target.size()[0])
        else:
            t = target
        if noise:
            data *= 0.0
            data += torch.clamp(torch.randn(data.size()), -1, 1)
        for i, d in enumerate(data):
            # d[:,2,:2*(t[i]+1):2] = 0.5
            d[:,0,t[i]] = 0.5
        return data, target

    def load_all_dataset(self):
        self.all_X_or, self.all_targets = [], []
        print('Loading in memory the whole dataset')
        for it_epc, (X_or, target) in tqdm(enumerate(self.dataloader, 0)):
            self.all_targets.append(target)
            self.all_X_or.append(X_or)
        self.all_X_or = torch.cat(self.all_X_or)
        self.all_targets = torch.cat(self.all_targets)

    def split_0to4(self):
        self.data_0to4, self.targets_0to4 = [], []
        self.data_5to9, self.targets_5to9 = [], []

        for i in range(args.split_point):
            idxs = self.all_targets == i
            self.data_0to4.append(self.all_X_or[torch.nonzero(idxs).squeeze()])
            self.targets_0to4.append(self.all_targets[torch.nonzero(idxs).squeeze()])

        if args.task2_reverse_task1 or args.repeat_task1:
            for i in range(5):
                idxs = self.all_targets == i
                self.data_5to9.append(self.all_X_or[torch.nonzero(idxs).squeeze()])
                self.targets_5to9.append(torch.abs(self.all_targets[torch.nonzero(idxs).squeeze()] - 5*(1 - int(args.repeat_task1))))
        else:
            # for i in range(5):
            for i in range(args.split_point, 10):
                idxs = self.all_targets == i
                self.data_5to9.append(self.all_X_or[torch.nonzero(idxs).squeeze()])
                self.targets_5to9.append((self.all_targets[torch.nonzero(idxs).squeeze()]))

        self.data_0to4, self.targets_0to4 = torch.cat(self.data_0to4), torch.cat(self.targets_0to4)
        self.data_5to9, self.targets_5to9 = torch.cat(self.data_5to9), torch.cat(self.targets_5to9)

        if args.five_outputs:
            self.targets_5to9 -= 5

        self.dataset_0to4 = torch.utils.data.TensorDataset(self.data_0to4, self.targets_0to4)
        self.dataset_5to9 = torch.utils.data.TensorDataset(self.data_5to9, self.targets_5to9)
        self.dataloader_0to4 = torch.utils.data.DataLoader(self.dataset_0to4, batch_size=args.batch_size, shuffle=True)
        self.dataloader_5to9 = torch.utils.data.DataLoader(self.dataset_5to9, batch_size=args.batch_size, shuffle=True)

        self.all_X_or = None

    def split_each_digit(self):
        self.data_dict = {}
        for i in range(10):
            idxs = self.all_targets == i
            self.data_dict[i] = self.all_X_or[torch.nonzero(idxs).squeeze()]
        self.all_X_or = None

class Hypernet(nn.Module):
    def __init__(self, p_dim, input_dim, output_dim, dataloader=None):
        self.train_loss = []
        self.accuracy = []
        super(Hypernet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.p_dim = p_dim
        self.hid_dim = 64  # p_dim * 2
        sel.lstm_size = 256 # args.imsize ** 2
        sigma = 0.01

        if args.RNN:
            self.lstm = torch.nn.LSTMCell(input_size=1, hidden_size=self.lstm_size)
            self.fc1 = nn.Parameter(torch.randn(self.lstm_size, 1) * sigma)
        elif args.CNN:
            self.conv1d = nn.ConvTranspose2d(self.p_dim // 4, self.hid_dim, kernel_size=4, padding=1, stride=2)
            self.conv1d1 = nn.ConvTranspose2d(self.hid_dim, self.hid_dim, kernel_size=4, padding=1, stride=2)
            self.conv1d2 = nn.ConvTranspose2d(self.hid_dim, self.hid_dim, kernel_size=4, padding=1, stride=2)
            self.conv2d = nn.ConvTranspose2d(self.hid_dim, self.p_dim, kernel_size=4, padding=1, stride=2)
            self.fc1 = nn.Parameter(torch.randn(1, self.input_dim))
        else:
            self.fc1 = nn.Parameter(torch.randn(1, self.hid_dim) * sigma)
            # self.fc1_b = nn.Parameter(torch.ones(self.input_dim))
        # self.fc2 = nn.Parameter(torch.randn(self.hid_dim, self.p_dim) * sigma)
        # self.fc3 = nn.Parameter(torch.randn(self.input_dim, self.hid_dim) * sigma)
        # self.fc4 = nn.Parameter(torch.randn(self.output_dim, self.hid_dim) * sigma)
        # # self.fc2_b = nn.Parameter(torch.ones(self.output_dim,1))
        #
        # self.bn1 = nn.BatchNorm1d(self.hid_dim)
        # self.bn1d1 = nn.BatchNorm1d(self.hid_dim)
        # self.bn1d2 = nn.BatchNorm1d(self.hid_dim)
        # self.bn1d3 = nn.BatchNorm1d(self.hid_dim)
        # self.bn1d4 = nn.BatchNorm1d(self.hid_dim)



        if dataloader is None:
            self.dataloader = Dataset(batch_size=256)
        else:
            self.dataloader = dataloader

    def implement_W(self, p):
        if args.RNN:
            p = p.view(1, args.p_dim, 1)
            outputs = []
            h_t = Variable(torch.zeros(p.size(0), self.lstm_size).float(), requires_grad=False).cuda()
            c_t = Variable(torch.zeros(p.size(0), self.lstm_size).float(), requires_grad=False).cuda()

            for i, input_t in enumerate(p.chunk(p.size(1), dim=1)):
                h_t, c_t = self.lstm(input_t, (h_t, c_t))
                # outputs += [h_t]
            c = 0
            while c < self.output_dim * args.imsize * args.imsize:
                for i, input_t in enumerate(p.chunk(p.size(1), dim=1)):
                    h_t, c_t = self.lstm(input_t, (h_t, c_t))
                    outputs += [torch.mm(h_t, self.fc1)]
                    c+=1
                    if c == self.output_dim * args.imsize * args.imsize:
                        break
                h1 = torch.stack(outputs, 1).squeeze()
            return h1.view(self.input_dim, self.output_dim), None
        elif args.CNN:
            h1 = F.selu(self.conv1d(p.view(1, args.p_dim // 4, 2, 2)))
            # h1 = self.bn1(h1)
            h1a = F.selu(self.conv1d1(h1))
            # h1a = self.bn1(h1a)
            h1b = F.selu(self.conv1d2(h1a))
            # h1b = self.bn1(h1b)
            h2 = self.conv2d(h1b)
            return h2.view(self.hid_dim, self.input_dim), F.selu(torch.mm(p, self.fc1))
        else:
            h1 = F.selu(torch.mm(p, self.fc1))  # + self.fc1_b)
            # h1 = self.bn1(h1)
            h2 = F.selu(torch.mm(self.fc2, h1))  # + self.fc2_b)
            # h2 = self.bn2(h2)
            w1 = F.selu(torch.mm(self.fc3, h2))  # + self.fc3_b)
            w2 = F.selu(torch.mm(self.fc4, h2))  # + self.fc4_b)
            return w1, w2

    def forward(self, p, input_data):
        weight1, weight2 = self.implement_W(p)
        # output = F.selu(F.linear(input=input_data, weight=weight1))
        output = F.linear(input=input_data, weight=weight1.t())
        # output = F.linear(input=output, weight=weight2)
        return F.log_softmax(output, dim=1)
        # return weight


def init_net(net, extra_scaling=1.0):
    import math
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, extra_scaling * math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            try:
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            except:
                print('no weights in BN!')

def test_p_on_data(H, p, dataloader, signature=False):
    H.eval()
    correct = 0
    new_loss = []
    for batch_idx, (data, target) in enumerate(dataloader):
        if args.signed:
            data, target = H.dataloader.add_signature(data, target)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = H.forward(p, data.view(-1, args.imsize * args.imsize))
        loss = F.nll_loss(output, target)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        new_loss.append(loss.data.cpu()[0])
        if batch_idx > 100 and args.debug:
            break
    mean_loss = np.mean(new_loss)
    print('* Test: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Train accuracy: {}/{} ({:.0f}%) \n'.format(
        -1, len(dataloader), len(dataloader),
        100. * batch_idx / len(dataloader), mean_loss,
        correct, (batch_idx + 1) * args.batch_size, 100.0 * correct / ((batch_idx + 1) * args.batch_size)))
    H.train()
    return 100.0 * correct / ((batch_idx + 1) * args.batch_size)

def train_epoch(H, p, p_optimize, epoch, dataloader, train_H, signature=False, task_id=''):
    H.train()
    correct = 0
    new_loss = []

    for batch_idx, (data, target) in enumerate(dataloader):
        if args.signed:
            data, target = H.dataloader.add_signature(data, target)
        # if args.one_wrong_example and batch_idx == 100:
        #     data[0] = data[1] + 0.0
        # if args.signature:
        #     data, target = add_signature(data, target, noise=args.train_on_signed_noise)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        H.optimizer.zero_grad()
        p_optimize.zero_grad()
        output = H.forward(p, data.view(-1, args.imsize * args.imsize))
        loss = F.nll_loss(output, target)
        # loss = torch.mean((output - H.dataloader.pseudo_targets.view(10,-1))**2)
        if train_H and not args.no_train_H and epoch > 0:
            H.optimizer.step()
        if not args.no_train_p and epoch > 0:
            p_optimize.step()
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        # correct += 0
        new_loss.append(loss.data.cpu()[0])
        # TODO loss through p as well!
        # new_grad.append(compute_grad_norm(self))
        # if 1:
        #     norm_relevant_pixels.append(torch.mean(torch.abs(self.fc2.weight.data[0,:10])))
        #     norm_other_pixels.append(torch.mean(torch.abs(self.fc2.weight.data[1:,:])))
        if batch_idx > 25 and args.debug:
            break
        if batch_idx % 200 == 0:
            print('...')
        if epoch == 0:
            print('Skip 0th epoch')
            break
    mean_loss = np.mean(new_loss)
    H.train_loss.append(mean_loss)
    H.accuracy.append(100.0 * correct / ((batch_idx + 1) * args.batch_size))
    print('* Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Train accuracy: {}/{} ({:.0f}%) \n'.format(
        epoch, len(dataloader), len(dataloader),
        100. * batch_idx / len(dataloader), mean_loss,
        correct, (batch_idx + 1) * args.batch_size, 100.0 * correct / ((batch_idx + 1) * args.batch_size)))
    if epoch == 0:
        save_image(1 - data.data.cpu(), args.path_img + '/data_sample.png')
    # self.train_norm_grad.append(np.mean(new_grad))
    # if args.reconstructor:
    #     self.train_loss_reconstructor.append(np.mean(new_loss_reconstructor))
    # self.norm_relevant_pixels.append(np.mean(norm_relevant_pixels))
    # self.norm_other_pixels.append(np.mean(norm_other_pixels))
    return

def train_net(H, p, p_optimize, epochs, dataloader, train_H, task_id='', old_p=[]):
    p_progress = old_p + [p.data.cpu().numpy()]
    for e in range(epochs):
        if not e % 1:
            if e > 0:
                plot_all(p_progress, task_id=task_id)
            weight = H.implement_W(p)[0]
            save_image(1 - weight.data.cpu().view(-1, 1, args.imsize, args.imsize),
                       args.path_img + '/task%s_weights_epc%06d.png' % (task_id, e), nrow=10, normalize=True, scale_each=True)
        train_epoch(H, p, p_optimize, epoch=e, dataloader=dataloader, train_H=train_H, task_id=task_id)
        p_progress.append(p.data.cpu().numpy())

        # stop if very good already
        if H.accuracy[-1] > 99.0:
            print('# # # Good enough, stop training')
            break

    return p_progress

def train_0to4_then_5to9(H, p1, p2, p_optimize1, p_optimize2, epochs):
    print('# # # # # Training 0 to 4 # # # # # ')
    p1_progress = train_net(H, p1, p_optimize1, epochs, H.dataloader.dataloader_0to4, task_id='0to4', train_H=True)
    test_mixed(H, p1, p2)
    # e=deepcopy(list(H.parameters()))
    print('# # # # # Training 5 to 9 # # # # # ')
    p2_progress = train_net(H, p2, p_optimize2, epochs, H.dataloader.dataloader_5to9, task_id='5to9', train_H=False, old_p=p1_progress)
    test_mixed(H, p1, p2)

def test_mixed(H, p1, p2):
    tasks_programs_accuracies = np.zeros((2,2))
    tasks_programs_accuracies[0,0] = test_p_on_data(H, p1, H.dataloader.dataloader_0to4)
    tasks_programs_accuracies[0,1] = test_p_on_data(H, p1, H.dataloader.dataloader_5to9)
    tasks_programs_accuracies[1,0] = test_p_on_data(H, p2, H.dataloader.dataloader_0to4)
    tasks_programs_accuracies[1,1] = test_p_on_data(H, p2, H.dataloader.dataloader_5to9)
    print('Accuracies for different programs')
    print(tasks_programs_accuracies)

    print('Accuracy on task1 with zeros as a program:', test_p_on_data(H, p1*0.0, H.dataloader.dataloader_0to4))

def plot_all(p_progress=None, task_id=''):
    plt.figure(figsize=(8, 5))
    plt.subplot(2, 1, 1)
    plt.semilogy(H.train_loss, label='train loss', linewidth=2.0)
    plt.legend()
    plt.ylabel('Training loss')
    plt.subplot(2, 1, 2)
    plt.plot(H.accuracy, label='accuracy', linewidth=2.0)
    plt.legend()
    plt.ylabel('Training accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(args.path_img + '/plot.png')
    plt.close()
    if p_progress is not None:
        imsave(args.path_img + '/p'+str(task_id)+'_progression.png', np.hstack(p_progress))

# numbr of output units
output_dim = 10

if args.five_outputs:
    output_dim = 5

H = Hypernet(p_dim=args.p_dim, input_dim=args.imsize * args.imsize, output_dim=output_dim).cuda()
p1 = Variable(torch.randn((args.p_dim, 1)).cuda(), requires_grad=True)
p2 = Variable(torch.randn((args.p_dim, 1)).cuda(), requires_grad=True)
p1 = nn.Parameter(p1.data)
p2 = nn.Parameter(p2.data)

H.optimizer = optim.Adam(H.parameters(), lr=0.001)
p_optimize1 = optim.Adam([p1], weight_decay=0.0, lr=0.01)
p_optimize2 = optim.Adam([p2], weight_decay=0.0, lr=0.01)
adap_optimizer2 = None
# adap_optimizer2 = torch.optim.lr_scheduler.ReduceLROnPlateau(p_optimize2, 'min', factor=0.5, patience=1000,
#                                                                  verbose=True, min_lr=1e-05, threshold=0.0001,
#                                                                  threshold_mode='rel')

# H.optimizer = optim.SGD(H.parameters(), lr=0.005, momentum=0.9, nesterov=True,)
# p_optimize = optim.SGD([p], weight_decay=0.0, lr=0.005, momentum=0.9, nesterov=True,)

train_0to4_then_5to9(H, p1, p2, p_optimize1, p_optimize2, args.epochs)

embed()

# def implement_W(self, p):
#     p = p.view(1, args.p_dim, 1)
#     outputs = []
#     h_t = Variable(torch.zeros(p.size(0), self.lstm_size).float(), requires_grad=False).cuda()
#     c_t = Variable(torch.zeros(p.size(0), self.lstm_size).float(), requires_grad=False).cuda()
#     for i, input_t in enumerate(p.chunk(p.size(1), dim=1)):
#         h_t, c_t = self.lstm(input_t, (h_t, c_t))
#         outputs += [h_t]
#     for j in range(args.imsize ** 2 - args.p_dim):
#         h_t, c_t = self.lstm(input_t * 0.0, (h_t, c_t))
#         outputs += [h_t]
#     h1 = torch.stack(outputs, 1).squeeze()
#     return h1
#
# #
