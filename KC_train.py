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
#from hypernet import Hypernet
from hyperLSTM_KC import Hypernet

#from multiplication_implementer_genetic import main2
parser = argparse.ArgumentParser(description='PyTorch MNIST RGAN')
# Choose dataset
parser.add_argument('--dataset', default='/home/ubuntu/PycharmProjects/IndependentMechanisms/mnist',
                    help='path to dataset', type=str)
parser.add_argument('--meta_folder', default='/home/ubuntu/implementer_data/meta/',
                    help='path to save models/programs', type=str)
# Specify size of images in dataset
parser.add_argument('--imsize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs during training')
parser.add_argument('--p_dim', type=int, default=2, help='the program size to network')
parser.add_argument('--lstm_size', type=int, default=256, help='Size of LSTM layers')
parser.add_argument('--mnist_size', type=int, default=100, help='Number of examples in test set')
parser.add_argument('--omniglot_size', type=int, default=100, help='Number of examples in test set')
parser.add_argument('--test_size', type=int, default=100, help='Number of examples in test set')
parser.add_argument('--batch_size', type=int, default=25000, help='Number of examples in test set')
parser.add_argument('--fc_size', type=int, default=256, help='Number of examples in test set')
parser.add_argument('--reg_lambda', type=float, default=0.001, help='Coefficient of regularization term')
parser.add_argument('--noise', type=float, default=0.2, help='Amount of noise to add to programs')
parser.add_argument('--H_lr', type=float, default=0.1, help='Learning rate for implementer')
parser.add_argument('--p_lr', type=float, default=0.1, help='Learning rate for programs')
parser.add_argument('--path_img',   default='~/Downloads/implmenter_ims', help='Path', type=str)
parser.add_argument('--activation_function', default='tanh', help='tanh|sin|LeakyReLU', type=str)
parser.add_argument('--no_train_H', action='store_true', default=False, help='Do not train the HyperNet')
parser.add_argument('--no_train_p', action='store_true', default=False, help='Do not train the program')

args = parser.parse_args()

#args.H_lr *= args.omniglot_size

args2 = deepcopy(args)
del args2.meta_folder
del args2.path_img
del args2.dataset
del args2.no_train_H
del args2.no_train_p

#args.batch_size = args.omniglot_size * args.test_size

now = datetime.now()
args2.time = now.strftime("%Y%m%d-%H%M%S")

if args.activation_function == 'tanh':
    args.activation_function = torch.tanh
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

print(args)
print(device)

ims = []
batch_size = 64
total_dataset_size = 392
test_size = 14
conv_mat_size = 7
train_size = total_dataset_size - test_size
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
train_idx = random.sample(range(total_dataset_size),total_dataset_size - test_size)
test_idx = list(set(range(total_dataset_size)).difference(set(train_idx)))


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
    test_features = []
    params = []
    counter = 0
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
                                filter[k][l] = 1. / ((blur + 1) * (blur + 1))
                    if inv == 1:
                        filter *= -1
                    if counter in train_idx:
                        features.append(torch.from_numpy(np.array(filter)).view(1, conv_mat_size, conv_mat_size).float())
                    else:
                        features.append(np.zeros((1, conv_mat_size, conv_mat_size)))
                        test_features.append(
                            torch.from_numpy(np.array(filter)).view(1, conv_mat_size, conv_mat_size).float())
                    counter += 1
                    params.append((i, j, inv, blur))
    return features, params, test_features

features1, params1, test1 = generate_translations_dataset()

class Dataset(object):
    def __init__(self, batch_size):
        function_dataset = utils.TensorDataset(torch.Tensor(range(392)), torch.stack([torch.Tensor(i) for i in generate_translations_dataset()]))
        self.function_dataloader = utils.DataLoader(function_dataset, batch_size=batch_size, shuffle=True)
        self.dataset = datasets.MNIST(root=args.dataset, train=True, download=True,
                                 transform=transforms.Compose([
                                     # transforms.Scale(args.imsize-4),
                                     transforms.Pad(padding=2),
                                     transforms.ToTensor(),
                                     transforms.Lambda(lambda x: (x * 2.0) - 1.0),
                                 ]))
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=False,
                                                      shuffle=True, num_workers=int(2))
        dataset_test = datasets.MNIST(root=args.dataset, train=False, download=True,
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

def length_regularization(programs, idxs):
    penalty = 0
    losses = args.p_dim - torch.Tensor(range(args.p_dim)).cuda()
    for program in [programs[i] for i in np.array(idxs).astype('int')]:
        program = program.view(-1)
        mask = torch.abs(program) > 0.001
        program_masked = torch.mul(mask.float(), program.float())
        penalty += torch.dot(torch.abs(program_masked), losses)
    return min(penalty / len(idxs), 10000)


def train_epoch(H, mnist_data, epoch, dataloader, train_H, signature=False, task_id=''):
    H.train()
    new_loss = []
    new_mse_losses = []
    new_reg_losses = []

    idxs = range(len(programs))
    target = torch.stack([dataset[i] for i in idxs])
    H.optimizer.zero_grad()
    for i in idxs:
        optimizers[int(i)].zero_grad()

        #output                 = H.implement_W(data, redundant_train=add_noise)
    input_data_mat = target.view(-1, 1, 16, 16)
    weights = H.forward(input_data_mat, torch.stack([programs[int(i)] for i in idxs]), epoch)
        #weights = H.forward(input_data_mat, torch.stack([programs[int(i)] for i in idxs]))
        #loss = F.mse_loss(result, H.target_mat)
    loss = F.mse_loss(weights, target)# + (length_regularization(programs, idxs) * (min(epoch, 1500) * 0.000000001))
    loss.backward()
        #print(programs[0].grad)
        #test_input_data_mat = mnist_test_data.view(-1, 1, 16, 16)
        #test_target_mat = []
        #for i in test_input_data_mat:
        #    test_target_mat.append(args.activation_function(torch.matmul(i.view(16, 16), target.view(16, 16))))
        #test_target_mat = torch.stack(test_target_mat).view(-1, 1, 16, 16)
        #test_output = args.activation_function(H.forward(test_input_data_mat))
        #test_loss = F.mse_loss(test_output, test_target_mat)
        #ETA = .000
        #grad_clip = 5.0
    if train_H and not args.no_train_H:
        H.optimizer.step()
        # Add timestep regularization after I figure it out
    new_loss.append(loss.data.cpu().item())
    if not args.no_train_p:
        for i in idxs:
            optimizers[int(i)].step()
    display_idxs = list(range(100)) + list(range(500,510)) + list(range(1000,1010))
    if epoch % 40000 == 0:
        plt.figure()
        plt.subplot(4, 4, 1)
        plt.imshow(target[1001][0].cpu().view(16,16))
        plt.subplot(4, 4, 2)
        plt.imshow(target[204][0].cpu().view(16, 16))
        min_loss = 100000
        #for i in range(len(programs)):
        #    temp = F.mse_loss(result[i], H.target_mat[0])
        #    if temp < min_loss:
        #        min_loss = temp
        #        min_i = i
        plt.subplot(4, 4, 3)
        plt.imshow(weights[1001][0].view(16, 16).cpu().detach().numpy())
        plt.subplot(4, 4, 4)
        plt.imshow(weights[204][0].view(16, 16).cpu().detach().numpy())
        plt.subplot(2,1,2)
        plt.imshow(torch.sigmoid(torch.stack([programs[i] for i in display_idxs]).view(-1, args.p_dim)).cpu().detach().numpy().T)
        plt.show()

    mean_loss = np.mean(new_loss)
    H.train_loss.append(mean_loss)
    if epoch % 10 == 0:
        print("Epoch " + str(epoch) + ":")
        print("Mean Loss for this epoch: ", loss.data.cpu().item())
        print("Mean Loss for this epoch: ", loss.data.cpu().item())
    #    print("Min Loss for this epoch:  ", min_loss.data.cpu().item())
        #print("Mean Test Loss for this epoch: ", test_loss.data.cpu().item())
    #H.accuracy.append(100.0 * correct / ((batch_idx + 1) * args.batch_size))
    #print('* Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Train accuracy: {}/{} ({:.0f}%) \n'.format(
    #    epoch, len(dataloader), len(dataloader),
    #    100. * batch_idx / len(dataloader), mean_loss,
    #    correct, (batch_idx + 1) * args.batch_size, 100.0 * correct / ((batch_idx + 1) * args.batch_size)))
    return

def train_net(H, epochs, dataloader, train_H, mnist_data, task_id='', old_p=[], add_noise=True, train_e=False):
    rand_ints = random.sample(range(train_size), 20)
    try:
        for e in range(epochs):
            scheduler.step()
            if e % 1000 == 0 and e != 0:
                model_file = "/home/ubuntu/implementer_data/DirectKCModel" + str(e) + "_" + str(
                    args.p_dim) + ".pyt"
                torch.save(H, model_file)
                program_file = "/home/ubuntu/implementer_data/DirectKCPrograms" + str(e) + "_" + str(args.p_dim) + ".pyt"
                torch.save(programs, program_file)
                #main2(args, writer, load_str=model_file, e=e)
            if e != 0:
                writer.add_scalar('data/total_loss', H.train_loss[-1], e)
                #writer.add_scalar('data/lr', H.optimizer.param_groups[0]['lr'], e)
         #   if not e % 1:
         #       if e > 0:

                #plot_all(p_progress, task_id=task_id)
                #conv_weight = H.implement_W(p)
                #save_image(1 - conv_weight.data.cpu().view(-1, 1, 7, 7),
                #           args.path_img + '/task%s_weights_epc%06d.png' % (task_id, e), nrow=10, normalize=True, scale_each=True)
            train_epoch(H, mnist_data, epoch=e, dataloader=dataloader, train_H=train_H, task_id=task_id)

    except KeyboardInterrupt:
        pass


        # stop if very good already
        #if H.accuracy[-1] > 99.0:
        #    print('# # # Good enough, stop training')
        #    break

    return

dataset = datasets.MNIST(root=args.path_img, train=True, download=True,
                                 transform=transforms.Compose([
                                     # transforms.Scale(args.imsize-4),
                                     transforms.Pad(padding=2),
                                     transforms.ToTensor(),
                                     #transforms.Lambda(lambda x: (x * 2.0) - 1.0),
                                 ]))
tdataloader = torch.utils.data.DataLoader(dataset, batch_size=1, drop_last=False, shuffle=False, num_workers=int(2))
# zeros = []
# ones = []
# twos = []
# threes = []
# for batch_idx, (mnist_data, mnist_target) in enumerate(tdataloader):
#     mnist_data = mnist_data.cuda()
#     mnist_data = F.upsample(mnist_data, size=(16, 16), mode='bilinear')
#     if mnist_target[0].item() == 0:
#         zeros.append(mnist_data)
#     elif mnist_target[0].item() == 1:
#         ones.append(mnist_data)
#     elif mnist_target[0].item() == 2:
#         twos.append(mnist_data)
#     elif mnist_target[0].item() == 3:
#         threes.append(mnist_data)
# zeros = torch.stack(zeros)
# ones = torch.stack(ones)
# twos = torch.stack(twos)
# threes = torch.stack(threes)

#   joblib.dump([zeros, ones, twos, threes], 'mnist_0123.npy')
zeros, ones, twos, threes = joblib.load('mnist_0123.npy')
#torch.random(())
noise = torch.FloatTensor(500, 1, 1, 16, 16).uniform_().cuda()
dataset = torch.cat([zeros[:500], ones[:500], twos[:500], threes[:500], noise])[:,0,:,:,:]

#tdataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, drop_last=False, shuffle=False, num_workers=int(2))
#dataset = datasets.ImageFolder(root='/home/ubuntu/PycharmProjects/IndependentMechanisms/images_background',
#                         transform=transforms.Compose([
#                             transforms.Scale(args.imsize - 4),
#                             transforms.ToTensor(),
#                             transforms.Lambda(lambda x: 1. - x),
#                             transforms.ToPILImage(),
#                             transforms.Pad(padding=2, fill=0),
#                             transforms.ToTensor(),
#                             transforms.Lambda(lambda x: torch.mean(x, dim=0).view(1, 16, 16))]))

H = Hypernet(p_dim=args.p_dim, input_dim=args.imsize * args.imsize, output_dim=256, dataloader=tdataloader).to(device)

#H.optimizer = optim.SGD(H.parameters(), lr=args.H_lr)
#H = torch.load("/home/ubuntu/implementer_data/DirectKCModel1000_30.pyt"  )
H.optimizer = optim.SGD(H.parameters(), lr=1)
H.precomputed = False
scheduler = StepLR(H.optimizer, step_size=max(1000, args.epochs) // 5, gamma=0.1)

programs = []
optimizers = []
schedulers = []

rand_ints = random.sample(range(train_size), 20)
#programs = torch.load("/home/ubuntu/implementer_data/DirectKCPrograms1000_30.pyt")
#for i, program in enumerate(programs):
#    noise = torch.normal(mean=torch.zeros(programs[i].shape), std=5.)
#    programs[i].data *= 0.0
#    programs[i].data += noise.cuda()

#programs = torch.load("/home/ubuntu/implementer_data/Great_programs.pyt")
for i in range(2500):
    programs.append(nn.Parameter(Variable(torch.randn((args.p_dim, 1)).to(device), requires_grad=True).data))
    optimizer_temp = optim.RMSprop([programs[i]], lr=args.p_lr)
    optimizers.append(optimizer_temp)
    schedulers.append(StepLR(optimizer_temp, step_size=max(1000, args.epochs) // 3, gamma=0.1))



train_net(H, args.epochs, tdataloader, True, dataset, train_e=False)

