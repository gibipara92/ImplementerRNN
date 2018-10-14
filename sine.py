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

print(args)
print(device)

ims = []
batch_size = 64
total_dataset_size = 392
test_size = 0
conv_mat_size = 7
train_size = total_dataset_size - test_size
random.seed(4)
torch.manual_seed(4)
torch.cuda.manual_seed_all(4)
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
    params_dict = {}
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
                                filter[k][l] = 1.
                    if inv == 1:
                        filter *= -1
                    if counter in train_idx:
                        features.append(torch.from_numpy(np.array(filter)).view(1, conv_mat_size, conv_mat_size).float())
                        params_dict[(i, j, inv, blur)] = counter
                    else:
                        features.append(np.zeros((1, conv_mat_size, conv_mat_size)))
                        test_features.append(
                            torch.from_numpy(np.array(filter)).view(1, conv_mat_size, conv_mat_size).float())
                    counter += 1
                    params.append((i, j, inv, blur))

    return features, params, test_features, params_dict

def generate_sinusoid_batch(batch_size, input_idx=None):
        # Note train arg is not used (but it is used for omniglot method.
        # input_idx is used during qualitative testing --the number of examples used for the grad update
    amp = np.random.uniform(0.1, 5.0, [batch_size])
    phase = np.random.uniform(0, np.pi, [batch_size])
    outputs = np.zeros([batch_size, 1000])
    init_inputs = np.zeros([batch_size, 1000])
    for func in range(batch_size):
        init_inputs[func] = np.random.uniform(-5.0, 5.0, [1000])
        outputs[func] = amp[func] * np.sin(init_inputs[func]-phase[func])
    return init_inputs, outputs, amp, phase


features1, params1, test1, params_dict = generate_translations_dataset()

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

class Hypernet(nn.Module):
    def __init__(self, p_dim, input_dim, output_dim, dataloader=None):
        self.train_loss = []
        self.mse_loss = []
        self.reg_loss = []
        self.accuracy = []
        super(Hypernet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.p_dim = p_dim
        self.lstm_size = args.lstm_size # args.imsize ** 2
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

        for i, input_t in enumerate(p.chunk(p.shape[1], dim=1)):
            h_t1, c_t1 = self.lstm1(input_t, (h_t1, c_t1))
        c2 = 0
        # Counter increments until weight vector of generated filter is complete
        while c2 < self.output_dim:
            for i, input_t in enumerate(range(self.output_dim)):
                h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
                outputs += [torch.mm(h_t2, self.fc1)]
                c2 += 1
                if c2 == self.output_dim:
                    break
            h2 = torch.stack(outputs, 1).squeeze()
        return h2.view(-1, 40, 2)

    def forward(self, p, input_data):
        weights = self.implement_W(torch.stack(p), redundant_train=True)
        result = torch.zeros((256, 1000))
        for i in range(256):
            output = F.linear(torch.Tensor(torch.Tensor(input_data[i]).view(-1, 1)), weight=torch.Tensor(weights[i][:, 0].cpu()).view(-1,1))
            output = F.linear(torch.Tensor(output), weight=torch.Tensor(weights[i][:, 1].cpu()).view(1,-1)).view(1000)
            result[i] = output
        return result

def train_epoch(H, programs, optimizers, epoch, train_H):
    H.train()
    new_loss = []
    batch = random.sample(range(2048), 256)
    init_inputs = np.zeros([256, 1000])
    target = np.zeros([256, 1000])
    for ei, i in enumerate(batch):
        init_inputs[ei] = sorted(np.random.uniform(-5.0, 5.0, [1000]))
        target[ei] = dataset[2][i] * np.sin(init_inputs[ei]-dataset[3][i])
    init_inputs = torch.Tensor(init_inputs)
    target = torch.Tensor(target)
    H.optimizer.zero_grad()
    for i in list(np.array(batch).astype(int)):
        optimizers[i].zero_grad()
    output = H.forward([programs[i] for i in batch], init_inputs)
    #target = torch.Tensor([dataset[1][i] for i in batch])
    mse_loss = F.mse_loss(output, target)
    #reg_loss = length_regularization(programs, idxs) * args.reg_lambda
    loss = mse_loss# + reg_loss
    loss.backward()
    if epoch % 1000 == 0:
        plt.plot(np.array(init_inputs[0]), np.array(target[0]))
        plt.plot(np.array(init_inputs[0]), np.array(output.cpu().detach().numpy()[0]))
        plt.show()
    #ETA = .000
    #grad_clip = 5.0

    # level of noise becomes lower as training goes on
    if train_H and not args.no_train_H:
        H.optimizer.step()
    if not args.no_train_p:
        for i in list(np.array(batch).astype(int)):
            #noise = sigma * Variable(torch.randn(programs[i].shape).cuda())
            #programs[i].grad *= len(programs)
            #programs[i].grad += noise
            #programs[i].grad = programs[i].grad.clamp(min=-grad_clip, max=grad_clip)
            optimizers[i].step()
    # Add timestep regularization after I figure it out
    new_loss.append(loss.data.cpu().item())
    #new_mse_losses.append(mse_loss.data.cpu().item())
    #new_reg_losses.append(reg_loss.data.cpu().item())
#     for i, p in enumerate(programs):
   #         programs[i] = data[i]
    mean_loss = np.mean(new_loss)
    #mean_mse_loss = np.mean(new_mse_losses)
  #  mean_reg_loss = np.mean(new_reg_losses)
    H.train_loss.append(mean_loss)
    #H.mse_loss.append(mean_mse_loss)
    #H.reg_loss.append(mean_reg_loss)
    plt.show()
    if epoch % 10 == 0:
        print("Epoch " + str(epoch) + ":")
        print("Mean Loss for this epoch: ", mean_loss)
        #print("Mean MSE Loss for this epoch: ", mean_mse_loss)
        #print("Mean Reg Loss for this epoch: ", mean_reg_loss)
    #H.accuracy.append(100.0 * correct / ((batch_idx + 1) * args.batch_size))
    #print('* Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Train accuracy: {}/{} ({:.0f}%) \n'.format(
    #    epoch, len(dataloader), len(dataloader),
    #    100. * batch_idx / len(dataloader), mean_loss,
    #    correct, (batch_idx + 1) * args.batch_size, 100.0 * correct / ((batch_idx + 1) * args.batch_size)))
    return

def train_net(H, programs, optimizers, epochs, dataloader, train_H, task_id='', old_p=[], add_noise=True, train_e=False):
    rand_ints = random.sample(range(train_size), 20)
    try:
        for e in range(epochs):
            scheduler.step()
            for sch in schedulers:
                sch.step()
            if e != 0 and not args.no_train_H:
                writer.add_scalar('data/total_loss', H.train_loss[-1], e)
                #writer.add_scalar('data/mse_loss', H.mse_loss[-1], e)
                #writer.add_scalar('data/reg_loss', H.reg_loss[-1], e)
                #writer.add_scalar('data/dims', args.p_dim, e)
            #if e != 0 and args.no_train_H:
                #writer.add_scalar('data/test_total_loss' + "_" + str(train_e), H.train_loss[-1], e)
                #writer.add_scalar('data/test_mse_loss' + "_" + str(train_e), H.mse_loss[-1], e)
                #writer.add_scalar('data/test_reg_loss' + "_" + str(train_e), H.reg_loss[-1], e)
                #writer.add_scalar('data/lr', H.optimizer.param_groups[0]['lr'], e)
         #   if not e % 1:
         #       if e > 0:

                #plot_all(p_progress, task_id=task_id)
                #conv_weight = H.implement_W(p)
                #save_image(1 - conv_weight.data.cpu().view(-1, 1, 7, 7),
                #           args.path_img + '/task%s_weights_epc%06d.png' % (task_id, e), nrow=10, normalize=True, scale_each=True)
            train_epoch(H, programs, optimizers, epoch=e, train_H=train_H)
            if e % 10 == 0:
                post_program_mat = np.zeros((len(programs), args.p_dim))
                for i, p in enumerate(programs):
                    post_program_mat[i, :] = np.array(torch.tanh(p.cpu().detach()).numpy()).reshape((args.p_dim))
                    post_program_mat[i, :] = np.array(torch.tanh(p.cpu().detach()).numpy()).reshape((args.p_dim))

            # joblib.dump((pre_program_mat, post_program_mat), "/home/ubuntu/implementer_data/programs.mat")
                if e % 100 == 0 and not args.no_train_H:
                    writer.add_histogram("data/weight_distribution",
                                     torch.tanh(torch.stack(programs).clone()).cpu().data.numpy().ravel(), e)
                    #generate_group_fig(programs, e)
                    #save_implementations(programs, rand_ints, e)





    except KeyboardInterrupt:
        pass


        # stop if very good already
        #if H.accuracy[-1] > 99.0:
        #    print('# # # Good enough, stop training')
        #    break

    return

def triangular_number(n):
    return n * (n + 1) // 2

loss_penalties = torch.Tensor(np.array([triangular_number(args.p_dim - 1 - i) for i in range(args.p_dim)])).cuda()

def length_regularization(programs, idxs):
    penalty = 0
    for program in [programs[i] for i in np.array(idxs).astype('int')]:
        program = program.view(-1)
        mask = torch.abs(program) < 0.01
        program_masked = torch.mul(mask.float(), program.float())
        penalty += torch.dot(torch.abs(program), loss_penalties)
    return penalty / len(idxs)


#mnist_dataset = Dataset(4)
#imgs = mnist_dataset.dataloader_test


#for i , (imgs, _) in  enumerate(mnist_dataloader):
#    break
dataset = generate_sinusoid_batch(2048)
function_dataset = utils.TensorDataset(torch.Tensor(range(total_dataset_size)), torch.stack([torch.Tensor(i) for i in generate_translations_dataset()[0]]))
function_dataloader = utils.DataLoader(function_dataset, batch_size=64, shuffle=True, drop_last=True)

#dataset = []
#for im in imgs:
#    for ej, mat in enumerate(function_dataloader):
#        result = F.conv2d(input=im, weight=mat, bias=None, padding=3)
#        dataset.append((result, ej, target))


H = Hypernet(p_dim=args.p_dim, input_dim=args.imsize * args.imsize, output_dim=80, dataloader=function_dataloader).to(device)
#H = torch.load("//home/ubuntu/implementer_data/good_model_length_20.pyt")
#H = torch.load("/home/ubuntu/implementer_data/model10000.pyt")
H.optimizer = optim.RMSprop(H.parameters(), lr=args.H_lr)

#scheduler = CyclicLR(H.optimizer, base_lr=0.000005, max_lr=0.001, step_size=args.epochs // 10, mode='triangular2')
scheduler = StepLR(H.optimizer, step_size=max(1000,args.epochs) // 3, gamma=0.1)

programs = []
optimizers = []
schedulers = []

rand_ints = random.sample(range(train_size), 20)

#programs = torch.load("/home/ubuntu/implementer_data/Great_programs.pyt")
for i in range(2048):
    programs.append(nn.Parameter(Variable(torch.randn((args.p_dim, 1)).to(device), requires_grad=True).data))
    optimizer_temp = optim.RMSprop([programs[i]], lr=args.p_lr)
    optimizers.append(optimizer_temp)
    #schedulers.append(CyclicLR(optimizer_temp, base_lr=0.00005, max_lr=0.01, step_size=args.epochs // 100, mode='triangular2'))
    schedulers.append(StepLR(optimizer_temp, step_size=max(1000, args.epochs) // 3, gamma=0.1))

#fig = plt.figure()



train_net(H, programs, optimizers, args.epochs, function_dataloader, True, train_e=False)


#ani = animation.ArtistAnimation(fig, ims, interval=25, blit=True, repeat_delay=0)

#ani.save('dynamic_images.mp4')

#plt.show()


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


# plt.figure()
# post_program_mat = np.zeros((len(programs), args.p_dim))
# for i in range(14):
#     post_program_mat[14*i:(14*i)+14,:] = np.array(torch.sigmoid(torch.stack(programs[i::14]).cpu().detach()).numpy()).reshape((50,args.p_dim))
# plt.imshow(post_program_mat.T)
# plt.show()

#plt.figure()
#post_program_mat = []
#for i in range(test_size):
#    post_program_mat.append(torch.stack(programs[i::test_size]))
#post_program_mat = torch.cat(post_program_mat).cpu().detach().numpy()
#plt.imshow(post_program_mat.reshape(test_size * args.test_copies, args.p_dim).T)
#plt.show()
