import torch.nn as nn
import torch
from torch.autograd import Variable, Function
from hyperlstm import HyperLSTMCell, LSTMCell

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        self.lstm_size = 256 # args.imsize ** 2
        sigma = 0.01
        self.hyper = HyperLSTMCell(
                input_size=1, hidden_size=self.lstm_size,
                hyper_hidden_size=128,
                hyper_embedding_size=4,
                use_layer_norm=True, dropout_prob=0.0)
        #self.fc1 = nn.Parameter(torch.randn(self.lstm_size, 256) * sigma)
        self.fc1 = nn.Parameter(torch.randn(self.lstm_size, 256) * sigma)

        if dataloader is None:
            self.dataloader = Dataset(batch_size=64)
        else:
            self.dataloader = dataloader

    def implement_W(self, p, epoch, redundant_train=False):
        p = p.view(-1, p.shape[-2])
        p = torch.tanh(p)
        if redundant_train:
            noise = torch.autograd.Variable(torch.randn((len(p), self.p_dim)).cuda() * min(epoch * 0.0003, 0.45))
            p = p + (noise.cuda() * (torch.abs(p) > 0.001).float())
        #p = p.view(-1, args.p_dim)
        batch = p.shape[0]
        outputs = []
        h_t2 = Variable(torch.zeros(batch, self.lstm_size).float(), requires_grad=False).to(device)
        c_t2 = Variable(torch.zeros(batch, self.lstm_size).float(), requires_grad=False).to(device)
        #h1 = Variable(torch.zeros(p.size(0), self.lstm_size).float(), requires_grad=False).to(device)
        state = hyper_state = None
        for i, input_t in enumerate(p.chunk(p.shape[1], dim=1)):
            results, state, hyper_state = self.hyper(x=input_t, state=state, hyper_state=hyper_state)
        return results.view(-1, 1, 16, 16)

    def forward(self, input_data, p, epoch, output_mult=False):
        weights = self.implement_W(p, epoch, redundant_train=True)
        if not output_mult:
            return weights
        result = []
        for j in range(len(weights)):
            inter = torch.tanh(torch.matmul(input_data.view(-1, 16, 16), weights[j][0].view(16, 16)))
            result.append(torch.tanh(torch.matmul(inter.view(-1, 16, 16), weights[j][1].view(16, 16))))
        result = torch.stack(result).view(-1, 1, 16, 16)
        return weights, result


