import torch.nn as nn
import torch
from torch.autograd import Variable, Function

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
        self.lstm1 = torch.nn.LSTMCell(input_size=1, hidden_size=self.lstm_size)
        self.lstm2 = torch.nn.LSTMCell(input_size=self.lstm_size, hidden_size=self.lstm_size)
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
            for i, input_t in enumerate(range(256)):
                h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
                #if c2 == self.output_dim:
                outputs += [torch.mm(h_t2, self.fc1)]
                break
            h2 = torch.stack(outputs, 1).squeeze()
            h2 = torch.stack(outputs, 1).squeeze()
            break
        #return h2.view(-1, 1, 16, 16)
        return h2.view(-1, 1, 16, 16)

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


