# -*- coding: utf-8 -*-
from libraries import *
from global_utilities import *

# inputs are (batch_size, 1, num_features/inputs)
class price_rnn(nn.Module):
    def __init__(self, mode='GRU'):
        super(price_rnn, self).__init__()
        
        # network for price
        if mode == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layer, nonlinearity='tanh')
        if mode == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layer)
        if mode == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layer)

        # linear layers
        self.linear_embd = nn.Linear(hidden_size, hidden_size)
        # linear layer
        self.linear_out = nn.Linear(hidden_size, output_size)
        # linear multiplier
        self.alpha = 0.5 #nn.Parameter(0.5 * torch.ones((seq_len,1)))

        self.weight_init()

    def weight_init(self):
        # print('initializing parameters for pricing network')
        for name, module in self.named_children():
            # print('initializing ', name)
            if name == "rnn":
                for layer in module._all_weights:
                    for p in layer:
                        if 'weight' in p:
                            torch.nn.init.xavier_uniform_(module.__getattr__(p), gain=2)
            else:
                module.weight = torch.nn.init.xavier_uniform_(module.weight, gain=2)
    
    def init_hidden(self, u):
        #print('Initializing hidden states for pricing network \n')
        h0 = torch.nn.Parameter(u.repeat(num_layer,1,hidden_size))
        return h0

    def forward(self, x, h0, out_prev): # 
        # pricing network
        out_rnn,_ = self.rnn(x, h0)
        out_embd = torch.sigmoid(self.linear_embd(out_rnn))*self.linear_embd(out_rnn)
        out_upd = F.softplus(self.linear_out(out_embd),beta=num_layer)
        out = (1-self.alpha)*out_prev + self.alpha*out_upd.squeeze() #
        return out

class delta_rnn(nn.Module):
    def __init__(self, mode='GRU'):
        super(delta_rnn, self).__init__()
        # network for delta
        if mode == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layer, nonlinearity='tanh')
        if mode == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layer)
        if mode == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layer)

        # linear layers
        self.linear_embd = nn.Linear(hidden_size, hidden_size)
        # linear layer
        self.linear_out = nn.Linear(hidden_size, output_size)
        # linear multiplier
        self.beta = 0.5 #nn.Parameter(0.5 * torch.ones((seq_len,1)))

        self.weight_init()

    def weight_init(self):
        # print('initializing parameters for pricing network')
        for name, module in self.named_children():
            # print('initializing ', name)
            if name == "rnn":
                for layer in module._all_weights:
                    for p in layer:
                        if 'weight' in p:
                            torch.nn.init.xavier_uniform_(module.__getattr__(p))
            else:
                module.weight = torch.nn.init.xavier_uniform_(module.weight)
    
    def init_hidden(self, z):
        #print('Initializing hidden states for delta network \n')
        h0 = torch.nn.Parameter(z.repeat(num_layer,1,hidden_size)) 
        return h0

    def forward(self, x, h0, out_prev): #
        # delta network
        out_rnn,_ = self.rnn(x,h0)
        out_embd = torch.sigmoid(self.linear_embd(out_rnn)) * self.linear_embd(out_rnn)
        out_upd = torch.sigmoid(self.linear_out(out_embd) * num_layer)
        out = (1-self.beta)*out_prev + self.beta*out_upd.squeeze() #
        return out

def norm(x):
    if x.min() == x.max():
        x = x
    else:
        x = (x - x.min())/(x.max() - x.min())
    return x

def bnorm(x):
    return (x - x.mean())/(x.std())