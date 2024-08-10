import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F

import math

class LSTMCell(nn.Module):

    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory' cell.
    http://www.bioinf.jku.at/publications/older/2604.pdf

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()



    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        
        hx, cx = hidden
        
        x = x.view(-1, x.size(1))
        
        gates = self.x2h(x) + self.h2h(hx)
    
        gates = gates.squeeze()
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        

        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)        

        hy = torch.mul(outgate, F.tanh(cy))
        
        return (hy, cy)


'''
STEP 3: CREATE MODEL CLASS
'''
 
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, teacher_fn, bias=True):
        super(LSTMModel, self).__init__()
        self.teacher_fn = teacher_fn
        # Hidden dimensions
        self.hidden_dim = hidden_dim
         
        # Number of hidden layers
        self.layer_dim = layer_dim
               
        self.lstm = LSTMCell(input_dim, hidden_dim, layer_dim)  
        
     
    
    def forward(self, x, teacher_mask, **kwargs):
        
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        #print(x.shape,"x.shape")100, 28, 28
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        # Initialize cell state
        if torch.cuda.is_available():
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

                    
       
        outs = []
        
        cn = c0[0,:,:]
        hn = h0[0,:,:]
        teacher_mask = teacher_mask.unsqueeze(-1)
        for seq in range(x.size(1)):
            mask = teacher_mask[:, seq, :]
            new_x = mask*self.teacher_fn(seq, (hn, cn), x[:, seq,:], **kwargs) + (1 - mask)*x[:, seq, :] 
            hn, cn = self.lstm(new_x, (hn,cn)) 
            outs.append(hn)
            
    

        out = torch.stack(outs, dim=1)
        
        # out.size() --> 100, 10
        return out, (hn, cn)
 

if __name__ == "__main__":
    '''
    STEP 4: INSTANTIATE MODEL CLASS
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 28
    hidden_dim = 128
    layer_dim = 1  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
     
    #model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim)

    input = torch.randn(5, 3, input_dim).to(device)
    mask = torch.randint(2, (5, 3)).to(device)
    def teacher_fn(idx, hidden, x):
        return mask[:, idx].unsqueeze(-1)

    model = LSTMModel(input_dim, hidden_dim, layer_dim, teacher_fn=teacher_fn).to(device)
    h0 = torch.randn(2, 3, 20)
    c0 = torch.randn(2, 3, 20)
    o = model(input, mask)
    print(o)
    #output, (hn, cn) = rnn(input, (h0, c0))

