import torch
import torch.nn as nn
from torch.autograd import Variable

class GraphLSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size):

        super(GraphLSTM, self).__init__()
        self.chains = [[15,14,13], [10,11,12], [5,4,3], [0,1,2]]
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(self.input_size, self.hidden_size)
        self.refine_layer = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, p3d):
        """
        p3d: [batch_size, 16, 3]
        """
        pre_p3d = p3d

        # forward LSTM
        forward_h = Variable(torch.zeros(pre_p3d.size(0), 16, self.hidden_size)).cuda()
        forward_c = Variable(torch.zeros(pre_p3d.size(0), 16, self.hidden_size)).cuda()
        for chain in self.chains:
            forward_h[:,chain[0],:], forward_c[:,chain[0],:] = self.cell(pre_p3d[:,chain[0],:])
            forward_h[:,chain[1],:], forward_c[:,chain[1],:] = self.cell(pre_p3d[:,chain[1],:], (forward_h[:,chain[0],:], forward_c[:,chain[0],:]))
            forward_h[:,chain[2],:], forward_c[:,chain[2],:] = self.cell(pre_p3d[:,chain[2],:], (forward_h[:,chain[1],:], forward_c[:,chain[1],:]))

        hx = torch.div(torch.add(forward_h[:,2,:], forward_h[:,3,:]))
        cx = torch.div(torch.add(forward_c[:,2,:], forward_c[:,3,:]))
        forward_h[:,6,:], forward_c[:,6,:] = self.cell(pre_p3d[:,6,:], (hx, cx))
        forward_h[:,7,:], forward_c[:,7,:] = self.cell(pre_p3d[:,7,:], (forward_h[:,6,:], forward_c[:,6,:]))

        hx = (forward_h[:,7,:] + forward_h[:,13,:] + forward_h[:,12,:])/3
        cx = (forward_c[:,7,:] + forward_c[:,13,:] + forward_c[:,12,:])/3
        forward_h[:,8,:], forward_c[:,8,:] = self.cell(pre_p3d[:,8,:], (hx, cx))
        forward_h[:,9,:], forward_c[:,9,:] = self.cell(pre_p3d[:,9,:], (forward_h[:,8,:], forward_c[:,8,:]))

        # backward LSTM
        back_h = Variable(torch.zeros(pre_p3d.size(0), 16, self.hidden_size)).cuda()
        back_c = Variable(torch.zeros(pre_p3d.size(0), 16, self.hidden_size)).cuda()

        back_h[:,9,:], back_c[:,9,:] = self.cell(pre_p3d[:,9,:])
        hx = back_h[:,9,:]
        cx = back_c[:,9,:]
        for i in [8,7,6]:
            back_h[:,i,:], back_c[:,i,:] = self.cell(pre_p3d[:,i,:], (hx,cx))
            hx = back_h[:,i,:]
            cx = back_c[:,i,:]
        
        for chain in self.chains:
            if chain[2] > 9:
                hx = back_h[:,8,:]
                cx = back_c[:,8,:]
            else:
                hx = back_h[:,6,:]
                cx = back_c[:,6,:]
            for i in chain.reverse():
                hx,cx = self.cell(pre_p3d[:,i,:], (hx,cx))
                back_h[:,i,:] = hx
                back_c[:,i,:] = cx

        # [batch_size, 16, lstm_size]
        output = (back_h + forward_h) / 2.0
        # [batch_size, 16, 3]
        output = self.refine_layer(output)

        return output
