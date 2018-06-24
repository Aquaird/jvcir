from .layers.Residual import Residual
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import ref
from torch.autograd import Variable

class Hourglass(nn.Module):
    def __init__(self, n, nModules, nFeats):
        super(Hourglass, self).__init__()
        self.n = n
        self.nModules = nModules
        self.nFeats = nFeats

        _up1_, _low1_, _low2_, _low3_ = [], [], [], []
        for j in range(self.nModules):
            _up1_.append(Residual(self.nFeats, self.nFeats))
        self.low1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        for j in range(self.nModules):
            _low1_.append(Residual(self.nFeats, self.nFeats))
    
        if self.n > 1:
            self.low2 = Hourglass(n - 1, self.nModules, self.nFeats)
        else:
            for j in range(self.nModules):
                _low2_.append(Residual(self.nFeats, self.nFeats))
            self.low2_ = nn.ModuleList(_low2_)
    
        for j in range(self.nModules):
            _low3_.append(Residual(self.nFeats, self.nFeats))
    
        self.up1_ = nn.ModuleList(_up1_)
        self.low1_ = nn.ModuleList(_low1_)
        self.low3_ = nn.ModuleList(_low3_)
    
        self.up2 = nn.Upsample(scale_factor = 2)
    
    def forward(self, x):
        up1 = x
        for j in range(self.nModules):
            up1 = self.up1_[j](up1)
    
        low1 = self.low1(x)
        for j in range(self.nModules):
            low1 = self.low1_[j](low1)
    
        if self.n > 1:
            low2 = self.low2(low1)
        else:
            low2 = low1
            for j in range(self.nModules):
                low2 = self.low2_[j](low2)
    
        low3 = low2
        for j in range(self.nModules):
            low3 = self.low3_[j](low3)
        up2 = self.up2(low3)

        return up1 + up2

class HourglassNet3D(nn.Module):
    def __init__(self, nStack, nModules, nFeats, nRegModules, nCamModules):
        super(HourglassNet3D, self).__init__()
        self.nStack = nStack
        self.nModules = nModules
        self.nFeats = nFeats
        self.nRegModules = nRegModules
        self.nCamModules = nCamModules
        self.conv1_ = nn.Conv2d(3, 64, bias = True, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=True)
        self.relu = nn.ReLU(inplace = True)
        self.r1 = Residual(64, 128)
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.r4 = Residual(128, 128)
        self.r5 = Residual(128, self.nFeats)
   
        _hourglass, _Residual, _lin_, _tmpOut, _ll_, _tmpOut_, _reg_, _camera_ = [], [], [], [], [], [], [], []
        for i in range(self.nStack):
            _hourglass.append(Hourglass(4, self.nModules, self.nFeats))
            for j in range(self.nModules):
                _Residual.append(Residual(self.nFeats, self.nFeats))
            lin = nn.Sequential(nn.Conv2d(self.nFeats, self.nFeats, bias = True, kernel_size = 1, stride = 1), 
                                nn.BatchNorm2d(self.nFeats), self.relu)
            _lin_.append(lin)
            _tmpOut.append(nn.Conv2d(self.nFeats, ref.nJoints, bias = True, kernel_size = 1, stride = 1))
            _ll_.append(nn.Conv2d(self.nFeats, self.nFeats, bias = True, kernel_size = 1, stride = 1))
            _tmpOut_.append(nn.Conv2d(ref.nJoints, self.nFeats, bias = True, kernel_size = 1, stride = 1))

        for i in range(4):
            for j in range(self.nRegModules):
                _reg_.append(Residual(self.nFeats, self.nFeats))
        
        for i in range(4):
            for j in range(self.nCamModules):
                _camera_.append(Residual(self.nFeats, self.nFeats))


        self.hourglass = nn.ModuleList(_hourglass)
        self.Residual = nn.ModuleList(_Residual)
        self.lin_ = nn.ModuleList(_lin_)
        self.tmpOut = nn.ModuleList(_tmpOut)
        self.ll_ = nn.ModuleList(_ll_)
        self.tmpOut_ = nn.ModuleList(_tmpOut_)
        self.reg_ = nn.ModuleList(_reg_)
        self.cam_ = nn.ModuleList(_camera_)
    
        self.reg = nn.Linear(4 * 4 * self.nFeats, ref.nJoints)
        self.cam = nn.Linear(4 * 4 * self.nFeats, 6)

    def rotation_tensor(self, param):
        #param [batch_size, 6]
        theta = param[:,0].view(-1,1,1)
        phi = param[:,1].view(-1,1,1)
        psi = param[:,2].view(-1,1,1)
        n_comps = param.size(0)
        #[batch_size, 3, 3]
        one = Variable(torch.ones(n_comps, 1, 1), requires_grad=False).cuda()
        zero = Variable(torch.zeros(n_comps, 1, 1), requires_grad=False).cuda()
        rot_x = torch.cat((
            torch.cat((one, zero, zero), 1),
            torch.cat((zero, theta.cos(), theta.sin()), 1),
            torch.cat((zero, -theta.sin(), theta.cos()), 1),)
            , 2)
        rot_y = torch.cat((
            torch.cat((phi.cos(), zero, -phi.sin()), 1),
            torch.cat((zero, one, zero), 1),
            torch.cat((phi.sin(), zero, phi.cos()), 1),
            ), 2)
        rot_z = torch.cat((
            torch.cat((psi.cos(), -psi.sin(), zero), 1),
            torch.cat((psi.sin(), psi.cos(), zero), 1),
            torch.cat((zero, zero, one), 1)
            ), 2)

        return torch.bmm(rot_z, torch.bmm(rot_y, rot_x))

    def softargmax(self, feature):
        #feature [batch_size, 16, 64, 64]
        feature = feature.view(-1, 16, self.nFeats*self.nFeats)
        feature = F.softmax(feature, dim=-1)
        feature = feature.view(-1, 16, self.nFeats, self.nFeats)

        x_expectation = torch.arange(0, self.nFeats).repeat(self.nFeats).view(self.nFeats, self.nFeats).cuda()
        y_expectation = torch.arange(0, self.nFeats).repeat(self.nFeats).view(self.nFeats, self.nFeats).transpose(0,1).cuda()

        softargmax_x = torch.sum(torch.matmul(feature, x_expectation).view(-1,16, self.nFeats*self.nFeats), dim=-1)
        softargmax_y = torch.sum(torch.matmul(feature, y_expectation).view(-1,16, self.nFeats*self.nFeats), dim=-1)

        # [batch_size, 16, 2]
        return torch.cat((softargmax_x.view(-1,16,1), softargmax_y.view(-1,16,1)), -1)

    def forward(self, x):
        x = self.conv1_(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.r1(x)
        x = self.maxpool(x)
        x = self.r4(x)
        x = self.r5(x)
    
        out = []
    
        for i in range(self.nStack):
            hg = self.hourglass[i](x)
            ll = hg
            for j in range(self.nModules):
                ll = self.Residual[i * self.nModules + j](ll)
            ll = self.lin_[i](ll)
            # [b, 16, 64, 64]
            tmpOut = self.tmpOut[i](ll)
            out.append(tmpOut)
            # conv(f)->bn->conv(f)
            ll_ = self.ll_[i](ll)
            # conv(f)->bn->conv(64)->conv(f)
            tmpOut_ = self.tmpOut_[i](tmpOut)
            x = x + ll_ + tmpOut_

        cx = x
        fx = x
        for i in range(4):
            for j in range(self.nRegModules):
                x = self.reg_[i * self.nRegModules + j](x)
            x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        reg = self.reg(x)
        out.append(reg)
 
        for i in range(4):
            for j in range(self.nCamModules):
               cx = self.reg_[i * self.nCamModules + j](cx)
            cx = self.maxpool(cx)
           
        cx = cx.view(x.size(0), -1)
        cam = self.cam(cx)
        out.append(reg)
        out.append(cam)

        # [batch_size, 16, 2]
        fx  = self.softargmax(fx)
        #[batch_size, 16, 3]
        #print(fx.size(), reg.view(-1,16,1).size())
        p3d = torch.cat((fx, reg.view(-1, 16, 1)), dim=-1)
        out.append(p3d)
        p3d = p3d.view(-1,16, 3, 1)
        # [batch_size, 3, 3]
        rotate_matrix = self.rotation_tensor(cam).view(-1,1,3,3)
        #print(rotate_matrix.size(), p3d.size())
        p3d_refine = torch.matmul(rotate_matrix.cuda(), p3d).view(-1,16,3)
        out.append(p3d_refine)
        #(0:nstack) [batch_size, 16, 64, 64] 2d heatmap
        #(nstack) [batch_size, 16] reg
        #(nstack+1) [batch_size, 6] camera param
        #(nstack+2) [batch_size, 16, 3] original p3d
        #(nstack+3) [batch_size, 16, 3] view adaptive p3d

        return out

