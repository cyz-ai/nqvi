import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distribution


class NeuralQuantileFlow(nn.Module):

    def __init__(self, dim, flow_length, inversion):
        super().__init__()
        self.dim = dim
        self.length = flow_length
        self.inversion = inversion
        self.transforms = nn.Sequential(
            GK(dim),
            *(PlanarFlow(dim) for _ in range(flow_length-1)
        ))
        self.base_dist = distribution.MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))
       
    def forward(self, z):
        n, d = z.size()
        sum_log_jacobians = torch.zeros(n, 1).to(z.device)
        for transform in self.transforms:
            sum_log_jacobians = sum_log_jacobians + transform.log_jacobian(z)
            z = transform(z)
        zk = z
        return zk, sum_log_jacobians.view(-1)
                  
    def to(self, device):
        super().to(device)
        self.base_dist = distribution.MultivariateNormal(torch.zeros(self.dim).to(device), torch.eye(self.dim).to(device))
        return self
       
    def print(self):
        GK = self.transforms[0]
        A,B,g,k,C,c = GK.reparam()
        print('A=', A.data)
        print('B=', B.data)
        print('g=', g.data)
        print('k=', k.data)
        print('c=', c.data)
        print('V=', C.mm(C.t()).data)
          
class GK(nn.Module):
    
    def __init__(self, dim):
        super().__init__()    
        self.dim = dim
        self.A_ = nn.Parameter(torch.Tensor(1, dim))
        self.B_ = nn.Parameter(torch.Tensor(1, dim))
        self.g_ = nn.Parameter(torch.Tensor(1, dim))
        self.c_ = nn.Parameter(torch.Tensor(1, dim))
        self.k_ = nn.Parameter(torch.Tensor(1, dim))
        self.C_ = nn.Parameter(torch.Tensor(dim, dim))

        self.A_.data.fill_(0)
        self.B_.data.fill_(0)  
        self.g_.data.fill_(0)
        self.c_.data.fill_(1.5)
        self.k_.data = (torch.zeros(self.k_.size())-0.69315).data
        self.C_.data.fill_(0)
 
    def reparam(self):
        A,B,g,k,c = self.A_,torch.exp(self.B_),self.g_,torch.exp(self.k_)-0.5,torch.sigmoid(self.c_)*0.82
        S = torch.tril(self.C_, diagonal=-1) + torch.diagflat(self.C_.diag().exp())
        T = S.mm(S.t())
        D = torch.diagflat(T.diag().pow(-0.5))
        C = D.mm(S)
        return A,B,g,k,C,c
        
    def forward(self, z):
        (n, d) = z.size()
        A,B,g,k,C,c = self.reparam()
        z = z.mm(C.t())
        w = (1-torch.exp(-g*z))/(1+torch.exp(-g*z))
        v = z*(1+z**2).pow(k)
        x = A + B*(1 + c*w)*v
        return x
     
    def log_jacobian(self, z):
        (n, d) = z.size()
        A,B,g,k,C,c = self.reparam()
        z = z.mm(C.t())
        w = (1-torch.exp(-g*z))/(1+torch.exp(-g*z))
        v = z*(1+z**2).pow(k)
        dw_z = 2*g*torch.exp(-g*z)/(1+torch.exp(-g*z))**2
        dv_z = (1+z**2).pow(k) + (2*k*z**2)*(1+z**2).pow(k-1)
        dx_z = B*(c*dw_z*v + (1+c*w)*dv_z)
        log_abs_det1 = torch.sum(torch.log(dx_z.abs()), dim=1)
        log_abs_det2 = torch.log(torch.det(C).abs()).repeat(n)
        return (log_abs_det1 + log_abs_det2).view(n, -1)
    
class PlanarFlow(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.tanh = nn.Tanh()

        self.weight.data.uniform_(-0.01, 0.01)
        self.scale.data.uniform_(-0.01, 0.01)
        self.bias.data.uniform_(-0.01, 0.01)
        
    def forward(self, z):
        activation = F.linear(z, self.weight, self.bias)
        return z + self.scale * self.tanh(activation)
    
    def log_jacobian(self, z):
        activation = F.linear(z, self.weight, self.bias)
        psi = (1 - self.tanh(activation) ** 2) * self.weight
        det_jacobian = 1 + torch.mm(psi, self.scale.t())
        return torch.log(det_jacobian.abs())