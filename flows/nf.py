import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distribution

class NormalizingFlow(nn.Module):

    def __init__(self, dim, flow_length):
        super().__init__()
        self.dim = dim
        # flows
        self.length = flow_length        
        self.transforms = nn.Sequential(*(
            PlanarFlow(dim) for _ in range(flow_length)
        ))
        # base distribution
        self.mu_ = nn.Parameter(torch.Tensor(1, dim))
        self.C_ = nn.Parameter(torch.Tensor(dim, dim))
        self.mu_.data.fill_(0)
        self.C_.data.fill_(0)
        self.base_dist = distribution.MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))
        
    def forward(self, z):
        sum_log_jacobians = torch.zeros(len(z), 1).to(z.device)
        z = z + self.mu_
        for transform in self.transforms:
            sum_log_jacobians = sum_log_jacobians + transform.log_jacobian(z)
            z = transform(z)
        zk = z
        return zk, sum_log_jacobians.view(-1)
    
    def to(self, device):
        super().to(device)
        self.device = device
        self.base_dist = distribution.MultivariateNormal(torch.zeros(self.dim).to(device), torch.eye(self.dim).to(device))
        return self
    
    def print(self):
        print('k=', self.length)
        print('self.mu_=', self.mu_)
        
class PlanarFlow(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
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
        return torch.log(det_jacobian.abs() + 1e-6)