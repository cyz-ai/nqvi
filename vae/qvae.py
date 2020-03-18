from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import pdb
import os

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('-d', type=int, default=0)
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

torch.manual_seed(args.seed)

device = torch.device("cuda:{}".format(args.d))

kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self, z_dim=20, hidden_dim=400):
        super(VAE, self).__init__()

        # encoder
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.fc23 = nn.Linear(hidden_dim, z_dim)
        self.fc24 = nn.Linear(hidden_dim, z_dim)
        self.fc25 = nn.Linear(hidden_dim, z_dim)
        # self.fc26 = nn.Linear(hidden_dim, z_dim ** 2)

        # decoder
        self.fc3 = nn.Linear(z_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        A_ = self.fc21(h1)
        B_ = self.fc22(h1)
        g_ = self.fc23(h1)
        k_ = self.fc24(h1)
        c_ = self.fc25(h1)
        return A_, B_, g_, k_, c_

    def reparameterize(self, A_, B_, g_, k_, c_):
        A, B, g, k, c = A_, torch.exp(B_), g_, torch.exp(k_)-0.5, torch.sigmoid(c_)*0.82
        z = torch.randn_like(A).to(A.device)
        w = (1 - torch.exp(- g * z)) / (1 + torch.exp(- g * z))
        v = z * (1 + z ** 2).pow(k)
        z = A + B * (1 + c * w) * v
        param = (A, B, g, k, c, w, v)
        # pdb.set_trace()
        return z, param

        # std = torch.exp(0.5*logvar)
        # eps = torch.randn_like(std)
        # return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        A_, B_, g_, k_, c_ = self.encode(x.view(-1, 784))
        z, param = self.reparameterize(A_, B_, g_, k_, c_)
        return self.decode(z), z, param


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, z, param):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    A, B, g, k, c, w, v = param
    # w = (1-torch.exp(-g*z))/(1+torch.exp(-g*z))
    # v = z*(1+z**2).pow(k)
    # Q = A + B * (1 + c * w) * v
    dw_z = 2*g*torch.exp(-g*z)/(1+torch.exp(-g*z))**2
    dv_z = (1+z**2).pow(k) + (2*k*z**2)*(1+z**2).pow(k-1)
    dx_z = B*(c*dw_z*v + (1+c*w)*dv_z)
    log_abs_det1 = torch.sum(torch.log(dx_z.abs()), dim=1)
    KLD = - log_abs_det1.sum()

    # # see Appendix B from VAE paper:
    # # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # # https://arxiv.org/abs/1312.6114
    # # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # print('BCE: {:.6f} KLD: {:.6f}'.format(BCE.item(), KLD.item()))
    return BCE + KLD
    # return BCE


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, z, param = model(data)
        loss = loss_function(recon_batch, data, z, param)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # print(recon_batch.min().item(), recon_batch.max().item())

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    makedirs('results/qvae/')

    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/qvae/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if __name__ == "__main__":
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/qvae/sample_' + str(epoch) + '.png')
