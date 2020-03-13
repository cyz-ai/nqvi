import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distribution

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@torch.no_grad()
def plot_target_density(u_z, ax, z_range=[-4, 4], n=200, device=torch.device('cpu'), output_file=None):
    x = torch.linspace(z_range[0], z_range[1], n)
    xx, yy = torch.meshgrid((x, x))
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=-1).squeeze().to(device)
    pk = torch.exp(-u_z(zz)).cpu()
    ax.pcolormesh(xx, yy, pk.view(n,n).data, cmap=plt.cm.jet)

    for ax in plt.gcf().axes:
        ax.set_xlim(z_range[0], z_range[1])
        ax.set_ylim(z_range[0], z_range[1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #ax.invert_yaxis()

    if output_file:
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

@torch.no_grad()
def plot_flow_density(flow, ax, z_range=[-4, 4], n=200, device=torch.device('cpu'), output_file=None):
    x = torch.linspace(-3.5, 3.5, n)
    xx, yy = torch.meshgrid((x, x))
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=-1).squeeze().to(device)

    # plot posterior approx density
    zzk, sum_log_abs_det_jacobians = flow(zz)
    log_q0 = flow.base_dist.log_prob(zz)
    log_qk = log_q0 - sum_log_abs_det_jacobians
    qk = log_qk.exp().cpu()
    zzk = zzk.cpu()
    ax.pcolormesh(zzk[:,0].view(n,n).data, zzk[:,1].view(n,n).data, qk.view(n,n).data, cmap=plt.cm.jet)
    ax.set_facecolor(plt.cm.jet(0.))

    for ax in plt.gcf().axes:
        ax.set_xlim(z_range[0], z_range[1])
        ax.set_ylim(z_range[0], z_range[1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #ax.invert_yaxis()

    if output_file:
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()