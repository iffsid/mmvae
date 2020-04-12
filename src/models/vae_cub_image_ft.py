# CUB Image feature model specification

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from numpy import sqrt
from torchvision.utils import make_grid, save_image

from datasets import CUBImageFt
from utils import Constants, NN_lookup
from vis import plot_embeddings, plot_kls_df
from .vae import VAE

# Constants
imgChans = 3
fBase = 64


class Enc(nn.Module):
    """ Generate latent parameters for CUB image feature. """

    def __init__(self, latent_dim, n_c):
        super(Enc, self).__init__()
        dim_hidden = 256
        self.enc = nn.Sequential()
        for i in range(int(torch.tensor(n_c / dim_hidden).log2())):
            self.enc.add_module("layer" + str(i), nn.Sequential(
                nn.Linear(n_c // (2 ** i), n_c // (2 ** (i + 1))),
                nn.ELU(inplace=True),
            ))
        # relies on above terminating at dim_hidden
        self.fc21 = nn.Linear(dim_hidden, latent_dim)
        self.fc22 = nn.Linear(dim_hidden, latent_dim)

    def forward(self, x):
        e = self.enc(x)
        return self.fc21(e), F.softplus(self.fc22(e)) + Constants.eta


class Dec(nn.Module):
    """ Generate a CUB image feature given a sample from the latent space. """

    def __init__(self, latent_dim, n_c):
        super(Dec, self).__init__()
        self.n_c = n_c
        dim_hidden = 256
        self.dec = nn.Sequential()
        for i in range(int(torch.tensor(n_c / dim_hidden).log2())):
            indim = latent_dim if i == 0 else dim_hidden * i
            outdim = dim_hidden if i == 0 else dim_hidden * (2 * i)
            self.dec.add_module("out_t" if i == 0 else "layer" + str(i) + "_t", nn.Sequential(
                nn.Linear(indim, outdim),
                nn.ELU(inplace=True),
            ))
        # relies on above terminating at n_c // 2
        self.fc31 = nn.Linear(n_c // 2, n_c)

    def forward(self, z):
        p = self.dec(z.view(-1, z.size(-1)))
        mean = self.fc31(p).view(*z.size()[:-1], -1)
        return mean, torch.tensor([0.01]).to(mean.device)


class CUB_Image_ft(VAE):
    """ Derive a specific sub-class of a VAE for a CNN sentence model. """

    def __init__(self, params):
        super(CUB_Image_ft, self).__init__(
            dist.Normal,  # prior
            dist.Laplace,  # likelihood
            dist.Normal,  # posterior
            Enc(params.latent_dim, 2048),
            Dec(params.latent_dim, 2048),
            params
        )
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.modelName = 'cubIft'
        self.dataSize = torch.Size([2048])

        self.llik_scaling = 1.

    @property
    def pz_params(self):
        return self._pz_params[0], \
            F.softplus(self._pz_params[1]) + Constants.eta

    # remember that when combining with captions, this should be x10
    def getDataLoaders(self, batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}

        train_dataset = CUBImageFt('../data', 'train', device)
        test_dataset = CUBImageFt('../data', 'test', device)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size, shuffle=shuffle, **kwargs)

        train_dataset._load_data()
        test_dataset._load_data()
        self.unproject = lambda emb_h, search_split='train', \
            te=train_dataset.ft_mat, td=train_dataset.data_mat, \
            se=test_dataset.ft_mat, sd=test_dataset.data_mat: \
            NN_lookup(emb_h, te, td) if search_split == 'train' else NN_lookup(emb_h, se, sd)

        return train_loader, test_loader

    def generate(self, runPath, epoch):
        N, K = 64, 9
        samples = super(CUB_Image_ft, self).generate(N, K).data.cpu()
        samples = self.unproject(samples, search_split='train')
        samples = samples.view(K, N, *samples.size()[1:]).transpose(0, 1)
        s = [make_grid(t, nrow=int(sqrt(K)), padding=0) for t in samples.data.cpu()]
        save_image(torch.stack(s),
                   '{}/gen_samples_{:03d}.png'.format(runPath, epoch),
                   nrow=int(sqrt(N)))

    def reconstruct(self, data, runPath, epoch):
        recon = super(CUB_Image_ft, self).reconstruct(data[:8])
        data_ = self.unproject(data[:8], search_split='test')
        recon_ = self.unproject(recon, search_split='train')
        comp = torch.cat([data_, recon_])
        save_image(comp.data.cpu(), '{}/recon_{:03d}.png'.format(runPath, epoch))

    def analyse(self, data, runPath, epoch):
        zemb, zsl, kls_df = super(CUB_Image_ft, self).analyse(data, K=10)
        labels = ['Prior', self.modelName.lower()]
        plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(runPath, epoch))
        plot_kls_df(kls_df, '{}/kl_distance_{:03d}.png'.format(runPath, epoch))
