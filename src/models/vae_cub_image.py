# CUB Image model specification

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from numpy import sqrt
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from utils import Constants
from vis import plot_embeddings, plot_kls_df
from .vae import VAE

# Constants
imgChans = 3
fBase = 64


# Classes
class Enc(nn.Module):
    """ Generate latent parameters for CUB image data. """

    def __init__(self, latentDim):
        super(Enc, self).__init__()
        modules = [
            # input size: 3 x 128 x 128
            nn.Conv2d(imgChans, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # input size: 1 x 64 x 64
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 32 x 32
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 16 x 16
            nn.Conv2d(fBase * 4, fBase * 8, 4, 2, 1, bias=True),
            nn.ReLU(True)]
        # size: (fBase * 8) x 4 x 4

        self.enc = nn.Sequential(*modules)
        self.c1 = nn.Conv2d(fBase * 8, latentDim, 4, 1, 0, bias=True)
        self.c2 = nn.Conv2d(fBase * 8, latentDim, 4, 1, 0, bias=True)
        # c1, c2 size: latentDim x 1 x 1

    def forward(self, x):
        e = self.enc(x)
        return self.c1(e).squeeze(), F.softplus(self.c2(e)).squeeze() + Constants.eta


class Dec(nn.Module):
    """ Generate an image given a sample from the latent space. """

    def __init__(self, latentDim):
        super(Dec, self).__init__()
        modules = [nn.ConvTranspose2d(latentDim, fBase * 8, 4, 1, 0, bias=True),
                   nn.ReLU(True), ]

        modules.extend([
            nn.ConvTranspose2d(fBase * 8, fBase * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 16 x 16
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 32 x 32
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 64 x 64
            nn.ConvTranspose2d(fBase, imgChans, 4, 2, 1, bias=True),
            nn.Sigmoid()
            # Output size: 3 x 128 x 128
        ])
        self.dec = nn.Sequential(*modules)

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:]))
        out = out.view(*z.size()[:-3], *out.size()[1:])
        return out, torch.tensor(0.01).to(z.device)


class CUB_Image(VAE):
    """ Derive a specific sub-class of a VAE for a CNN sentence model. """

    def __init__(self, params):
        super(CUB_Image, self).__init__(
            dist.Laplace,  # prior
            dist.Laplace,  # likelihood
            dist.Laplace,  # posterior
            Enc(params.latent_dim),
            Dec(params.latent_dim),
            params
        )
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.modelName = 'cubI'
        self.dataSize = torch.Size([3, 64, 64])
        self.llik_scaling = 1.

    @property
    def pz_params(self):
        return self._pz_params[0], F.softplus(self._pz_params[1]) + Constants.eta

    # remember that when combining with captions, this should be x10
    def getDataLoaders(self, batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        tx = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('../data/cub/train', transform=tx),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('../data/cub/test', transform=tx),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train_loader, test_loader

    def generate(self, runPath, epoch):
        N, K = 64, 9
        samples = super(CUB_Image, self).generate(N, K).data.cpu()
        # wrangle things so they come out tiled
        samples = samples.view(K, N, *samples.size()[1:]).transpose(0, 1)
        s = [make_grid(t, nrow=int(sqrt(K)), padding=0) for t in samples.data.cpu()]
        save_image(torch.stack(s),
                   '{}/gen_samples_{:03d}.png'.format(runPath, epoch),
                   nrow=int(sqrt(N)))

    def reconstruct(self, data, runPath, epoch):
        recon = super(CUB_Image, self).reconstruct(data[:8])
        comp = torch.cat([data[:8], recon])
        save_image(comp.data.cpu(), '{}/recon_{:03d}.png'.format(runPath, epoch))

    def analyse(self, data, runPath, epoch):
        zemb, zsl, kls_df = super(CUB_Image, self).analyse(data, K=10)
        labels = ['Prior', self.modelName.lower()]
        plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(runPath, epoch))
        plot_kls_df(kls_df, '{}/kl_distance_{:03d}.png'.format(runPath, epoch))
