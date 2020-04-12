# Sentence model specification - real CUB image version
import os
import json

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader

from datasets import CUBSentences
from utils import Constants, FakeCategorical
from .vae import VAE

# Constants
maxSentLen = 32  # max length of any description for birds dataset
minOccur = 3
embeddingDim = 128
lenWindow = 3
fBase = 32
vocabSize = 1590
vocab_path = '../data/cub/oc:{}_sl:{}_s:{}_w:{}/cub.vocab'.format(minOccur, maxSentLen, 300, lenWindow)


# Classes
class Enc(nn.Module):
    """ Generate latent parameters for sentence data. """

    def __init__(self, latentDim):
        super(Enc, self).__init__()
        self.embedding = nn.Embedding(vocabSize, embeddingDim, padding_idx=0)
        self.enc = nn.Sequential(
            # input size: 1 x 32 x 128
            nn.Conv2d(1, fBase, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 16 x 64
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 32
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # # size: (fBase * 4) x 4 x 16
            nn.Conv2d(fBase * 4, fBase * 4, (1, 4), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            nn.Conv2d(fBase * 4, fBase * 4, (1, 4), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
        )
        self.c1 = nn.Conv2d(fBase * 4, latentDim, 4, 1, 0, bias=False)
        self.c2 = nn.Conv2d(fBase * 4, latentDim, 4, 1, 0, bias=False)
        # c1, c2 size: latentDim x 1 x 1

    def forward(self, x):
        e = self.enc(self.embedding(x.long()).unsqueeze(1))
        mu, logvar = self.c1(e).squeeze(), self.c2(e).squeeze()
        return mu, F.softplus(logvar) + Constants.eta


class Dec(nn.Module):
    """ Generate a sentence given a sample from the latent space. """

    def __init__(self, latentDim):
        super(Dec, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latentDim, fBase * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
            nn.ConvTranspose2d(fBase * 4, fBase * 4, (1, 4), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            nn.ConvTranspose2d(fBase * 4, fBase * 4, (1, 4), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 4) x 8 x 32
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 16 x 64
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 32 x 128
            nn.ConvTranspose2d(fBase, 1, 4, 2, 1, bias=False),
            nn.ReLU(True)
            # Output size: 1 x 64 x 256
        )
        # inverts the 'embedding' module upto one-hotness
        self.toVocabSize = nn.Linear(embeddingDim, vocabSize)

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:])).view(-1, embeddingDim)

        return self.toVocabSize(out).view(*z.size()[:-3], maxSentLen, vocabSize),


class CUB_Sentence(VAE):
    """ Derive a specific sub-class of a VAE for a sentence model. """

    def __init__(self, params):
        super(CUB_Sentence, self).__init__(
            prior_dist=dist.Normal,
            likelihood_dist=FakeCategorical,
            post_dist=dist.Normal,
            enc=Enc(params.latent_dim),
            dec=Dec(params.latent_dim),
            params=params)
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.modelName = 'cubS'
        self.llik_scaling = 1.

        self.tie_modules()

        self.fn_2i = lambda t: t.cpu().numpy().astype(int)
        self.fn_trun = lambda s: s[:np.where(s == 2)[0][0] + 1] if 2 in s else s
        self.vocab_file = vocab_path

        self.maxSentLen = maxSentLen
        self.vocabSize = vocabSize

    def tie_modules(self):
        # This looks dumb, but is actually dumber than you might realise.
        # A linear(a, b) module has a [b x a] weight matrix, but an embedding(a, b)
        # module has a [a x b] weight matrix. So when we want the transpose at
        # decoding time, we just use the weight matrix as is.
        self.dec.toVocabSize.weight = self.enc.embedding.weight

    @property
    def pz_params(self):
        return self._pz_params[0], F.softplus(self._pz_params[1]) + Constants.eta

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        tx = lambda data: torch.Tensor(data)
        t_data = CUBSentences('../data', split='train', transform=tx, max_sequence_length=maxSentLen)
        s_data = CUBSentences('../data', split='test', transform=tx, max_sequence_length=maxSentLen)

        train_loader = DataLoader(t_data, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(s_data, batch_size=batch_size, shuffle=shuffle, **kwargs)

        return train_loader, test_loader

    def reconstruct(self, data, runPath, epoch):
        recon = super(CUB_Sentence, self).reconstruct(data[:8]).argmax(dim=-1).squeeze()
        recon, data = self.fn_2i(recon), self.fn_2i(data[:8])
        recon, data = [self.fn_trun(r) for r in recon], [self.fn_trun(d) for d in data]
        i2w = self.load_vocab()
        print("\n Reconstruction examples (excluding <PAD>):")
        for r_sent, d_sent in zip(recon[:3], data[:3]):
            print('[DATA]  ==> {}'.format(' '.join(i2w[str(i)] for i in d_sent)))
            print('[RECON] ==> {}\n'.format(' '.join(i2w[str(i)] for i in r_sent)))

        with open('{}/recon_{:03d}.txt'.format(runPath, epoch), "w+") as txt_file:
            for r_sent, d_sent in zip(recon, data):
                txt_file.write('[DATA]  ==> {}\n'.format(' '.join(i2w[str(i)] for i in d_sent)))
                txt_file.write('[RECON] ==> {}\n\n'.format(' '.join(i2w[str(i)] for i in r_sent)))

    def generate(self, runPath, epoch):
        N, K = 5, 4
        i2w = self.load_vocab()
        samples = super(CUB_Sentence, self).generate(N, K).argmax(dim=-1).squeeze()
        samples = samples.view(K, N, samples.size(-1)).transpose(0, 1)  # N x K x 64
        samples = [[self.fn_trun(s) for s in ss] for ss in self.fn_2i(samples)]
        # samples = [self.fn_trun(s) for s in samples]
        print("\n Generated examples (excluding <PAD>):")
        for s_sent in samples[0][:3]:
            print('[GEN]   ==> {}'.format(' '.join(i2w[str(i)] for i in s_sent if i != 0)))

        with open('{}/gen_samples_{:03d}.txt'.format(runPath, epoch), "w+") as txt_file:
            for s_sents in samples:
                for s_sent in s_sents:
                    txt_file.write('{}\n'.format(' '.join(i2w[str(i)] for i in s_sent)))
                txt_file.write('\n')

    def analyse(self, data, runPath, epoch):
        pass

    def load_vocab(self):
        # call dataloader function to create vocab file
        if not os.path.exists(self.vocab_file):
            _, _ = self.getDataLoaders(256)
        with open(self.vocab_file, 'r') as vocab_file:
            vocab = json.load(vocab_file)
        return vocab['i2w']
