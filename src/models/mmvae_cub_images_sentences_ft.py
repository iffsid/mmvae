# cub multi-modal model specification
import matplotlib.pyplot as plt
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from numpy import sqrt, prod
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset
from torchvision.utils import save_image, make_grid

from utils import Constants
from vis import plot_embeddings, plot_kls_df
from .mmvae import MMVAE
from .vae_cub_image_ft import CUB_Image_ft
from .vae_cub_sent_ft import CUB_Sentence_ft

# Constants
maxSentLen = 32
minOccur = 3


# This is required because there are 10 captions per image.
# Allows easier reuse of the same image for the corresponding set of captions.
def resampler(dataset, idx):
    return idx // 10


class CUB_Image_Sentence_ft(MMVAE):

    def __init__(self, params):
        super(CUB_Image_Sentence_ft, self).__init__(dist.Normal, params, CUB_Image_ft, CUB_Sentence_ft)
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.vaes[0].llik_scaling = self.vaes[1].maxSentLen / prod(self.vaes[0].dataSize) \
            if params.llik_scaling == 0 else params.llik_scaling

        for vae in self.vaes:
            vae._pz_params = self._pz_params
        self.modelName = 'cubISft'

        self.i2w = self.vaes[1].load_vocab()

    @property
    def pz_params(self):
        return self._pz_params[0], \
            F.softplus(self._pz_params[1]) + Constants.eta

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda'):
        # load base datasets
        t1, s1 = self.vaes[0].getDataLoaders(batch_size, shuffle, device)
        t2, s2 = self.vaes[1].getDataLoaders(batch_size, shuffle, device)

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train_loader = DataLoader(TensorDataset([
            ResampleDataset(t1.dataset, resampler, size=len(t1.dataset) * 10),
            t2.dataset]), batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(TensorDataset([
            ResampleDataset(s1.dataset, resampler, size=len(s1.dataset) * 10),
            s2.dataset]), batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train_loader, test_loader

    def generate(self, runPath, epoch):
        N = 8
        samples = super(CUB_Image_Sentence_ft, self).generate(N)
        samples[0] = self.vaes[0].unproject(samples[0], search_split='train')
        images, captions = [sample.data.cpu() for sample in samples]
        captions = self._sent_preprocess(captions)
        fig = plt.figure(figsize=(8, 6))
        for i, (image, caption) in enumerate(zip(images, captions)):
            fig = self._imshow(image, caption, i, fig, N)

        plt.savefig('{}/gen_samples_{:03d}.png'.format(runPath, epoch))
        plt.close()

    def reconstruct(self, raw_data, runPath, epoch):
        N = 8
        recons_mat = super(CUB_Image_Sentence_ft, self).reconstruct([d[:N] for d in raw_data])
        fns = [lambda images: images.data.cpu(), lambda sentences: self._sent_preprocess(sentences)]
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                data = fns[r](raw_data[r][:N])
                recon = fns[o](recon.squeeze())
                if r != o:
                    fig = plt.figure(figsize=(8, 6))
                    for i, (_data, _recon) in enumerate(zip(data, recon)):
                        image, caption = (_data, _recon) if r == 0 else (_recon, _data)
                        search_split = 'test' if r == 0 else 'train'
                        image = self.vaes[0].unproject(image.unsqueeze(0), search_split=search_split)
                        fig = self._imshow(image, caption, i, fig, N)
                    plt.savefig('{}/recon_{}x{}_{:03d}.png'.format(runPath, r, o, epoch))
                    plt.close()
                else:
                    if r == 0:
                        data_ = self.vaes[0].unproject(data, search_split='test')
                        recon_ = self.vaes[0].unproject(recon, search_split='train')
                        comp = torch.cat([data_, recon_])
                        save_image(comp, '{}/recon_{}x{}_{:03d}.png'.format(runPath, r, o, epoch))
                    else:
                        with open('{}/recon_{}x{}_{:03d}.txt'.format(runPath, r, o, epoch), "w+") as txt_file:
                            for r_sent, d_sent in zip(recon, data):
                                txt_file.write('[DATA]  ==> {}\n'.format(' '.join(self.i2w[str(i)] for i in d_sent)))
                                txt_file.write('[RECON] ==> {}\n\n'.format(' '.join(self.i2w[str(i)] for i in r_sent)))

    def analyse(self, data, runPath, epoch):
        zemb, zsl, kls_df = super(CUB_Image_Sentence_ft, self).analyse(data, K=10)
        labels = ['Prior', *[vae.modelName.lower() for vae in self.vaes]]
        plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(runPath, epoch))
        plot_kls_df(kls_df, '{}/kl_distance_{:03d}.png'.format(runPath, epoch))

    def _sent_preprocess(self, sentences):
        """make sure raw data is always passed as dim=2 to avoid argmax.
        last dimension must always be word embedding."""
        if len(sentences.shape) > 2:
            sentences = sentences.argmax(-1).squeeze()
        return [self.vaes[1].fn_trun(s) for s in self.vaes[1].fn_2i(sentences)]

    def _imshow(self, image, caption, i, fig, N):
        """Imshow for Tensor."""
        ax = fig.add_subplot(N // 2, 4, i * 2 + 1)
        ax.axis('off')
        image = image.numpy().transpose((1, 2, 0))  #
        plt.imshow(image)
        ax = fig.add_subplot(N // 2, 4, i * 2 + 2)
        pos = ax.get_position()
        ax.axis('off')
        plt.text(
            x=0.5 * (pos.x0 + pos.x1),
            y=0.5 * (pos.y0 + pos.y1),
            ha='left',
            s='{}'.format(
                ' '.join(self.i2w[str(i)] + '\n' if (n + 1) % 5 == 0
                         else self.i2w[str(i)] for n, i in enumerate(caption))),
            fontsize=6,
            verticalalignment='center',
            horizontalalignment='center'
        )
        return fig
