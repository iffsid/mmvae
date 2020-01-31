"""Calculate data marginal likelihood p(x) evaluated on the trained generative model."""
import os
import sys
import argparse

import numpy as np
import torch
from torchvision.utils import save_image

# relative import hacks (sorry)
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) # for bash user
os.chdir(parentdir) # for pycharm user

import models
from utils import Logger, Timer, unpack_data, log_mean_exp

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Analysing MM-DGM results')
parser.add_argument('--save-dir', type=str, default="",
                    metavar='N', help='save directory of results')
parser.add_argument('--iwae-samples', type=int, default=1000, metavar='I',
                    help='number of samples to estimate marginal log likelihood (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA use')
cmds = parser.parse_args()
runPath = cmds.save_dir

sys.stdout = Logger('{}/llik.log'.format(runPath))
args = torch.load(runPath + '/args.rar')

# cuda stuff
needs_conversion = cmds.no_cuda and args.cuda
conversion_kwargs = {'map_location': lambda st, loc: st} if needs_conversion else {}
args.cuda = not cmds.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
torch.manual_seed(args.seed)

modelC = getattr(models, 'VAE_{}'.format(args.model))
model = modelC(args)
if args.cuda:
    model.cuda()

model.load_state_dict(torch.load(runPath + '/model.rar', **conversion_kwargs), strict=False)
B = 12000 // cmds.iwae_samples  # rough batch size heuristic
train_loader, test_loader = model.getDataLoaders(B, device=device)
N = len(test_loader.dataset)


def m_iwae(qz_xs, px_zs, zss, x):
    """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised"""
    lws = []
    for r, qz_x in enumerate(qz_xs):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return log_mean_exp(torch.cat(lws)).sum()


def iwae(qz_x, px_z, zs, x):
    """IWAE estimate for log p_\theta(x) -- fully vectorised."""
    lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) * model.llik_scaling
    lqz_x = qz_x.log_prob(zs).sum(-1)
    return log_mean_exp(lpz + lpx_z.sum(-1) - lqz_x).sum()


@torch.no_grad()
def joint_elbo(K):
    model.eval()
    llik = 0
    obj = locals()[('m_' if hasattr(model, 'vaes') else '') + 'iwae']()
    for dataT in test_loader:
        data = unpack_data(dataT, device=device)
        llik += obj(model, data, K).item()
    print('Marginal Log Likelihood of joint {} (IWAE, K = {}): {:.4f}'
          .format(model.modelName, K, llik / N))


def cross_iwaes(qz_xs, px_zs, zss, x):
    lws = []
    for e, _px_zs in enumerate(px_zs):  # rows are encoders
        lpz = model.pz(*model.pz_params).log_prob(zss[e]).sum(-1)
        lqz_x = qz_xs[e].log_prob(zss[e]).sum(-1)
        _lpx_zs = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1).sum(-1)
                   for d, px_z in enumerate(_px_zs)]
        lws.append([log_mean_exp(_lpx_z + lpz - lqz_x).sum() for _lpx_z in _lpx_zs])
    return lws


def individual_iwaes(qz_xs, px_zs, zss, x):
    lws = []
    for d, _px_zs in enumerate(np.array(px_zs).T):  # rows are decoders now
        lw = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1).sum(-1)
              + model.pz(*model.pz_params).log_prob(zss[e]).sum(-1)
              - log_mean_exp(torch.stack([qz_x.log_prob(zss[e]).sum(-1) for qz_x in qz_xs]))
              for e, px_z in enumerate(_px_zs)]
        lw = torch.cat(lw)
        lws.append(log_mean_exp(lw).sum())
    return lws


@torch.no_grad()
def m_llik_eval(K):
    model.eval()
    llik_joint = 0
    llik_synergy = np.array([0 for _ in model.vaes])
    lliks_cross = np.array([[0 for _ in model.vaes] for _ in model.vaes])
    for dataT in test_loader:
        data = unpack_data(dataT, device=device)
        qz_xs, px_zs, zss = model(data, K)
        objs = individual_iwaes(qz_xs, px_zs, zss, data)
        objs_cross = cross_iwaes(qz_xs, px_zs, zss, data)
        llik_joint += m_iwae(qz_xs, px_zs, zss, data)
        llik_synergy = llik_synergy + np.array(objs)
        lliks_cross = lliks_cross + np.array(objs_cross)

    print('Marginal Log Likelihood of joint {} (IWAE, K = {}): {:.4f}'
          .format(model.modelName, K, llik_joint / N))
    print('-' * 89)

    for i, llik in enumerate(llik_synergy):
        print('Marginal Log Likelihood of {} from {} (IWAE, K = {}): {:.4f}'
              .format(model.vaes[i].modelName, model.modelName, K, (llik / N).item()))
    print('-' * 89)

    for e, _lliks_cross in enumerate(lliks_cross):
        for d, llik_cross in enumerate(_lliks_cross):
            print('Marginal Log Likelihood of {} from {} (IWAE, K = {}): {:.4f}'
                  .format(model.vaes[d].modelName, model.vaes[e].modelName, K, (llik_cross / N).item()))
    print('-' * 89)


@torch.no_grad()
def llik_eval(K):
    model.eval()
    llik_joint = 0
    for dataT in test_loader:
        data = unpack_data(dataT, device=device)
        qz_xs, px_zs, zss = model(data, K)
        llik_joint += iwae(qz_xs, px_zs, zss, data)
    print('Marginal Log Likelihood of joint {} (IWAE, K = {}): {:.4f}'
          .format(model.modelName, K, llik_joint / N))


@torch.no_grad()
def generate_sparse(D, steps, J):
    """generate `steps` perturbations for all `D` latent dimensions on `J` datapoints. """
    model.eval()
    for i, dataT in enumerate(test_loader):
        data = unpack_data(dataT, require_length=(args.projection == 'Sft'), device=device)
        qz_xs, _, zss = model(data, args.K)
        for i, (qz_x, zs) in enumerate(zip(qz_xs, zss)):
            embs = []
            # for delta in torch.linspace(0.01, 0.99, steps=steps):
            for delta in torch.linspace(-5, 5, steps=steps):
                for d in range(D):
                    mod_emb = qz_x.mean + torch.zeros_like(qz_x.mean)
                    mod_emb[:, d] += model.vaes[i].pz(*model.vaes[i].pz_params).stddev[:, d] * delta
                    embs.append(mod_emb)
            embs = torch.stack(embs).transpose(0, 1).contiguous()
            for r in range(2):
                samples = model.vaes[r].px_z(*model.vaes[r].dec(embs.view(-1, D)[:((J) * steps * D)])).mean
                save_image(samples.cpu(), os.path.join(runPath, 'latent-traversals-{}x{}.png'.format(i, r)), nrow=D)
        break


if __name__ == '__main__':
    with Timer('MM-VAE analysis') as t:
        # likelihood evaluation
        print('-' * 89)
        eval = locals()[('m_' if hasattr(model, 'vaes') else '') + 'llik_eval']
        eval(cmds.iwae_samples)
        print('-' * 89)
