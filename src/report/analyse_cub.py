"""Calculate cross and joint coherence of language and image generation on CUB dataset using CCA."""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

# relative import hack (sorry)
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) # for system user
os.chdir(parentdir) # for pycharm user

import models
from utils import Logger, Timer, unpack_data
from helper import cca, fetch_emb, fetch_weights, fetch_pc, apply_weights, apply_pc

# variables
RESET = True
USE_PCA = True
maxSentLen = 32
minOccur = 3
lenEmbedding = 300
lenWindow = 3
fBase = 96
vocab_dir = '../data/cub/oc:{}_sl:{}_s:{}_w:{}'.format(minOccur, maxSentLen, lenEmbedding, lenWindow)
batch_size = 256

# args
torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Analysing MM-DGM results')
parser.add_argument('--save-dir', type=str, default=".",
                    metavar='N', help='save directory of results')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA use')
cmds = parser.parse_args()
runPath = cmds.save_dir
sys.stdout = Logger('{}/analyse.log'.format(runPath))
args = torch.load(runPath + '/args.rar')

# cuda stuff
needs_conversion = cmds.no_cuda and args.cuda
conversion_kwargs = {'map_location': lambda st, loc: st} if needs_conversion else {}
args.cuda = not cmds.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
torch.manual_seed(args.seed)

forward_args = {'drop_modality': True} if args.model == 'mcubISft' else {}

# load trained model
modelC = getattr(models, 'VAE_{}'.format(args.model))
model = modelC(args)
if args.cuda:
    model.cuda()
model.load_state_dict(torch.load(runPath + '/model.rar', **conversion_kwargs), strict=False)
train_loader, test_loader = model.getDataLoaders(batch_size, device=device)
N = len(test_loader.dataset)

# generate word embeddings and sentence weighting
emb_path = os.path.join(vocab_dir, 'cub.emb')
weights_path = os.path.join(vocab_dir, 'cub.weights')
vocab_path = os.path.join(vocab_dir, 'cub.vocab')
pc_path = os.path.join(vocab_dir, 'cub.pc')

emb = fetch_emb(lenWindow, minOccur, emb_path, vocab_path, RESET)
weights = fetch_weights(weights_path, vocab_path, RESET, a=1e-3)
emb = torch.from_numpy(emb).to(device)
weights = torch.from_numpy(weights).to(device).type(emb.dtype)
u = fetch_pc(emb, weights, train_loader, pc_path, RESET)

# set up word to sentence functions
fn_to_emb = lambda data, emb=emb, weights=weights, u=u: \
    apply_pc(apply_weights(emb, weights, data), u)


def calculate_corr(images, embeddings):
    global RESET
    if not os.path.exists(runPath + '/images_mean.pt') or RESET:
        generate_cca_projection()
        RESET = False
    im_mean = torch.load(runPath + '/images_mean.pt')
    emb_mean = torch.load(runPath + '/emb_mean.pt')
    im_proj = torch.load(runPath + '/im_proj.pt')
    emb_proj = torch.load(runPath + '/emb_proj.pt')
    with torch.no_grad():
        corr = F.cosine_similarity((images - im_mean) @ im_proj,
                                   (embeddings - emb_mean) @ emb_proj).mean()
    return corr


def generate_cca_projection():
    images, sentences = [torch.cat(l) for l in zip(*[(d[0], d[1][0]) for d in train_loader])]
    emb = fn_to_emb(sentences.int())
    corr, (im_proj, emb_proj) = cca([images, emb], k=40)
    print("Largest eigen value from CCA: {:.3f}".format(corr[0]))
    torch.save(images.mean(dim=0), runPath + '/images_mean.pt')
    torch.save(emb.mean(dim=0), runPath + '/emb_mean.pt')
    torch.save(im_proj, runPath + '/im_proj.pt')
    torch.save(emb_proj, runPath + '/emb_proj.pt')


def cross_coherence():
    model.eval()
    with torch.no_grad():
        i2t = []
        s2i = []
        gt = []
        for i, dataT in enumerate(test_loader):
            # get the inputs
            images, sentences = unpack_data(dataT, device=device)
            if images.shape[0] != batch_size:
                break
            _, px_zs, _ = model([images, sentences], K=1, **forward_args)
            cross_sentences = px_zs[0][1].mean.argmax(dim=-1).squeeze(0)
            cross_images = px_zs[1][0].mean.squeeze(0)
            # calculate correlation with CCA:
            i2t.append(calculate_corr(images, fn_to_emb(cross_sentences)))
            s2i.append(calculate_corr(cross_images, fn_to_emb(sentences.int())))
            gt.append(calculate_corr(images, fn_to_emb(sentences.int())))
    print("Coherence score: \nground truth {:10.9f}, \nimage to sentence {:10.9f}, "
          "\nsentence to image {:10.9f}".format(sum(gt) / len(gt),
                                                sum(i2t) / len(gt),
                                                sum(s2i) / len(gt)))


def joint_coherence():
    model.eval()
    with torch.no_grad():
        pzs = model.pz(*model.pz_params).sample([1000])
        gen_images = model.vaes[0].dec(pzs)[0].squeeze(1)
        gen_sentences = model.vaes[1].dec(pzs)[0].argmax(dim=-1).squeeze(1)
        score = calculate_corr(gen_images, fn_to_emb(gen_sentences))
        print("joint generation {:10.9f}".format(score))


if __name__ == '__main__':
    with Timer('MM-VAE analysis') as t:
        print('-' * 89)
        cross_coherence()
        print('-' * 89)
        joint_coherence()
