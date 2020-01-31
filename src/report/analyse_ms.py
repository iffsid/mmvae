"""Calculate cross and joint coherence of trained model on MNIST-SVHN dataset.
Train and evaluate a linear model for latent space digit classification."""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

# relative import hacks (sorry)
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) # for bash user
os.chdir(parentdir) # for pycharm user

import models
from helper import Latent_Classifier, SVHN_Classifier, MNIST_Classifier
from utils import Logger, Timer


torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Analysing MM-DGM results')
parser.add_argument('--save-dir', type=str, default="",
                    metavar='N', help='save directory of results')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA use')
cmds = parser.parse_args()
runPath = cmds.save_dir

sys.stdout = Logger('{}/ms_acc.log'.format(runPath))
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
B = 256  # rough batch size heuristic
train_loader, test_loader = model.getDataLoaders(B, device=device)
N = len(test_loader.dataset)


def classify_latents(epochs, option):
    model.eval()
    vae = unpack_model(option)
    if '_' not in args.model:
        epochs *= 10  # account for the fact the mnist-svhn has more examples (roughly x10)
    classifier = Latent_Classifier(args.latent_dim, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        total_iters = len(train_loader)
        print('\n====> Epoch: {:03d} '.format(epoch))
        for i, data in enumerate(train_loader):
            # get the inputs
            x, targets = unpack_data_mlp(data, option)
            x, targets = x.to(device), targets.to(device)
            with torch.no_grad():
                qz_x_params = vae.enc(x)
                zs = vae.qz_x(*qz_x_params).rsample()
            optimizer.zero_grad()
            outputs = classifier(zs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if (i + 1) % 1000 == 0:
                print('iteration {:04d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / 1000))
                running_loss = 0.0
    print('Finished Training, calculating test loss...')

    classifier.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x, targets = unpack_data_mlp(data, option)
            x, targets = x.to(device), targets.to(device)
            qz_x_params = vae.enc(x)
            zs = vae.qz_x(*qz_x_params).rsample()
            outputs = classifier(zs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print('The classifier correctly classified {} out of {} examples. Accuracy: '
          '{:.2f}%'.format(correct, total, correct / total * 100))


def _maybe_train_or_load_digit_classifier_img(path, epochs):

    options = [o for o in ['mnist', 'svhn'] if not os.path.exists(path.format(o))]

    for option in options:
        print("Cannot find trained {} digit classifier in {}, training...".
              format(option, path.format(option)))
        classifier = globals()['{}_Classifier'.format(option.upper())]().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            total_iters = len(train_loader)
            print('\n====> Epoch: {:03d} '.format(epoch))
            for i, data in enumerate(train_loader):
                # get the inputs
                x, targets = unpack_data_mlp(data, option)
                x, targets = x.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = classifier(x)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                if (i + 1) % 1000 == 0:
                    print('iteration {:04d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / 1000))
                    running_loss = 0.0
        print('Finished Training, calculating test loss...')

        classifier.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, targets = unpack_data_mlp(data, option)
                x, targets = x.to(device), targets.to(device)
                outputs = classifier(x)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        print('The classifier correctly classified {} out of {} examples. Accuracy: '
              '{:.2f}%'.format(correct, total, correct / total * 100))

        torch.save(classifier.state_dict(), path.format(option))

    mnist_net, svhn_net = MNIST_Classifier().to(device), SVHN_Classifier().to(device)
    mnist_net.load_state_dict(torch.load(path.format('mnist')))
    svhn_net.load_state_dict(torch.load(path.format('svhn')))
    return mnist_net, svhn_net

def cross_coherence(epochs):
    model.eval()

    mnist_net, svhn_net = _maybe_train_or_load_digit_classifier_img("../data/{}_model.pt", epochs=epochs)
    mnist_net.eval()
    svhn_net.eval()

    total = 0
    corr_m = 0
    corr_s = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            mnist, svhn, targets = unpack_data_mlp(data, option='both')
            mnist, svhn, targets = mnist.to(device), svhn.to(device), targets.to(device)
            _, px_zs, _ = model([mnist, svhn], 1)
            mnist_mnist = mnist_net(px_zs[1][0].mean.squeeze(0))
            svhn_svhn = svhn_net(px_zs[0][1].mean.squeeze(0))

            _, pred_m = torch.max(mnist_mnist.data, 1)
            _, pred_s = torch.max(svhn_svhn.data, 1)
            total += targets.size(0)
            corr_m += (pred_m == targets).sum().item()
            corr_s += (pred_s == targets).sum().item()

    print('Cross coherence: \n SVHN -> MNIST {:.2f}% \n MNIST -> SVHN {:.2f}%'.format(
        corr_m / total * 100, corr_s / total * 100))


def joint_coherence():
    model.eval()
    mnist_net, svhn_net = MNIST_Classifier().to(device), SVHN_Classifier().to(device)
    mnist_net.load_state_dict(torch.load('../data/mnist_model.pt'))
    svhn_net.load_state_dict(torch.load('../data/svhn_model.pt'))

    mnist_net.eval()
    svhn_net.eval()

    total = 0
    corr = 0
    with torch.no_grad():
        pzs = model.pz(*model.pz_params).sample([10000])
        mnist = model.vaes[0].dec(pzs)
        svhn = model.vaes[1].dec(pzs)

        mnist_mnist = mnist_net(mnist[0].squeeze(1))
        svhn_svhn = svhn_net(svhn[0].squeeze(1))

        _, pred_m = torch.max(mnist_mnist.data, 1)
        _, pred_s = torch.max(svhn_svhn.data, 1)
        total += pred_m.size(0)
        corr += (pred_m == pred_s).sum().item()

    print('Joint coherence: {:.2f}%'.format(corr / total * 100))


def unpack_data_mlp(dataB, option='both'):
    if len(dataB[0]) == 2:
        if option == 'both':
            return dataB[0][0], dataB[1][0], dataB[1][1]
        elif option == 'svhn':
            return dataB[1][0], dataB[1][1]
        elif option == 'mnist':
            return dataB[0][0], dataB[0][1]
    else:
        return dataB


def unpack_model(option='svhn'):
    if 'mnist_svhn' in args.model:
        return model.vaes[1] if option == 'svhn' else model.vaes[0]
    else:
        return model


if __name__ == '__main__':
    with Timer('MM-VAE analysis') as t:
        print('-' * 25 + 'latent classification accuracy' + '-' * 25)
        print("Calculating latent classification accuracy for single MNIST VAE...")
        classify_latents(epochs=30, option='mnist')
        # #
        print("\n Calculating latent classification accuracy for single SVHN VAE...")
        classify_latents(epochs=30, option='svhn')
        #
        print('\n' + '-' * 45 + 'cross coherence' + '-' * 45)
        cross_coherence(epochs=30)
        #
        print('\n' + '-' * 45 + 'joint coherence' + '-' * 45)
        joint_coherence()
