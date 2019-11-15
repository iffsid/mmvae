import json
import os
import pickle
from collections import Counter, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import FastText
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy.linalg import eig
from skimage.filters import threshold_yen as threshold


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered."""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def cca(views, k=None, eps=1e-12):
    """Compute (multi-view) CCA

    Args:
        views (list): list of views where each view `v_i` is of size `N x o_i`
        k (int): joint projection dimension | if None, find using Otsu
        eps (float): regulariser [default: 1e-12]

    Returns:
        correlations: correlations along each of the k dimensions
        projections: projection matrices for each view
    """
    V = len(views)  # number of views
    N = views[0].size(0)  # number of observations (same across views)
    os = [v.size(1) for v in views]
    kmax = np.min(os)
    ocum = np.cumsum([0] + os)
    os_sum = sum(os)
    A, B = np.zeros([os_sum, os_sum]), np.zeros([os_sum, os_sum])

    for i in range(V):
        v_i = views[i]
        v_i_bar = v_i - v_i.mean(0).expand_as(v_i)  # centered, N x o_i
        C_ij = (1.0 / (N - 1)) * torch.mm(v_i_bar.t(), v_i_bar)
        # A[ocum[i]:ocum[i + 1], ocum[i]:ocum[i + 1]] = C_ij
        B[ocum[i]:ocum[i + 1], ocum[i]:ocum[i + 1]] = C_ij
        for j in range(i + 1, V):
            v_j = views[j]  # N x o_j
            v_j_bar = v_j - v_j.mean(0).expand_as(v_j)  # centered
            C_ij = (1.0 / (N - 1)) * torch.mm(v_i_bar.t(), v_j_bar)
            A[ocum[i]:ocum[i + 1], ocum[j]:ocum[j + 1]] = C_ij
            A[ocum[j]:ocum[j + 1], ocum[i]:ocum[i + 1]] = C_ij.t()

    A[np.diag_indices_from(A)] += eps
    B[np.diag_indices_from(B)] += eps

    eigenvalues, eigenvectors = eig(A, B)
    # TODO: sanity check to see that all eigenvalues are e+0i
    idx = eigenvalues.argsort()[::-1]  # sort descending
    eigenvalues = eigenvalues[idx]  # arrange in descending order

    if k is None:
        t = threshold(eigenvalues.real[:kmax])
        k = np.abs(np.asarray(eigenvalues.real[0::10]) - t).argmin() * 10  # closest k % 10 == 0 idx
        print('k unspecified, (auto-)choosing:', k)

    eigenvalues = eigenvalues[idx[:k]]
    eigenvectors = eigenvectors[:, idx[:k]]

    correlations = torch.from_numpy(eigenvalues.real).type_as(views[0])
    proj_matrices = torch.split(torch.from_numpy(eigenvectors.real).type_as(views[0]), os)

    return correlations, proj_matrices


def fetch_emb(lenWindow, minOccur, emb_path, vocab_path, RESET):
    if not os.path.exists(emb_path) or RESET:
        with open('../data/cub/text_trainvalclasses.txt', 'r') as file:
            text = file.read()
            sentences = sent_tokenize(text)

        texts = []
        for i, line in enumerate(sentences):
            words = word_tokenize(line)
            texts.append(words)

        model = FastText(size=300, window=lenWindow, min_count=minOccur)
        model.build_vocab(sentences=texts)
        model.train(sentences=texts, total_examples=len(texts), epochs=10)

        with open(vocab_path, 'rb') as file:
            vocab = json.load(file)

        i2w = vocab['i2w']
        base = np.ones((300,), dtype=np.float32)
        emb = [base * (i - 1) for i in range(3)]
        for word in list(i2w.values())[3:]:
            emb.append(model[word])

        emb = np.array(emb)
        with open(emb_path, 'wb') as file:
            pickle.dump(emb, file)

    else:
        with open(emb_path, 'rb') as file:
            emb = pickle.load(file)

    return emb


def fetch_weights(weights_path, vocab_path, RESET, a=1e-3):
    if not os.path.exists(weights_path) or RESET:
        with open('../data/cub/text_trainvalclasses.txt', 'r') as file:
            text = file.read()
            sentences = sent_tokenize(text)
            occ_register = OrderedCounter()

            for i, line in enumerate(sentences):
                words = word_tokenize(line)
                occ_register.update(words)

        with open(vocab_path, 'r') as file:
            vocab = json.load(file)
        w2i = vocab['w2i']
        weights = np.zeros(len(w2i))
        total_occ = sum(list(occ_register.values()))
        exc_occ = 0
        for w, occ in occ_register.items():
            if w in w2i.keys():
                weights[w2i[w]] = a / (a + occ / total_occ)
            else:
                exc_occ += occ
        weights[0] = a / (a + exc_occ / total_occ)

        with open(weights_path, 'wb') as file:
            pickle.dump(weights, file)
    else:
        with open(weights_path, 'rb') as file:
            weights = pickle.load(file)

    return weights


def fetch_pc(emb, weights, train_loader, pc_path, RESET):
    sentences = torch.cat([d[1][0] for d in train_loader]).int()
    emb_dataset = apply_weights(emb, weights, sentences)

    if not os.path.exists(pc_path) or RESET:
        _, _, V = torch.svd(emb_dataset - emb_dataset.mean(dim=0), some=True)
        v = V[:, 0].unsqueeze(-1)
        u = v.mm(v.t())
        with open(pc_path, 'wb') as file:
            pickle.dump(u, file)
    else:
        with open(pc_path, 'rb') as file:
            u = pickle.load(file)
    return u


def apply_weights(emb, weights, data):
    fn_trun = lambda s: s[:np.where(s == 2)[0][0] + 1] if 2 in s else s
    batch_emb = []
    for sent_i in data:
        emb_stacked = torch.stack([emb[idx] for idx in fn_trun(sent_i)])
        weights_stacked = torch.stack([weights[idx] for idx in fn_trun(sent_i)])
        batch_emb.append(torch.sum(emb_stacked * weights_stacked.unsqueeze(-1), dim=0) / emb_stacked.shape[0])

    return torch.stack(batch_emb, dim=0)


def apply_pc(weighted_emb, u):
    return torch.cat([e - torch.matmul(u, e.unsqueeze(-1)).squeeze() for e in weighted_emb.split(2048, 0)])


class Latent_Classifier(nn.Module):
    """ Generate latent parameters for SVHN image data. """

    def __init__(self, in_n, out_n):
        super(Latent_Classifier, self).__init__()
        self.mlp = nn.Linear(in_n, out_n)

    def forward(self, x):
        return self.mlp(x)


class SVHN_Classifier(nn.Module):
    def __init__(self):
        super(SVHN_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class MNIST_Classifier(nn.Module):
    def __init__(self):
        super(MNIST_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
