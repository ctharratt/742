### Imports ###

import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from glob import glob

from src.CustomData import Dataset


### Functions ###

def load_embeddings(src):
    embs = []
    for auth in os.listdir(src):
        for doc in os.listdir(os.path.join(src, auth)):
            if doc.startswith('known'):
                embs.append(torch.load(os.path.join(src, auth, doc)))
    return torch.stack(embs).cpu().detach()


def all_vs_all(src):
    embeddings = load_embeddings(src)
    dists = embeddings @ embeddings.T
    # heatmap = (dists - np.percentile(dists, 5)) / (np.percentile(dists, 95) - np.percentile(dists, 5))
    # heatmap = np.clip(heatmap, 0, 1)
    # plt.plot(np.percentile(heatmap, list(range(0, 101, 1))), list(range(0, 101, 1)))
    # plt.hist(np.reshape(heatmap, -1))
    # plt.imshow(heatmap)
    # plt.colorbar()
    # plt.show()

    ranks_train = []
    # ranks_gpt = []
    for i in range((embeddings.size()[0] // 4)):
        for j in range(0, 3):
            rankings = np.argsort(dists[i * 4 + j])
            ranks_train += [np.where(rankings == i * 4 + k)[0][0] for k in range(j + 1, 4)]
            # ranks_gpt.append(np.where(rankings == i*6)[0][0])
    print(np.percentile(ranks_train, list(range(0, 101, 10))))
    # print(np.percentile(ranks_gpt, list(range(0, 101, 10))))


def one_vs_one(src, num=10000):
    data = Dataset(src)
    count = 0
    for _ in tqdm(range(num)):
        auth = random.randint(0, len(data)-1)
        query = data.get_known(auth)[0]
        query = torch.load(os.path.join(data.data[auth], query))
        gold = data.get_known(auth, mask=(3,))[0]
        gold = torch.load(os.path.join(data.data[auth], gold))

        new_auth = random.choice(list(set(range(len(data))) - {auth}))
        neg = data.get_known(new_auth, mask=(3,))[0]
        neg = torch.load(os.path.join(data.data[new_auth], neg))

        if query @ gold > query @ neg:
            count += 1

    return count / num


def auth_vs_auth(src, num=10000, method='max'):
    data = Dataset(src)
    count = 0
    for _ in tqdm(range(num)):
        auth = random.randint(0, len(data)-1)
        query = data.get_known(auth, mask=(3,))[0]
        query = torch.load(os.path.join(data.data[auth], query))
        golds = data.get_known(auth)
        golds = torch.stack([torch.load(os.path.join(data.data[auth], gold)) for gold in golds])

        new_auth = random.choice(list(set(range(len(data))) - {auth}))
        negs = data.get_known(new_auth)
        negs = torch.stack([torch.load(os.path.join(data.data[new_auth], neg)) for neg in negs])

        if method == 'max' and max(query @ golds.T) > max(query @ negs.T):
            count += 1
        elif method == 'mean' and torch.mean(query @ golds.T) > torch.mean(query @ negs.T):
            count += 1
        elif method == 'min' and min(query @ golds.T) > min(query @ negs.T):
            count += 1

    return count / num


def one_vs_gpt(src, num=1000):
    golds = [torch.load(os.path.join(src, 'ChatGPT', doc)) for doc in os.listdir(os.path.join(src, 'ChatGPT'))]

    count = 0
    auths = os.listdir(src)
    auths.remove('ChatGPT')
    for auth in random.choices(auths, k=num):
        query = torch.load(glob(os.path.join(src, auth, 'gpt_*.txt'))[0])
        docs = sorted(glob(os.path.join(src, auth, 'known*.txt')))[:3]
        neg = torch.load(docs[0])

        if query @ golds[random.randint(0, len(golds)-1)] > query @ neg:
            count += 1
    return count / num


def auth_vs_gpt(src, num=1000, method='mean'):
    golds = torch.stack([torch.load(os.path.join(src, 'ChatGPT', doc))
                         for doc in os.listdir(os.path.join(src, 'ChatGPT'))])

    count = 0
    auths = os.listdir(src)
    auths.remove('ChatGPT')
    for auth in random.choices(auths, k=num):
        query = torch.load(glob(os.path.join(src, auth, 'gpt_*.txt'))[0])
        docs = sorted(glob(os.path.join(src, auth, 'known*.txt')))[:3]
        negs = torch.stack([torch.load(doc) for doc in docs])

        if method == 'max' and max(query @ golds.T) > max(query @ negs.T):
            count += 1
        elif method == 'mean' and torch.mean(query @ golds.T) > torch.mean(query @ negs.T):
            count += 1
        elif method == 'min' and min(query @ golds.T) > min(query @ negs.T):
            count += 1

    return count / num


### Run Code ###

if __name__ == '__main__':
    embedding_src = './Data/RedditEmbeddings2'
    # print(f"One vs One accuracy: {one_vs_one(embedding_src):.2%}")
    # print(f"Author vs Author accuracy: {auth_vs_auth(embedding_src):.2%}")
    # print(f"Author Mean vs Author Mean accuracy: {auth_vs_auth(embedding_src, method='mean'):.2%}")
    # print(f"Author Furthest vs Author Furthest accuracy: {auth_vs_auth(embedding_src, method='min'):.2%}")
    print(f"One vs ChatGPT accuracy: {one_vs_gpt(embedding_src):.2%}")
    print(f"Author vs ChatGPT accuracy: {auth_vs_gpt(embedding_src):.2%}")
    print(f"Author Mean vs ChatGPT Mean accuracy: {auth_vs_gpt(embedding_src, method='mean'):.2%}")
    print(f"Author Furthest vs ChatGPT Furthest accuracy: {auth_vs_gpt(embedding_src, method='min'):.2%}")
