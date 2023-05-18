### Imports ###

import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
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


def load_embeddings_split(src, include_gpt=True):
    embs_train, embs_eval, embs_mimics = [], [], []
    for i, auth in enumerate(os.listdir(src)):
        # Ignore ChatGPT if requested
        if not include_gpt and auth == 'ChatGPT':
            continue
        for doc in sorted(os.listdir(os.path.join(src, auth))):
            # Collect embeddings
            emb = torch.load(os.path.join(src, auth, doc))

            # Store in associated list
            if doc.startswith('known'):
                if len(embs_train) < (i+1)*3:
                    embs_train.append(emb)
                else:
                    embs_eval.append(emb)
            elif doc.startswith('gpt'):
                embs_mimics.append(emb)
    # Stack and return embeddings
    return torch.stack(embs_train).cpu().detach(), torch.stack(embs_eval).cpu().detach(),\
        torch.stack(embs_mimics).cpu().detach()


def full_vs_train(src, eval_set=False, mean_train=False):
    # Load train and eval embeddings
    train_embs, eval_embs, _ = load_embeddings_split(src)

    if eval_set:
        if mean_train:
            # Compare eval set to the training set with author samples averaged
            scores = eval_embs @ train_embs.reshape(-1, 3, 768).mean(dim=1).T
        else:
            # Compare eval set to the training set
            scores = eval_embs @ train_embs.T
    else:
        # Compare train set with train set removing self comparisons
        scores = train_embs @ train_embs.T
        scores = scores.fill_diagonal_(0)

    # Find max author indexes and associated correct labels
    maxes = torch.argmax(scores, dim=-1) // (3 if not eval_set or (eval_set and not mean_train) else 1)
    labels = torch.arange(0, maxes.size()[0], dtype=torch.long) // (1 if eval_set else 3)

    # Return accuracy
    return (maxes == labels).float().mean().item()


def full_vs_train_recall(src, r, mean_train=False):
    # Collect embeddings
    train_embs, eval_embs, _ = load_embeddings_split(src)

    # Calculate scores
    if mean_train:
        scores = eval_embs @ train_embs.reshape(-1, 3, 768).mean(dim=1).T
    else:
        scores = eval_embs @ train_embs.T

    # Sort based on score and find index of correct labels
    order = torch.argsort(scores, dim=-1, descending=True) // (1 if mean_train else 3)
    labels = torch.arange(0, scores.size()[0], dtype=torch.long)
    indexes = [(o == l).nonzero(as_tuple=True)[0][0] for o, l in zip(order, labels)]

    # Find accuracy for a given recall amount
    if isinstance(r, int):
        return sum(np.array(indexes) < r) / scores.size()[0]
    return [sum(np.array(indexes) < r_i) / scores.size()[0] for r_i in r]


def eval_vs_train_recall_plot(src, save=None):
    # Collect embeddings
    train_embs, eval_embs, _ = load_embeddings_split(src)

    # Calculate scores
    scores = eval_embs @ train_embs.T

    # Sort based on score and find index of correct labels
    order = torch.argsort(scores, dim=-1, descending=True) // 3
    labels = torch.arange(0, scores.size()[0], dtype=torch.long)
    indexes = [(o == l).nonzero(as_tuple=True)[0][0] for o, l in zip(order, labels)]

    # Randomize order and calculate indexes of randomization
    order_rand = torch.clone(order)
    for i in range(order_rand.size(0)):
        order_rand[i] = order_rand[i, torch.randperm(order_rand.size(1))]
    indexes_rand = [(o == l).nonzero(as_tuple=True)[0][0] for o, l in zip(order_rand, labels)]

    # Plot expected cumulative probabilities
    _, _, patches = plt.hist(indexes_rand+[3000], scores.size()[1], density=True, histtype='step',
                             cumulative=True, label='Random')
    patches[0].set_xy(patches[0].get_xy()[:-1])

    # Plot evaluated cumulative probabilities
    _, _, patches = plt.hist(indexes+[3000], scores.size()[1], density=True, histtype='step',
                             cumulative=True, label='Contriever')
    patches[0].set_xy(patches[0].get_xy()[:-1])

    # Make plots pretty
    plt.xlabel('Number of Predictions')
    plt.ylabel('Probability')
    plt.title('One vs One:\nProbability of Correct Author Within N Predictions')
    plt.legend()
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()


def eval_vs_mean_train_recall_plot(src, save=None):
    # Collect embeddings
    train_embs, eval_embs, _ = load_embeddings_split(src)

    # Calculate scores
    scores = eval_embs @ train_embs.reshape(-1, 3, 768).mean(dim=1).T

    # Sort based on score and find index of correct labels
    order = torch.argsort(scores, dim=-1, descending=True)
    labels = torch.arange(0, scores.size()[0], dtype=torch.long)
    indexes = [(o == l).nonzero(as_tuple=True)[0][0] for o, l in zip(order, labels)]

    # Randomize order and calculate indexes of randomization
    order_rand = torch.clone(order)
    for i in range(order_rand.size(0)):
        order_rand[i] = order_rand[i, torch.randperm(order_rand.size(1))]
    indexes_rand = [(o == l).nonzero(as_tuple=True)[0][0] for o, l in zip(order_rand, labels)]

    # Plot expected cumulative probabilities
    _, _, patches = plt.hist(indexes_rand + [1000], scores.size()[1], density=True, histtype='step',
                             cumulative=True, label='Random')
    patches[0].set_xy(patches[0].get_xy()[:-1])

    # Plot expected and evaluation cumulative probabilities
    n, bins, patches = plt.hist(indexes + [1000], scores.size()[1], density=True, histtype='step',
                                cumulative=True, label='Contriever')
    patches[0].set_xy(patches[0].get_xy()[:-1])

    # Make plots pretty
    plt.xlabel('Number of Predictions')
    plt.ylabel('Probability')
    plt.title('Author vs Author:\nProbability of Correct Author Within N Predictions')
    plt.legend()
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()


def one_vs_one(src, num=1000):
    # Collect dataset of authors embeddings
    data = Dataset(src)
    count = 0
    for _ in tqdm(range(num)):
        # Choose author #1
        auth = random.randint(0, len(data)-1)
        # Collect random train sample from author #1 for query/anchor
        query = data.get_known(auth)[0]
        query = torch.load(os.path.join(data.data[auth], query))
        # Collect eval sample from author #1 for positive pair
        gold = data.get_known(auth, mask=(3,))[0]
        gold = torch.load(os.path.join(data.data[auth], gold))

        # Collect eval sample from random author #2 for negative pair
        new_auth = random.choice(list(set(range(len(data))) - {auth}))
        neg = data.get_known(new_auth, mask=(3,))[0]
        neg = torch.load(os.path.join(data.data[new_auth], neg))

        # Check if embeddings properly match samples
        if query @ gold > query @ neg:
            count += 1

    # Return accuracy
    return count / num


def auth_vs_auth(src, num=1000, method='max'):
    # Collect dataset of authors embeddings
    data = Dataset(src)
    count = 0
    for _ in tqdm(range(num)):
        # Choose author #1
        auth = random.randint(0, len(data)-1)
        # Collect eval sample from author #1 for query/anchor
        query = data.get_known(auth, mask=(3,))[0]
        query = torch.load(os.path.join(data.data[auth], query))
        # Collect all train samples from author #1 for positive pairs
        golds = data.get_known(auth)
        golds = torch.stack([torch.load(os.path.join(data.data[auth], gold)) for gold in golds])

        # Collect all train samples from random author #2 for negative pairs
        new_auth = random.choice(list(set(range(len(data))) - {auth}))
        negs = data.get_known(new_auth)
        negs = torch.stack([torch.load(os.path.join(data.data[new_auth], neg)) for neg in negs])

        # Evaluate embeddings using certain method
        if method == 'max' and max(query @ golds.T) > max(query @ negs.T):
            # Which author has the highest scoring pair with anchor
            count += 1
        elif method == 'mean' and torch.mean(query @ golds.T) > torch.mean(query @ negs.T):
            # Which author's mean score is highest
            count += 1
        elif method == 'min' and min(query @ golds.T) > min(query @ negs.T):
            # Which author is the highest score based on their worst score
            count += 1

    # Return accuracy
    return count / num


def one_vs_gpt(src, num=1000):
    # Collect training samples from ChatGPT as positive pairs
    golds = [torch.load(os.path.join(src, 'ChatGPT', doc))
             for doc in os.listdir(os.path.join(src, 'ChatGPT'))[:3]]

    # Collect authors without ChatGPT
    auths = os.listdir(src)
    auths.remove('ChatGPT')
    count = 0
    for auth in random.choices(auths, k=num):
        # Collect ChatGPT mimic embedding for the chosen author
        query = torch.load(glob(os.path.join(src, auth, 'gpt_*.txt'))[0])
        # Collect a training sample from chosen author as the negative sample
        docs = sorted(glob(os.path.join(src, auth, 'known*.txt')))[:3]
        random.shuffle(docs)
        neg = torch.load(docs[0])

        # Evaluate if embeddings are accuracy
        if query @ golds[random.randint(0, len(golds)-1)] > query @ neg:
            count += 1
    # Return accuracy
    return count / num


def auth_vs_gpt(src, num=1000, method='max'):
    # Collect training samples from ChatGPT as positive pairs
    golds = torch.stack([torch.load(os.path.join(src, 'ChatGPT', doc))
                         for doc in os.listdir(os.path.join(src, 'ChatGPT'))[:3]])

    # Collect authors without ChatGPT
    auths = os.listdir(src)
    auths.remove('ChatGPT')
    count = 0
    for auth in random.choices(auths, k=num):
        # Collect ChatGPT mimic embedding for the chosen author
        query = torch.load(glob(os.path.join(src, auth, 'gpt_*.txt'))[0])
        # Collect training samples from chosen author as negative samples
        docs = sorted(glob(os.path.join(src, auth, 'known*.txt')))[:3]
        negs = torch.stack([torch.load(doc) for doc in docs])

        # Evaluate if embeddings are accuracy using the chosen method
        if method == 'max' and max(query @ golds.T) > max(query @ negs.T):
            # Which author has the highest scoring pair with anchor
            count += 1
        elif method == 'mean' and torch.mean(query @ golds.T) > torch.mean(query @ negs.T):
            # Which author's mean score is highest
            count += 1
        elif method == 'min' and min(query @ golds.T) > min(query @ negs.T):
            # Which author is the highest score based on their worst score
            count += 1

    # Return accuracy
    return count / num


### Run Code ###

if __name__ == '__main__':
    embedding_src = './Data/RedditEmbeddings2'
    print(f"One vs One accuracy: {one_vs_one(embedding_src, num=10000):.2%}")
    print(f"Author vs Author accuracy: {auth_vs_auth(embedding_src, num=10000):.2%}")
    # print(f"Author Mean vs Author Mean accuracy: {auth_vs_auth(embedding_src, method='mean'):.2%}")
    # print(f"Author Furthest vs Author Furthest accuracy: {auth_vs_auth(embedding_src, method='min'):.2%}")
    print(f"One vs ChatGPT accuracy: {one_vs_gpt(embedding_src, num=10000):.2%}")
    print(f"Author vs ChatGPT accuracy: {auth_vs_gpt(embedding_src, num=10000):.2%}")
    print(f"Author Mean vs ChatGPT Mean accuracy: {auth_vs_gpt(embedding_src, method='mean'):.2%}")
    print(f"Author Furthest vs ChatGPT Furthest accuracy: {auth_vs_gpt(embedding_src, method='min'):.2%}")
    print(f"Train set vs Train set accuracy: {full_vs_train(embedding_src):.2%}")
    print(f"Eval set vs Train set accuracy: {full_vs_train(embedding_src, eval_set=True):.2%}")
    print(f"Eval set vs Mean Train set accuracy: {full_vs_train(embedding_src, eval_set=True, mean_train=True):.2%}")
    r5, r20, r100 = full_vs_train_recall(embedding_src, [5, 20, 100])
    print(f"Eval set vs Train set recall @ 5: {r5:.2%}, 20: {r20:.2%}, 100: {r100:.2%}")
    r5, r20, r100 = full_vs_train_recall(embedding_src, [5, 20, 100], mean_train=True)
    print(f"Eval set vs Mean Train set recall @ 5: {r5:.2%}, 20: {r20:.2%}, 100: {r100:.2%}")
    eval_vs_train_recall_plot(embedding_src, save='./eval_vs_train.png')
    eval_vs_mean_train_recall_plot(embedding_src, save='./eval_vs_mean_train.png')
