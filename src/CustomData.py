### Imports ###

import os
import random
from torch.utils import data
from src import normalize_text
from glob import glob


### Dataset Class ###

class Dataset(data.Dataset):
    def __init__(self,
                 datapaths,
                 negative_ctxs=1,
                 negative_hard_ratio=0.0,
                 negative_hard_min_idx=0,
                 training=False,
                 global_rank=-1,
                 world_size=-1,
                 maxload=None,
                 normalize=False,
                 include_mimic=False
                 ):
        if isinstance(datapaths, str):
            datapaths = [datapaths]

        # Collect authors in dataset
        self.data = []
        for datapath in datapaths:
            for author in os.listdir(datapath):
                auth_path = os.path.join(datapath, author)
                if os.path.isdir(auth_path):
                    self.data.append(auth_path)
        # Shuffle dataset
        random.shuffle(self.data)

        self.training = training
        self.negatives = negative_ctxs
        self.include_mimic = include_mimic
        self.normalize_fn = normalize_text.normalize if normalize_text else lambda x: x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if not self.training:
            # Load sample from training set as query/anchor
            with open(os.path.join(self.data[index], self.get_known(index)[0]), encoding="utf8") as fin:
                query = fin.read()
            # Load eval sample as positive pair
            with open(os.path.join(self.data[index], self.get_known(index, mask=[3, ])[0]), encoding="utf8") as fin:
                gold = fin.read()

            # Choose random other author and load eval sample as negative pair
            negs = []
            auths = random.sample(list(set(range(len(self.data))) - {index}), k=self.negatives)
            for auth in auths:
                with open(os.path.join(self.data[auth], self.get_known(auth, mask=[3, ])[0]), encoding="utf8") as fin:
                    negs.append(fin.read())

            # Add ChatGPT mimic as negative sample
            if self.include_mimic:
                path = glob(os.path.join(self.data[index], 'gpt_*.txt'))
                if len(path) > 0:
                    with open(path[0], encoding="utf8") as fin:
                        negs.append(fin.read())
        else:
            # Collect random query/anchor and positive pair from train set
            pos_pair = self.get_known(index)[:2]
            with open(os.path.join(self.data[index], pos_pair[0]), encoding="utf8") as fin:
                query = fin.read()
            with open(os.path.join(self.data[index], pos_pair[1]), encoding="utf8") as fin:
                gold = fin.read()

            # Collect negative pair(s) for random other author
            negs = []
            auths = random.sample(list(set(range(len(self.data))) - {index}), k=self.negatives)
            for auth in auths:
                f = self.get_known(auth)[0]
                with open(os.path.join(self.data[auth], f), encoding="utf8") as fin:
                    negs.append(fin.read())

        # Compile into dictionary with normalization function
        example = {
            'query': self.normalize_fn(query),
            'gold': self.normalize_fn(gold),
            'negatives': [self.normalize_fn(neg) for neg in negs]
        }

        return example

    def get_known(self, idx, mask=(0, 1, 2)):
        # Get known data with randomization from train set (0, 1, 2) or eval set (3, )
        files = os.listdir(self.data[idx])
        files = [f for f in files if f.startswith("known")]  # Only use known files
        files = sorted(files)
        files = [files[i] for i in mask]
        random.shuffle(files)
        return files


# Collator from finetune_data.py in original Contriever codebase
class Collator(object):
    def __init__(self, tokenizer, passage_maxlength=200):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength

    def __call__(self, batch):
        queries = [ex["query"] for ex in batch]
        golds = [ex["gold"] for ex in batch]
        negs = [item for ex in batch for item in ex["negatives"]]
        allpassages = golds + negs

        qout = self.tokenizer.batch_encode_plus(
            queries,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        kout = self.tokenizer.batch_encode_plus(
            allpassages,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        q_tokens, q_mask = qout["input_ids"], qout["attention_mask"].bool()
        k_tokens, k_mask = kout["input_ids"], kout["attention_mask"].bool()

        g_tokens, g_mask = k_tokens[: len(golds)], k_mask[: len(golds)]
        n_tokens, n_mask = k_tokens[len(golds):], k_mask[len(golds):]

        batch = {
            "q_tokens": q_tokens,
            "q_mask": q_mask,
            "k_tokens": k_tokens,
            "k_mask": k_mask,
            "g_tokens": g_tokens,
            "g_mask": g_mask,
            "n_tokens": n_tokens,
            "n_mask": n_mask,
        }

        return batch


### Test Dataset ###

if __name__ == "__main__":
    dataset = Dataset(['../Data/RedditData'])
    print(len(dataset))
    print(dataset[0])
    dataset = Dataset(['../Data/RedditData'], training=True)
    print(len(dataset))
    print(dataset[0])
