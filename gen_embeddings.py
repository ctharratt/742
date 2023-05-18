### Imports ###

from tqdm import tqdm
import torch
import sys
import os

from src.contriever import load_retriever


### Functions ###

def gen_embedding(src, encoder, tokenizer):
    # Collect documents in the src folder
    docs = os.listdir(src)
    passages = []
    for doc in docs:
        # Open and save each passage
        with open(os.path.join(src, doc), encoding="utf8") as fin:
            passages.append(fin.read())
    # Convert passages into tokens on the GPU
    tokens = tokenizer(passages, padding=True, truncation=True, return_tensors="pt")
    tokens = {k: v.cuda() for k, v in tokens.items()}
    # Return the list of documents and associated embeddings
    return docs, encoder(**tokens, normalize=True)


def data_loop(src, dist, encoder, tokenizer):
    # Create output folder if needed
    if not os.path.exists(dist):
        os.makedirs(dist)

    # Search through all author folders
    for auth in tqdm(os.listdir(src), file=sys.stdout):
        # Ignore non-directories and folders that have already been processed
        if os.path.isdir(os.path.join(src, auth)) and not os.path.exists(os.path.join(dist, auth)):
            os.makedirs(os.path.join(dist, auth))
            # Generate embeddings and save
            docs, embeddings = gen_embedding(os.path.join(src, auth), encoder, tokenizer)
            for d, e in zip(docs, embeddings):
                torch.save(e, os.path.join(dist, auth, d))


### Run Code ###

if __name__ == '__main__':
    # Path to saved contriever model
    model_path = './checkpoint/exp1/checkpoint/step-5000'
    if len(sys.argv) >= 2:
        model_path = sys.argv[1]

    # Load model and tokenize and move model to GPU
    model, tokenizer, _ = load_retriever(model_path)
    model = model.cuda()
    model.eval()

    # Generate embeddings for dataset
    data_loop('./Data/RedditData', './Data/RedditEmbeddings2', model, tokenizer)

