import faiss
from torch.nn import functional as F
from torch.nn import CosineSimilarity
import torch
from nltk.tokenize import sent_tokenize

def compute_scores(candididate_sentences, candidate_embeddings,
        reference_sentences, reference_embeddings, k=10, is_self=False):
    d = candidate_embeddings.shape[1]
    # similarity = CosineSimilarity(dim=0)
    # resources = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(d)   # build the index

    # TODO: normalize all embeddings before ADDING and SEARCHING!
    candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=1)
    reference_embeddings = F.normalize(reference_embeddings, p=2, dim=1)

    # METRIC_INNER_PRODUCT
    # add the reference embeddings to the set
    index.add(reference_embeddings.cpu().numpy())
    # print(candidate_embeddings.
    assert index.ntotal==reference_embeddings.shape[0]
    # assert candidate_embeddings.shape[0]==reference_embeddings.shape[0]

    # do an assert that the search is 10x the input size
    # the rows are the candidate embeddings, columns are IDs of reference embeddings.
    distances, ids = index.search(candidate_embeddings.cpu().numpy(), k=k)
    if is_self:
        # cut off the nearest neighbor because that is the search vector
        ids = ids[:, 1:]
        k -= 1

    # convert the IDs back into embeddings
    ids = ids.reshape(-1, 1)
    nearest_neighbors = reference_embeddings[ids]
    nearest_neighbors = nearest_neighbors.reshape(candidate_embeddings.shape[0], k, d)

    # do a sanity check by verifying that my reshaped distances are the same that are expect.
    # similarities = [similarity(nearest_neighbors[0, i], candidate_embeddings[0]) for i in range(k)]
    # print(distances[0])
    # print(similarities)

    # mean pool
    nearest_neighbors = nearest_neighbors.mean(dim=1)
    similarities = (candidate_embeddings*nearest_neighbors).sum(-1)
    # similarities = torch.dist(candidate_embeddings, nearest_neighbors)
    return torch.mean(similarities, dim=0)

def encode_sentences(model, sentences, batch_size=500, device="cuda"):
    # tokenize the references into sentences
    tokenized_sentences = []
    n_sent = len(sentences)
    for sentence in sentences:
        tokens = sent_tokenize(sentence)
        if len(tokens) > 0:
            tokenized_sentences.append(tokens)
    # print(tokenized_sentences)
    # input("Waiting...")
    max_len = len(max(tokenized_sentences, key=len))
    padding, tokens = [], []
    for sent_tokens in tokenized_sentences:
        sent_tokens, sent_padding = pad(sent_tokens, max_len)
        tokens.extend(sent_tokens)
        padding.extend(sent_padding)
        
    # print(len(tokens), len(tokens[0]), len(sentences), max_len)
    # assert len(tokens)==len(sentences)*max_len

    embeddings = model.encode(tokens, convert_to_numpy=False, batch_size=batch_size)
    embeddings = torch.stack(embeddings)

    d = embeddings.shape[1]
    # TODO: verify that the embeddings are the embedding for the proper sentences.
    padding = torch.tensor(padding).to(device).unsqueeze(1)
    padding_mask = padding.repeat(1, d)
    embeddings = embeddings*padding_mask
    embeddings = embeddings.reshape(-1, max_len, d)
    padding = padding.reshape(-1, max_len, 1)
    # padding = padding!=1  # select non-padded values
    embeddings = (embeddings).sum(dim=1) / padding.sum(dim=1)
    return embeddings

def pad(l, n):
    # print("List:", l)
    # print(["a" for _ in range(n-len(l))])
    # print()
    # Padding is 1 if the sentence is not padded
    tokens = l + ["a" for _ in range(n-len(l))]
    paddings = [1 for _ in range(len(l))] + [0 for _ in range(n - len(l))]
    return tokens, paddings
