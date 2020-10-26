import numpy as np

def accuracy(preds, labels):

    rank1, rank5 = 0, 0
    for pred, label in zip(preds, labels):
        # Sort prediction confidence
        pred = np.argsort(pred)[::-1]
        # Rank 5 accuracy
        if label in pred[:5]:
            rank5 += 1
        # Rank 1 accuracy
        if label == pred[0]:
            rank1 += 1
    return rank1 / float(len(preds)), rank5 / float(len(preds))
