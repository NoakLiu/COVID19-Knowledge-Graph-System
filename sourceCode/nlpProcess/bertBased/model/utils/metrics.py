from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.scheme import IOBES

# test_seq1 = ['O', 'O', 'B-LocalID', 'I-LocalID', 'I-LocalID', 'O', 'B-LocalID', 'I-LocalID']
# test_seq2 = ['O', 'S-LocalID', 'O', 'B-LocalID', 'I-LocalID', 'I-LocalID', 'O', 'B-LocalID', 'I-LocalID', 'I-LocalID']
#
# true_seq = ['O', 'O', 'B-LocalID', 'I-LocalID', 'I-LocalID', 'O', 'S-LocalID', 'O']


def score(true_tags, pred_tags):
    precision = precision_score(true_tags, pred_tags, mode='strict', scheme=IOBES)
    recall = recall_score(true_tags, pred_tags, mode='strict', scheme=IOBES)
    f1 = f1_score(true_tags, pred_tags, mode='strict', scheme=IOBES)

    return f1, precision, recall
