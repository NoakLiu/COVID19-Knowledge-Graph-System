import torch
from torch.nn.utils.rnn import pad_sequence
from utils.convert import id2label
from utils.metrics import score
import numpy as np
from utils.NERmodels import Bert_CRF, Bert_BiLSTM_CRF, Roberta_CRF, Roberta_BiLSTM_CRF


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def evaluate_ensemble(model, data_loader):
    model.eval()
    pred_tags = []
    true_tags = []
    with torch.no_grad():
        for batch in data_loader:
            words = batch['words'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(words, attention_mask=attention_mask)
            pred = np.argmax(outputs.detach().cpu().numpy(), axis=2)
            tags = labels.cpu().numpy()

            for i, indices in enumerate(pred):
                pred_tags.extend([id2label.get(idx)
                                  for idx in indices[: torch.count_nonzero(attention_mask[i]) - 2]])
            for i, indices in enumerate(tags):
                true_tags.extend([id2label.get(idx) if idx != -1 else 'O'
                                  for idx in indices[: torch.count_nonzero(attention_mask[i]) - 2]])

    assert len(pred_tags) == len(true_tags)

    metrics = {}
    metrics['f1'], metrics['precision'], metrics['recall'] = score(true_tags, pred_tags)

    return metrics


def evaluate(model, data_loader):
    model.eval()
    total_loss = 0.
    pred_tags = []
    true_tags = []
    with torch.no_grad():
        for batch in data_loader:
            words = batch['words'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(words, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            total_loss += loss.item()

            if isinstance(model, (Bert_CRF, Bert_BiLSTM_CRF, Roberta_CRF, Roberta_BiLSTM_CRF)):
                seq_mask = [i[1: torch.count_nonzero(i) - 1] for i in attention_mask]
                seq_mask = pad_sequence(seq_mask, batch_first=True).gt(0)
                pred = model.crf.decode(outputs[1], mask=seq_mask)
            else:
                pred = np.argmax(outputs[1].detach().cpu().numpy(), axis=2)
            tags = labels.cpu().numpy()

            for i, indices in enumerate(pred):
                pred_tags.extend([id2label.get(idx)
                                  for idx in indices[: torch.count_nonzero(attention_mask[i]) - 2]])
            for i, indices in enumerate(tags):
                true_tags.extend([id2label.get(idx) if idx != -1 else 'O'
                                  for idx in indices[: torch.count_nonzero(attention_mask[i]) - 2]])

    assert len(pred_tags) == len(true_tags)

    metrics = {}
    metrics['loss'] = total_loss / len(data_loader)
    metrics['f1'], metrics['precision'], metrics['recall'] = score(true_tags, pred_tags)

    return metrics
