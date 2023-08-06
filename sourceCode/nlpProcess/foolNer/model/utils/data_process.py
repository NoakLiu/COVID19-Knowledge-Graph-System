import json
from torch.utils.data import Dataset
import torch
from utils.convert import label2id, labels_name


class NERDataset(Dataset):
    def __init__(self, words, labels, tokenizer, max_len):
        self.words = words
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
        self.dataset = self.process(words, labels)

    def process(self, origin_sentences, origin_labels):
        sentences = []
        labels = []
        attention_masks = []
        for sentence in origin_sentences:
            tokens_ids = self.tokenizer.convert_tokens_to_ids(sentence)
            tokens_ids = [101] + tokens_ids + [102]
            attention_mask = [1] * len(tokens_ids)

            tokens_ids = fill_padding(tokens_ids, self.max_len, 0)
            attention_mask = fill_padding(attention_mask, self.max_len, 0)

            sentences.append(tokens_ids)
            attention_masks.append(attention_mask)

        for label_list in origin_labels:
            label_ids = [self.label2id[i] for i in label_list]
            label_ids = fill_padding(label_ids, self.max_len, -1)
            labels.append(label_ids)

        return {
            'sentences': sentences,
            "attention_masks": attention_masks,
            "labels": labels
        }

    def __len__(self):
        return len(self.words)

    def __getitem__(self, item):
        return {
            'words': self.dataset['sentences'][item],
            'attention_mask': self.dataset['attention_masks'][item],
            'labels': self.dataset['labels'][item]
        }


def fill_padding(data, max_len, pad_id):
    if len(data) < max_len:
        pad_len = max_len - len(data)
        padding = [pad_id for _ in range(pad_len)]
        data = torch.tensor(data + padding)
    else:
        data = torch.tensor(data[: max_len])
    return data


def preprocess(file):
    word_list = []
    label_list = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            json_line = json.loads(line.strip())
            text = json_line['text']
            words = list(text)
            label_entities = json_line.get('label', None)
            labels = ['O'] * len(words)

            if label_entities is not None:
                for name, dict in label_entities.items():
                    if name not in labels_name:
                        continue

                    for _, idxs in dict.items():
                        for start_idx, end_idx in idxs:
                            labels[start_idx] = 'B-' + name
                            labels[start_idx + 1: end_idx + 1] = ['I-' + name] * (end_idx - start_idx)

            word_list.append(words)
            label_list.append(labels)

    return word_list, label_list
