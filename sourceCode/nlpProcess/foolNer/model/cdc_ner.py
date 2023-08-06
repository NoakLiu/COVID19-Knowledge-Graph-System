from utils.data_process import preprocess, NERDataset
from utils.NERmodels import Bert_MLP, Bert_BiLSTM_CRF, Roberta_MLP, Roberta_BiLSTM_CRF, Ensemble, Bert_CRF, Roberta_CRF
from transformers import BertTokenizer, logging
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils.convert import id2label
import numpy as np
import torch
import argparse
import json
import os

logging.set_verbosity_error()


def vote(models):
    word_list, label_list = preprocess('../../../dataPreProcess/test.txt')

    model_list = []
    for m in models:
        if m == "Bert_MLP":
            model_list.append(Bert_MLP.from_pretrained('../res/Bert_MLP/best').to(device).eval())
        elif m == "Bert_CRF":
            model_list.append(Bert_CRF.from_pretrained('../res/Bert_CRF/best').to(device).eval())
        elif m == "Bert_BiLSTM_CRF":
            model_list.append(Bert_BiLSTM_CRF.from_pretrained('../res/Bert_BiLSTM_CRF/best').to(device).eval())
        elif m == "Roberta_MLP":
            model_list.append(Roberta_MLP.from_pretrained('../res/Roberta_MLP/best').to(device).eval())
        elif m == "Roberta_CRF":
            model_list.append(Roberta_CRF.from_pretrained('../res/Roberta_CRF/best').to(device).eval())
        elif m == "Roberta_BiLSTM_CRF":
            model_list.append(Roberta_BiLSTM_CRF.from_pretrained('../res/Roberta_BiLSTM_CRF/best').to(device).eval())

    tokenizer = BertTokenizer.from_pretrained('../../../../dependencies/bert-base-chinese')
    dataset = NERDataset(word_list, label_list, tokenizer, 512)
    model = Ensemble(model_list).eval()

    data_loader = DataLoader(dataset, batch_size=1)

    pred_tags = []
    indiv_words = []
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            words = batch['words'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(words, attention_mask=attention_mask)
            pred = np.argmax(outputs.detach().cpu().numpy(), axis=2)

            for i, indices in enumerate(pred):
                pred_tags.append([id2label.get(idx)
                                  for idx in indices[: torch.count_nonzero(attention_mask[i]) - 2]])

            # indiv_words.append([tokenizer.convert_ids_to_tokens(i.item())
            #                     for i in words[i][1: torch.count_nonzero(attention_mask[i]) - 1]])
            indiv_words.append("".join(word_list[step][: min(len(word_list[step]), 511)]))

    # pred_tags -> (num_model, seq_len)

    if not os.path.exists('../../../nlpRes'):
        os.makedirs('../../../nlpRes')

    with open('../../../nlpRes/results.txt', 'w', encoding='utf-8') as f:
        for i in range(len(pred_tags)):
            for j in range(len(pred_tags[i])):
                f.write("{} {} {}\n".format(j, indiv_words[i][j], pred_tags[i][j]))

    with open('../../../nlpRes/foolNer.txt', 'w', encoding='utf-8') as f:
        for i in range(len(pred_tags)):
            data = {}
            para = ''.join(indiv_words[i])
            data['text'] = para

            entities = []
            for j in range(len(pred_tags[i])):
                if pred_tags[i][j].startswith('B-'):
                    start_pos = j
                if pred_tags[i][j].startswith('I-') and pred_tags[i][j - 1].startswith('B-'):
                    length = 2
                    for k in range(j + 1, len(pred_tags)):
                        if pred_tags[i][k].startswith('I-'):
                            length += 1
                        else:
                            break
                    entities.append([start_pos, start_pos + length, pred_tags[i][j][2:],
                                     indiv_words[i][start_pos: start_pos + length]])

            data['entities'] = entities
            article = json.dumps(data, ensure_ascii=False)
            f.write("{}\n".format(article))


def indiv(policy):
    word_list, label_list = preprocess('../../../dataPreProcess/test.txt')

    if policy == "Bert_MLP":
        model = Bert_MLP.from_pretrained('../res/Bert_MLP/best')
    elif policy == "Bert_CRF":
        model = Bert_CRF.from_pretrained('../res/Bert_CRF/best')
    elif policy == 'Bert_BiLSTM_CRF':
        model = Bert_BiLSTM_CRF.from_pretrained('../res/Bert_BiLSTM_CRF/best')
    elif policy == "Roberta_MLP":
        model = Roberta_MLP.from_pretrained('../res/Roberta_MLP/best')
    elif policy == "Roberta_CRF":
        model = Roberta_CRF.from_pretrained('../res/Roberta_CRF/best')
    elif policy == "Roberta_BiLSTM_CRF":
        model = Roberta_BiLSTM_CRF.from_pretrained('../res/Roberta_BiLSTM_CRF/best')
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained('../../../../dependencies/bert-base-chinese')
    dataset = NERDataset(word_list, label_list, tokenizer, 512)

    test_data_loader = DataLoader(dataset, batch_size=1)

    model.eval()
    pred_tags = []
    indiv_words = []
    with torch.no_grad():
        for step, batch in enumerate(test_data_loader):
            words = batch['words'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(words, attention_mask=attention_mask)

            if isinstance(model, (Bert_CRF, Bert_BiLSTM_CRF, Roberta_CRF, Roberta_BiLSTM_CRF)):
                seq_mask = [i[1: torch.count_nonzero(i) - 1] for i in attention_mask]
                seq_mask = pad_sequence(seq_mask, batch_first=True).gt(0)
                pred = model.crf.decode(outputs[0], mask=seq_mask)
            else:
                pred = np.argmax(outputs[0].detach().cpu().numpy(), axis=2)

            for i, indices in enumerate(pred):
                pred_tags.append([id2label.get(idx)
                                  for idx in indices[: torch.count_nonzero(attention_mask[i]) - 2]])

            # indiv_words.append([tokenizer.convert_ids_to_tokens(i.item())
            #                     for i in words[i][1: torch.count_nonzero(attention_mask[i]) - 1]])
            indiv_words.append("".join(word_list[step][: min(len(word_list[step]), 511)]))

    if not os.path.exists('../../../nlpRes'):
        os.makedirs('../../../nlpRes')

    with open('../../../nlpRes/results.txt', 'w', encoding='utf-8') as f:
        for i in range(len(pred_tags)):
            for j in range(len(pred_tags[i])):
                f.write("{} {} {}\n".format(j, indiv_words[i][j], pred_tags[i][j]))

    with open('../../../nlpRes/foolNer.txt', 'w', encoding='utf-8') as f:
        for i in range(len(pred_tags)):
            data = {}
            para = ''.join(indiv_words[i])
            data['text'] = para

            entities = []
            for j in range(len(pred_tags[i])):
                if pred_tags[i][j].startswith('B-'):
                    start_pos = j
                if pred_tags[i][j].startswith('I-') and pred_tags[i][j - 1].startswith('B-'):
                    length = 2
                    for k in range(j + 1, len(pred_tags)):
                        if pred_tags[i][k].startswith('I-'):
                            length += 1
                        else:
                            break
                    entities.append([start_pos, start_pos + length, pred_tags[i][j][2:],
                                     indiv_words[i][start_pos: start_pos + length]])

            data['entities'] = entities
            article = json.dumps(data, ensure_ascii=False)
            f.write("{}\n".format(article))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble', type=str, default="", help="此参数优先级更高")
    parser.add_argument('--policy', type=str)
    args = parser.parse_args()

    args.ensemble = args.ensemble.split(',') if len(args.ensemble) else []

    if len(args.ensemble):
        for m in args.ensemble:
            if m not in ['Bert_MLP', 'Bert_CRF', 'Bert_BiLSTM_CRF', 'Roberta_MLP', 'Roberta_CRF', 'Roberta_BiLSTM_CRF']:
                print("--train-ensemble should be chosen in ['Bert_MLP', 'Bert_CRF', 'Bert_BiLSTM_CRF', 'Roberta_MLP', 'Roberta_CRF', 'Roberta_BiLSTM_CRF']")
                exit(0)
    else:
        if args.policy not in ['Bert_MLP', 'Bert_CRF', 'Bert_BiLSTM_CRF', 'Roberta_MLP', 'Roberta_CRF', 'Roberta_BiLSTM_CRF']:
            print("--policy should be in ['Bert_MLP', 'Bert_CRF', 'Bert_BiLSTM_CRF', 'Roberta_MLP', 'Roberta_CRF', 'Roberta_BiLSTM_CRF']")
            exit(0)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if len(args.ensemble):
        vote(args.ensemble)
    else:
        indiv(args.policy)
