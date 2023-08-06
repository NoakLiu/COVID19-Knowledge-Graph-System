from utils.data_process import preprocess, NERDataset
from utils.NERmodels import Bert_MLP, Bert_BiLSTM_CRF, Roberta_MLP, Roberta_BiLSTM_CRF, Ensemble
from transformers import BertTokenizer, logging
from transformers.utils.notebook import format_time
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.convert import id2label
from utils.metrics import score
import torch.nn as nn
import numpy as np
import torch
import time
import logging as log

logging.set_verbosity_error()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_labels = 3


def test(models, model=None, after_epoch=False):
    words, labels = preprocess(
        '../../../../dataset/Dataset-Epidemiologic-Investigation-COVID19-master/ECR-COVID-19/test.txt')

    tokenizer = BertTokenizer.from_pretrained('../../../../dependencies/bert-base-chinese')
    dataset = NERDataset(words, labels, tokenizer, 512)

    test_data_loader = DataLoader(dataset, batch_size=4)

    model_list = []
    for m in models:
        if m == "Bert_MLP":
            model_list.append(Bert_MLP.from_pretrained('../res/Bert_MLP/best').to(device).eval())
        elif m == "Bert_BiLSTM_CRF":
            model_list.append(Bert_BiLSTM_CRF.from_pretrained('../res/Bert_BiLSTM_CRF/best').to(device).eval())
        elif m == "Roberta_MLP":
            model_list.append(Roberta_MLP.from_pretrained('../res/Roberta_MLP/best').to(device).eval())
        elif m == "Roberta_BiLSTM_CRF":
            model_list.append(Roberta_BiLSTM_CRF.from_pretrained('../res/Roberta_BiLSTM_CRF/best').to(device).eval())

    if not after_epoch:
        model = Ensemble(model_list).to(device)
        model.load_state_dict(torch.load('../res/Ensemble/best.pt'))

    model.eval()
    total_test_loss = 0.
    pred_tags = []
    true_tags = []
    with torch.no_grad():
        for batch in test_data_loader:
            words = batch['words'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(words, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            total_test_loss += loss.item()

            pred = np.argmax(outputs[1].detach().cpu().numpy(), axis=2)
            tags = labels.to('cpu').numpy()

            for i, indices in enumerate(pred):
                pred_tags.extend([id2label.get(idx)
                                  for idx in indices[: torch.count_nonzero(attention_mask[i]) - 2]])
            for i, indices in enumerate(tags):
                true_tags.extend([id2label.get(idx) if idx != -1 else 'O'
                                  for idx in indices[: torch.count_nonzero(attention_mask[i]) - 2]])

    assert len(pred_tags) == len(true_tags)

    metrics = {}
    metrics['loss'] = total_test_loss / len(test_data_loader)
    metrics['f1'], metrics['precision'], metrics['recall'] = score(true_tags, pred_tags)

    log.info('test: precision: {:.2f}%  recall: {:.2f}%  f1: {:.2f}%'.format(metrics['precision'] * 100.,
                                                                                     metrics['recall'] * 100.,
                                                                                     metrics['f1'] * 100.))


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

            pred = np.argmax(outputs[1].detach().cpu().numpy(), axis=2)
            tags = labels.to('cpu').numpy()

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


def train(models, EPOCHS, batch_size, lr, resume=False, trainAll=False):
    word_train, label_train = preprocess(
        '../../../../dataset/Dataset-Epidemiologic-Investigation-COVID19-master/ECR-COVID-19/' + ('train_all.txt' if trainAll else 'train.txt'))
    word_val, label_val = preprocess(
        '../../../../dataset/Dataset-Epidemiologic-Investigation-COVID19-master/ECR-COVID-19/valid.txt')

    tokenizer = BertTokenizer.from_pretrained('../../../../dependencies/bert-base-chinese')
    train_dataset = NERDataset(word_train, label_train, tokenizer, 512)
    val_dataset = NERDataset(word_val, label_val, tokenizer, 512)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size)
    log.info("-----Dataloader Build!-----")

    model_list = []
    for m in models:
        if m == "Bert_MLP":
            model_list.append(Bert_MLP.from_pretrained('../res/Bert_MLP/best').to(device).eval())
        elif m == "Bert_BiLSTM_CRF":
            model_list.append(Bert_BiLSTM_CRF.from_pretrained('../res/Bert_BiLSTM_CRF/best').to(device).eval())
        elif m == "Roberta_MLP":
            model_list.append(Roberta_MLP.from_pretrained('../res/Roberta_MLP/best').to(device).eval())
        elif m == "Roberta_BiLSTM_CRF":
            model_list.append(Roberta_BiLSTM_CRF.from_pretrained('../res/Roberta_BiLSTM_CRF/best').to(device).eval())

    model = Ensemble(model_list, num_labels).to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    train_loss = []
    val_loss = []
    val_f1 = []
    max_val_f1 = 0.
    t0 = time.time()
    for epoch in range(EPOCHS):
        total_train_loss = 0.
        model.train()
        for step, batch in enumerate(train_data_loader):
            words = batch['words'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            model.zero_grad()
            outputs = model(words, attention_mask=attention_mask, labels=labels)

            loss = outputs[0]
            total_train_loss += loss.item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            optimizer.step()

            if step % (len(train_data_loader) // 9) == 0:
                log.info("epoch: {} step: {}/{}   {}".format(epoch, step, len(train_data_loader),
                                                               format_time(time.time() - t0)))

        torch.save(model.state_dict(), '../res/Ensemble/last.pt')

        avg_train_loss = total_train_loss / len(train_data_loader)
        train_loss.append(avg_train_loss)

        # print("Evaluating......")
        train_metrics = evaluate(model, train_data_loader)
        val_metrics = evaluate(model, val_data_loader)

        val_loss.append(val_metrics['loss'])
        val_f1.append(val_metrics['f1'])

        if val_metrics['f1'] > max_val_f1:
            max_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), '../res/Ensemble/best.pt')
            log.info("-----Best Model Saved!-----")

        # print("-----------------------------------------------------------------------------")
        log.info("epoch: {}  train_loss: {}  val_loss: {}\n".format(epoch, avg_train_loss, val_metrics['loss']) +
                 "   train: precision: {:.2f}%  recall: {:.2f}%  f1: {:.2f}%\n".format(train_metrics['precision'] * 100.,
                                                                                     train_metrics['recall'] * 100.,
                                                                                     train_metrics['f1'] * 100.) +
                 "   val: precision: {:.2f}%  recall: {:.2f}%  f1: {:.2f}%".format(val_metrics['precision'] * 100.,
                                                                                   val_metrics['recall'] * 100.,
                                                                                   val_metrics['f1'] * 100.))
        # print("-----------------------------------------------------------------------------")

    log.info('-----Training Finished!-----')
