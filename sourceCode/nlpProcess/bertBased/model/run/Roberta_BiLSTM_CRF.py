from utils.data_process import preprocess, NERDataset
from utils.NERmodels import Roberta_BiLSTM_CRF
from transformers import BertTokenizer, logging
from transformers.utils.notebook import format_time
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW
from utils.utils import evaluate
import torch.nn as nn
import torch
import time
import logging as log

logging.set_verbosity_error()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def test(model=None, test_data_loader=None):
    if model is None:
        model = Roberta_BiLSTM_CRF.from_pretrained('../res/Roberta_BiLSTM_CRF/best').to(device)

    if test_data_loader is None:
        words, labels = preprocess(
            '../../../../dataset/Dataset-Epidemiologic-Investigation-COVID19-master/ECR-COVID-19/test.txt')
        tokenizer = BertTokenizer.from_pretrained('../../../../dependencies/bert-base-chinese')
        dataset = NERDataset(words, labels, tokenizer, 512)
        test_data_loader = DataLoader(dataset, batch_size=4)

    metrics = evaluate(model, test_data_loader)

    log.info('   test: precision: {:.2f}%  recall: {:.2f}%  f1: {:.2f}%'.format(metrics['precision'] * 100.,
                                                                                metrics['recall'] * 100.,
                                                                                metrics['f1'] * 100.))


def train(EPOCHS, batch_size, lr, full_fine_tuning=False, resume=False, trainAll=False):
    word_train, label_train = preprocess(
        '../../../../dataset/Dataset-Epidemiologic-Investigation-COVID19-master/ECR-COVID-19/' + (
            'train_all.txt' if trainAll else 'train.txt'))
    word_val, label_val = preprocess(
        '../../../../dataset/Dataset-Epidemiologic-Investigation-COVID19-master/ECR-COVID-19/valid.txt')

    word_test, label_test = preprocess(
        '../../../../dataset/Dataset-Epidemiologic-Investigation-COVID19-master/ECR-COVID-19/test.txt')

    tokenizer = BertTokenizer.from_pretrained('../../../../dependencies/bert-base-chinese')
    train_dataset = NERDataset(word_train, label_train, tokenizer, 512)
    val_dataset = NERDataset(word_val, label_val, tokenizer, 512)
    test_dataset = NERDataset(word_test, label_test, tokenizer, 512)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)
    log.info("-----Dataloader Build!-----")

    if not resume:
        model = Roberta_BiLSTM_CRF.from_pretrained('../../../../dependencies/chinese-roberta-wwm-ext').to(device)
    else:
        log.info('-----Resume training from ../res/Roberta_BiLSTM_CRF/best!-----')
        model = Roberta_BiLSTM_CRF.from_pretrained('../res/Roberta_BiLSTM_CRF/best').to(device)

    if full_fine_tuning:
        optimizer_grouped_parameters = [
            {'params': model.roberta.parameters()},
            {'params': model.bilstm.parameters(), 'lr': lr * 5},
            {'params': model.classifier.parameters(), 'lr': lr * 5},
            {'params': model.crf.parameters(), 'lr': lr * 10}
        ]
    else:
        bert_optimizer = list(model.roberta.named_parameters())
        classifier_optimizer = list(model.classifier.named_parameters())
        lstm_optimizer = list(model.bilstm.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.},
            {'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
             'lr': lr * 5, 'weight_decay': 0.01},
            {'params': [p for n, p in lstm_optimizer if any(nd in n for nd in no_decay)],
             'lr': lr * 5, 'weight_decay': 0.},
            {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
             'lr': lr * 5, 'weight_decay': 0.01},
            {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
             'lr': lr * 5, 'weight_decay': 0.},
            {'params': model.crf.parameters(), 'lr': lr * 10}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=len(train_data_loader) * EPOCHS // 10,
                                                num_training_steps=len(train_data_loader) * EPOCHS)

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
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            optimizer.step()
            scheduler.step()

            if step % (len(train_data_loader) // 9) == 0:
                log.info("epoch: {} step: {}/{}   {}".format(epoch, step, len(train_data_loader),
                                                               format_time(time.time() - t0)))

        model.save_pretrained('../res/Roberta_BiLSTM_CRF/last')

        avg_train_loss = total_train_loss / len(train_data_loader)
        train_loss.append(avg_train_loss)

        # print("Evaluating......")
        train_metrics = evaluate(model, train_data_loader)
        val_metrics = evaluate(model, val_data_loader)

        val_loss.append(val_metrics['loss'])
        val_f1.append(val_metrics['f1'])

        if val_metrics['f1'] > max_val_f1:
            max_val_f1 = val_metrics['f1']
            model.save_pretrained('../res/Roberta_BiLSTM_CRF/best')
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
        test(model, test_data_loader)

    log.info('-----Training Finished!-----')
