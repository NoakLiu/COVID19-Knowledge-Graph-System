import torch.nn as nn
import torch
from transformers import BertModel, BertPreTrainedModel
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
from utils.convert import num_labels


class Bert_MLP(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert_MLP, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
            )
        seq_output = outputs[0]
        # 去掉[CLS]、[SEP]、[PAD]
        origin_seq_output = [words[1: torch.count_nonzero(mask) - 1]
                             for words, mask in zip(seq_output, attention_mask)]

        padded_seq_output = pad_sequence(origin_seq_output, batch_first=True)
        padded_seq_output = self.dropout(padded_seq_output)
        # logits -> (batch_size, seq_length, num_labels)
        logits = self.classifier(padded_seq_output)

        outputs = (logits, )

        if labels is not None:
            # loss_mask -> (batch_size, max_len)
            origin_labels = [label[: torch.count_nonzero(mask) - 2]
                             for label, mask in zip(labels, attention_mask)]
            padded_labels = pad_sequence(origin_labels, batch_first=True, padding_value=-1)
            loss_mask = padded_labels.gt(-1)
            loss_fn = nn.CrossEntropyLoss()
            if loss_mask is not None:
                loss_mask = loss_mask.view(-1)
                active_logits = logits.view(-1, self.num_labels)[loss_mask]
                active_labels = padded_labels.view(-1)[loss_mask]
                loss = loss_fn(active_logits, active_labels)
            else:
                loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss, ) + outputs

        return outputs


class Bert_BiLSTM_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert_BiLSTM_CRF, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bilstm = nn.LSTM(
            input_size=config.lstm_embedding_size,
            hidden_size=config.hidden_size // 2,
            batch_first=True,
            num_layers=2,
            dropout=config.lstm_dropout_prob,
            bidirectional=True
        )
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
            )
        seq_output = outputs[0]
        # 去掉[CLS]、[SEP]、[PAD]
        origin_seq_output = [words[1: torch.count_nonzero(mask) - 1]
                             for words, mask in zip(seq_output, attention_mask)]

        padded_seq_output = pad_sequence(origin_seq_output, batch_first=True)
        padded_seq_output = self.dropout(padded_seq_output)
        # logits -> (batch_size, seq_length, num_labels)
        bilstm_output, _ = self.bilstm(padded_seq_output)
        logits = self.classifier(bilstm_output)

        outputs = (logits, )

        if labels is not None:
            # loss_mask -> (batch_size, max_len)
            origin_labels = [label[: torch.count_nonzero(mask) - 2]
                             for label, mask in zip(labels, attention_mask)]
            padded_labels = pad_sequence(origin_labels, batch_first=True, padding_value=-1)
            loss_mask = padded_labels.gt(-1)
            if loss_mask is not None:
                # loss_mask = loss_mask.view(-1)
                # active_logits = logits.view(-1, self.num_labels)[loss_mask].unsqueeze(0)
                # active_labels = padded_labels.view(-1)[loss_mask].unsqueeze(0)
                loss = self.crf(logits, padded_labels, loss_mask) * (-1)
            else:
                loss = self.crf(logits, padded_labels) * (-1)
            outputs = (loss,) + outputs

        return outputs


class Roberta_MLP(BertPreTrainedModel):
    def __init__(self, config):
        super(Roberta_MLP, self).__init__(config)
        self.num_labels = num_labels
        self.roberta = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
            )
        seq_output = outputs[0]
        # 去掉[CLS]、[SEP]、[PAD]
        origin_seq_output = [words[1: torch.count_nonzero(mask) - 1]
                             for words, mask in zip(seq_output, attention_mask)]

        padded_seq_output = pad_sequence(origin_seq_output, batch_first=True)
        padded_seq_output = self.dropout(padded_seq_output)
        # logits -> (batch_size, seq_length, num_labels)
        logits = self.classifier(padded_seq_output)

        outputs = (logits, )

        if labels is not None:
            # loss_mask -> (batch_size, max_len)
            origin_labels = [label[: torch.count_nonzero(mask) - 2]
                             for label, mask in zip(labels, attention_mask)]
            padded_labels = pad_sequence(origin_labels, batch_first=True, padding_value=-1)
            loss_mask = padded_labels.gt(-1)
            loss_fn = nn.CrossEntropyLoss()
            if loss_mask is not None:
                loss_mask = loss_mask.view(-1)
                active_logits = logits.view(-1, self.num_labels)[loss_mask]
                active_labels = padded_labels.view(-1)[loss_mask]
                loss = loss_fn(active_logits, active_labels)
            else:
                loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss, ) + outputs

        return outputs


class Roberta_BiLSTM_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super(Roberta_BiLSTM_CRF, self).__init__(config)
        self.num_labels = num_labels
        self.roberta = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bilstm = nn.LSTM(
            input_size=config.lstm_embedding_size,
            hidden_size=config.hidden_size // 2,
            batch_first=True,
            num_layers=2,
            dropout=config.lstm_dropout_prob,
            bidirectional=True
        )
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
            )
        seq_output = outputs[0]
        # 去掉[CLS]、[SEP]、[PAD]
        origin_seq_output = [words[1: torch.count_nonzero(mask) - 1]
                             for words, mask in zip(seq_output, attention_mask)]

        padded_seq_output = pad_sequence(origin_seq_output, batch_first=True)
        # padded_seq_output -> (batch_size, seq_length, hidden_size)
        padded_seq_output = self.dropout(padded_seq_output)
        # logits -> (batch_size, seq_length, num_labels)
        bilstm_output, _ = self.bilstm(padded_seq_output)
        logits = self.classifier(bilstm_output)

        outputs = (logits, )

        if labels is not None:
            # loss_mask -> (batch_size, max_len)
            origin_labels = [label[: torch.count_nonzero(mask) - 2]
                             for label, mask in zip(labels, attention_mask)]
            padded_labels = pad_sequence(origin_labels, batch_first=True, padding_value=-1)
            loss_mask = padded_labels.gt(-1)
            if loss_mask is not None:
                # loss_mask = loss_mask.view(-1)
                # active_logits = logits.view(-1, self.num_labels)[loss_mask].unsqueeze(0)
                # active_labels = padded_labels.view(-1)[loss_mask].unsqueeze(0)
                loss = self.crf(logits, padded_labels, loss_mask) * (-1)
            else:
                loss = self.crf(logits, padded_labels) * (-1)
            outputs = (loss,) + outputs

        return outputs


class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models
        self.num_labels = num_labels
        # self.classifier = nn.Linear(num_labels * len(models), num_labels, bias=False)

    def forward(self, input_ids, attention_mask=None):
        prob = []
        with torch.no_grad():
            for model in self.models:
                output = model(input_ids, attention_mask)
                if isinstance(model, (Bert_CRF, Bert_BiLSTM_CRF, Roberta_CRF, Roberta_BiLSTM_CRF)):
                    seq_mask = [i[1: torch.count_nonzero(i) - 1] for i in attention_mask]
                    seq_mask = pad_sequence(seq_mask, batch_first=True).gt(0)
                    pred = model.crf.decode(output[0], mask=seq_mask)
                    crf_prob = torch.zeros_like(output[0])
                    for i, sentence in enumerate(pred):
                        for k, label_id in enumerate(sentence):
                            crf_prob[i][k][label_id] = 0.35
                    prob.append(crf_prob)
                else:
                    prob.append(torch.softmax(output[0], dim=2))

        # ensemble_prob -> (batch_size, seq_length, num_labels)
        ensemble_prob = prob[0]
        for i in range(1, len(self.models)):
            ensemble_prob = torch.add(ensemble_prob, prob[i])

        return ensemble_prob
        # ensemble_prob = torch.cat(prob, dim=2)
        # logits = self.classifier(ensemble_prob)
        #
        # outputs = (logits,)
        #
        # if labels is not None:
        #     # loss_mask -> (batch_size, max_len)
        #     origin_labels = [label[: torch.count_nonzero(mask) - 2]
        #                      for label, mask in zip(labels, attention_mask)]
        #     padded_labels = pad_sequence(origin_labels, batch_first=True, padding_value=-1)
        #     loss_mask = padded_labels.gt(-1)
        #     loss_fn = nn.CrossEntropyLoss()
        #     if loss_mask is not None:
        #         loss_mask = loss_mask.view(-1)
        #         active_logits = logits.view(-1, self.num_labels)[loss_mask]
        #         active_labels = padded_labels.view(-1)[loss_mask]
        #         loss = loss_fn(active_logits, active_labels)
        #     else:
        #         loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        #     outputs = (loss,) + outputs
        #
        # return outputs


class Bert_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert_CRF, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
            )
        seq_output = outputs[0]
        # 去掉[CLS]、[SEP]、[PAD]
        origin_seq_output = [words[1: torch.count_nonzero(mask) - 1]
                             for words, mask in zip(seq_output, attention_mask)]

        padded_seq_output = pad_sequence(origin_seq_output, batch_first=True)
        padded_seq_output = self.dropout(padded_seq_output)
        # logits -> (batch_size, seq_length, num_labels)
        logits = self.classifier(padded_seq_output)

        outputs = (logits, )

        if labels is not None:
            # loss_mask -> (batch_size, max_len)
            origin_labels = [label[: torch.count_nonzero(mask) - 2]
                             for label, mask in zip(labels, attention_mask)]
            padded_labels = pad_sequence(origin_labels, batch_first=True, padding_value=-1)
            loss_mask = padded_labels.gt(-1)
            if loss_mask is not None:
                # loss_mask = loss_mask.view(-1)
                # active_logits = logits.view(-1, self.num_labels)[loss_mask].unsqueeze(0)
                # active_labels = padded_labels.view(-1)[loss_mask].unsqueeze(0)
                loss = self.crf(logits, padded_labels, loss_mask) * (-1)
            else:
                loss = self.crf(logits, padded_labels) * (-1)
            outputs = (loss,) + outputs

        return outputs


class Roberta_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super(Roberta_CRF, self).__init__(config)
        self.num_labels = num_labels
        self.roberta = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
            )
        seq_output = outputs[0]
        # 去掉[CLS]、[SEP]、[PAD]
        origin_seq_output = [words[1: torch.count_nonzero(mask) - 1]
                             for words, mask in zip(seq_output, attention_mask)]

        padded_seq_output = pad_sequence(origin_seq_output, batch_first=True)
        padded_seq_output = self.dropout(padded_seq_output)
        # logits -> (batch_size, seq_length, num_labels)
        logits = self.classifier(padded_seq_output)

        outputs = (logits, )

        if labels is not None:
            # loss_mask -> (batch_size, max_len)
            origin_labels = [label[: torch.count_nonzero(mask) - 2]
                             for label, mask in zip(labels, attention_mask)]
            padded_labels = pad_sequence(origin_labels, batch_first=True, padding_value=-1)
            loss_mask = padded_labels.gt(-1)
            if loss_mask is not None:
                # loss_mask = loss_mask.view(-1)
                # active_logits = logits.view(-1, self.num_labels)[loss_mask].unsqueeze(0)
                # active_labels = padded_labels.view(-1)[loss_mask].unsqueeze(0)
                loss = self.crf(logits, padded_labels, loss_mask) * (-1)
            else:
                loss = self.crf(logits, padded_labels) * (-1)
            outputs = (loss,) + outputs

        return outputs
