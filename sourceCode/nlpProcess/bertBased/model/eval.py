from utils.data_process import preprocess, NERDataset
from utils.NERmodels import Bert_MLP, Bert_CRF, Bert_BiLSTM_CRF, Roberta_MLP, Roberta_CRF, Roberta_BiLSTM_CRF, Ensemble
from transformers import BertTokenizer, logging
import logging as log
from utils.utils import evaluate_ensemble, evaluate
from torch.utils.data import DataLoader
import torch
import argparse
import os

logging.set_verbosity_error()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble', type=str, default="", help="此参数优先级更高")
    parser.add_argument('--policy', type=str)
    args = parser.parse_args()

    args.ensemble = args.ensemble.split(',') if len(args.ensemble) else []

    if len(args.ensemble):
        for m in args.ensemble:
            if m not in ['Bert_MLP', 'Bert_CRF', 'Bert_BiLSTM_CRF', 'Roberta_MLP', 'Roberta_CRF', 'Roberta_BiLSTM_CRF']:
                print("--ensemble should be chosen in ['Bert_MLP', 'Bert_CRF', 'Bert_BiLSTM_CRF', 'Roberta_MLP', 'Roberta_CRF', 'Roberta_BiLSTM_CRF']")
                exit(0)
    else:
        if args.policy not in ['Bert_MLP', 'Bert_CRF', 'Bert_BiLSTM_CRF', 'Roberta_MLP', 'Roberta_CRF', 'Roberta_BiLSTM_CRF']:
            print("--policy should be in ['Bert_MLP', 'Bert_CRF', 'Bert_BiLSTM_CRF', 'Roberta_MLP',  'Roberta_CRF', 'Roberta_BiLSTM_CRF']")
            exit(0)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    logger = log.getLogger()
    logger.setLevel(log.INFO)

    if not os.path.exists('../res'):
        os.makedirs('../res')

    file_handler = log.FileHandler('../res/eval.log')
    file_handler.setFormatter(log.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    stream_handler = log.StreamHandler()
    stream_handler.setFormatter(log.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    log.info('------------------------------------------------')
    log.info("device: {}".format(device))
    log.info("model: {}".format('Ensemble({})'.format(args.ensemble) if len(args.ensemble) else args.policy))
    log.info('------------------------------------------------')

    # ---------------------------------------------------------------------------------
    tokenizer = BertTokenizer.from_pretrained('../../../../dependencies/bert-base-chinese')

    words, labels = preprocess(
        '../../../../dataset/Dataset-Epidemiologic-Investigation-COVID19-master/ECR-COVID-19/train.txt')
    dataset = NERDataset(words, labels, tokenizer, 512)
    train_data_loader = DataLoader(dataset, batch_size=1)

    words, labels = preprocess(
        '../../../../dataset/Dataset-Epidemiologic-Investigation-COVID19-master/ECR-COVID-19/valid.txt')
    dataset = NERDataset(words, labels, tokenizer, 512)
    val_data_loader = DataLoader(dataset, batch_size=1)

    words, labels = preprocess(
        '../../../../dataset/Dataset-Epidemiologic-Investigation-COVID19-master/ECR-COVID-19/test.txt')
    dataset = NERDataset(words, labels, tokenizer, 512)
    test_data_loader = DataLoader(dataset, batch_size=1)

    if len(args.ensemble):
        model_list = []
        for m in args.ensemble:
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

        model = Ensemble(model_list)

        train_metrics = evaluate_ensemble(model, train_data_loader)
        val_metrics = evaluate_ensemble(model, val_data_loader)
        test_metrics = evaluate_ensemble(model, test_data_loader)

    else:
        if args.policy == 'Bert_MLP':
            model = Bert_MLP.from_pretrained('../res/Bert_MLP/best')
        elif args.policy == "Bert_CRF":
            model = Bert_CRF.from_pretrained('../res/Bert_CRF/best')
        elif args.policy == "Bert_BiLSTM_CRF":
            model = Bert_BiLSTM_CRF.from_pretrained('../res/Bert_BiLSTM_CRF/best')
        elif args.policy == "Roberta_MLP":
            model = Roberta_MLP.from_pretrained('../res/Roberta_MLP/best')
        elif args.policy == "Roberta_CRF":
            model = Roberta_CRF.from_pretrained('../res/Roberta_CRF/best')
        elif args.policy == "Roberta_BiLSTM_CRF":
            model = Roberta_BiLSTM_CRF.from_pretrained('../res/Roberta_BiLSTM_CRF/best')
        model.to(device)

        train_metrics = evaluate(model, train_data_loader)
        val_metrics = evaluate(model, val_data_loader)
        test_metrics = evaluate(model,  test_data_loader)

    log.info("train: precision: {:.2f}%  recall: {:.2f}%  f1: {:.2f}%\n".format(train_metrics['precision'] * 100.,
                                                                                train_metrics['recall'] * 100.,
                                                                                train_metrics['f1'] * 100.) +
             "val: precision: {:.2f}%  recall: {:.2f}%  f1: {:.2f}%\n".format(val_metrics['precision'] * 100.,
                                                                              val_metrics['recall'] * 100.,
                                                                              val_metrics['f1'] * 100.) +
             "test: precision: {:.2f}%  recall: {:.2f}%  f1: {:.2f}%".format(test_metrics['precision'] * 100.,
                                                                              test_metrics['recall'] * 100.,
                                                                              test_metrics['f1'] * 100.))
