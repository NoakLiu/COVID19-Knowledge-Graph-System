import os
from transformers import logging
import torch
import argparse
import logging as log
from run import ensemble, Bert_MLP, Bert_BiLSTM_CRF, Roberta_MLP, Roberta_BiLSTM_CRF, Bert_CRF, Roberta_CRF

logging.set_verbosity_error()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--resume', nargs='?', const=True, default=False)
    parser.add_argument('--policy', type=str)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--full-fine-tuning', nargs='?', const=True, default=False)
    parser.add_argument('--trainAll', nargs='?', const=True, default=False)
    parser.add_argument('--train-ensemble', type=str, default="", help="deprecated")
    args = parser.parse_args()

    args.train_ensemble = args.train_ensemble.split(',') if len(args.train_ensemble) else []

    if len(args.train_ensemble):
        for m in args.train_ensemble:
            if m not in ['Bert_MLP', 'Bert_CRF', 'Bert_BiLSTM_CRF', 'Roberta_MLP', 'Roberta_CRF', 'Roberta_BiLSTM_CRF']:
                print("--train-ensemble should be chosen in ['Bert_MLP', 'Bert_CRF', 'Bert_BiLSTM_CRF', 'Roberta_MLP', 'Roberta_CRF', 'Roberta_BiLSTM_CRF']")
                exit(0)
    else:
        if args.policy not in ['Bert_MLP', 'Bert_CRF', 'Bert_BiLSTM_CRF', 'Roberta_MLP', 'Roberta_CRF', 'Roberta_BiLSTM_CRF']:
            print( "--policy should be in ['Bert_MLP', 'Bert_CRF', 'Bert_BiLSTM_CRF', 'Roberta_MLP', 'Roberta_CRF', 'Roberta_BiLSTM_CRF']")
            exit(0)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    logger = log.getLogger()
    logger.setLevel(log.INFO)

    if not os.path.exists('../res'):
        os.makedirs('../res')
    if len(args.train_ensemble):
        if not os.path.exists('../res/Ensemble'):
            os.makedirs('../res/Ensemble')
    elif not os.path.exists('../res/' + args.policy):
        os.makedirs('../res/' + args.policy)

    file_handler = log.FileHandler('../res/' + ('Ensemble' if len(args.train_ensemble) else args.policy) + '/train.log')
    file_handler.setFormatter(log.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    stream_handler = log.StreamHandler()
    stream_handler.setFormatter(log.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    log.info('------------------------------------------------')
    log.info("device: {}".format(device))
    log.info("epochs: {}  batch-size: {}  resume: {}".format(args.epochs, args.batch_size, args.resume))
    log.info('------------------------------------------------')

    if len(args.train_ensemble):
        ensemble.train(args.train_ensemble, args.epochs, args.batch_size, args.lr, args.resume, args.trainAll)
        ensemble.test(args.train_ensemble)
    else:
        eval(args.policy + '.train(EPOCHS={}, batch_size={}, lr={}, full_fine_tuning={}, resume={}, trainAll={})'.format(args.epochs, args.batch_size, args.lr, args.full_fine_tuning, args.resume, args.trainAll))
        eval(args.policy + '.test()')
