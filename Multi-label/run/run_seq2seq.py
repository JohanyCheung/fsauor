# -*- coding: utf-8 -*-
# base
import os
import argparse
import logging
import csv
import numpy as np
# model
import torch
from torch.optim.lr_scheduler import StepLR
import torchtext
import torchtext.vocab as vocab
# import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint
# my
from config import Config
from helper import get_chars, process_csv_dict, process_pred, generate
from evaluate import f1_mltc

# conf-my
myconf = Config("conf/seq2seq.conf")
fields = myconf.data.fields.split(',')
if not os.path.exists(".vector_cache"):
    os.mkdir(".vector_cache")

# usage:
#   TRAIN_PATH=./preprocess/train.tsv
#   DEV_PATH=./preprocess/val_10000.tsv
#   EXPT_PATH=./model/seq2seq_char
#   TEST_PATH=./preprocess/val_5000.csv
#   PRED_PATH=./preprocess/pred_5000.csv
#
# stopwords
#   TRAIN_PATH=./preprocess/stopwords/train.tsv
#   DEV_PATH=./preprocess/stopwords/val_10000.tsv
#   EXPT_PATH=./model/seq2seq_char_stopwords
#   TEST_PATH=./preprocess/val_5000.csv
#   PRED_PATH=./preprocess/pred_5000.csv
#
# base
#   TRAIN_PATH=./preprocess/base/train.tsv
#   DEV_PATH=./preprocess/base/val_10000.tsv
#   EXPT_PATH=./model/seq2seq_char_base
#   TEST_PATH=./preprocess/val_5000.csv
#   PRED_PATH=./preprocess/pred_5000.csv
#
# submit_stopwords
#   TRAIN_PATH=./preprocess/stopwords/train.tsv
#   DEV_PATH=./preprocess/stopwords/val.tsv
#   EXPT_PATH=./model/seq2seq_char_stopwords
#   TEST_PATH=./preprocess/testa.csv
#   PRED_PATH=./preprocess/pred.csv
#
# submit_base
#   TRAIN_PATH=./preprocess/base/train.tsv
#   DEV_PATH=./preprocess/base/val.tsv
#   EXPT_PATH=./model/seq2seq_char_base
#   TEST_PATH=./preprocess/testa.csv
#   PRED_PATH=./preprocess/pred.csv
#
#   # training
#       python run_seq2seq.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --test_path $TEST_PATH --pred_path $PRED_PATH
#   # resuming from the latest checkpoint of the experiment
#       python run_seq2seq.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --test_path $TEST_PATH --pred_path $PRED_PATH --resume
#   # resuming from a specific checkpoint
#       python run_seq2seq.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --test_path $TEST_PATH --pred_path $PRED_PATH --load_checkpoint $CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path', default='./preprocess/train.tsv',
                    help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path', default='./preprocess/val_10000.tsv',
                    help='Path to dev data')
parser.add_argument('--test_path', action='store', dest='test_path', default='./preprocess/val_5000.csv',
                    help='Path to test data')
parser.add_argument('--pred_path', action='store', dest='pred_path', default='./preprocess/pred_5000.csv',
                    help='Path to pred data')
parser.add_argument('--submit_path', action='store', dest='submit_path', default='./submit/seq2seq/submit.csv',
                    help='Path to submit data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./model/seq2seq_char',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')

opt = parser.parse_args()


LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    # Prepare dataset
    src = SourceField()
    tgt = TargetField()
    max_len = int(myconf.model.maxlen)
    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len
    train = torchtext.data.TabularDataset(
        path=opt.train_path, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )
    dev = torchtext.data.TabularDataset(
        path=opt.dev_path, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )
    # 构建语料库的 Vocabulary，同时加载预训练的 word-embedding。通过 vocab.Vectors 使用自定义的 vectors。
    vectors = vocab.Vectors(myconf.embedding.char2vec)
    src.build_vocab(train, max_size=int(myconf.model.src_vocab_size))
    tgt.build_vocab(train, max_size=int(myconf.model.tgt_vocab_size))
    src.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # NOTE: If the source field name and the target field name
    # are different from 'src' and 'tgt' respectively, they have
    # to be set explicitly before any training or inference
    # seq2seq.src_field_name = 'src'
    # seq2seq.tgt_field_name = 'tgt'

    # Prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size = 256
        bidirectional = True
        # encoder = EncoderRNN(len(src.vocab), max_len, hidden_size,
        #                      bidirectional=bidirectional, variable_lengths=True)
        # decoder = DecoderRNN(len(tgt.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
        #                      dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
        #                      eos_id=tgt.eos_id, sos_id=tgt.sos_id)
        encoder = EncoderRNN(len(src.vocab), max_len, hidden_size,
                             n_layers=2, bidirectional=bidirectional, variable_lengths=True)
        decoder = DecoderRNN(len(tgt.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                             n_layers=2, dropout_p=0.5, use_attention=True, bidirectional=bidirectional,
                             eos_id=tgt.eos_id, sos_id=tgt.sos_id)
        seq2seq = Seq2seq(encoder, decoder)
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        # optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters(), amsgrad=True, weight_decay=0.0005), max_grad_norm=10)
        optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=10)
        scheduler = StepLR(optimizer.optimizer, 1)
        optimizer.set_scheduler(scheduler)

    # train
    t = SupervisedTrainer(loss=loss, batch_size=128,
                          checkpoint_every=200,
                          print_every=200, expt_dir=opt.expt_dir)

    seq2seq = t.train(seq2seq, train,
                      num_epochs=5, dev_data=dev,
                      optimizer=optimizer,
                      teacher_forcing_ratio=0.5,
                      resume=opt.resume)

predictor = Predictor(seq2seq, input_vocab, output_vocab)


def predict_text(text):
    seq = get_chars(text)
    # base
    # res = predictor.predict(seq)[:-1]

    # custom
    softmax_list, tgt_seq = predictor.predict(seq)
    res = tgt_seq[:-1]
    print(softmax_list)

    return res


def predict_seq(line):
    # val
    seq = line.rstrip().split('\t')[0].split(' ')
    # test
    # seq = line.rstrip().split(' ')
    return predictor.predict(seq)[:-1]


def predict_subsets(row, fields):
    """pred subsets"""
    seq = get_chars(row['content'])

    # base
    # preds = predictor.predict(seq)[:-1]

    # custom
    softmax_list, tgt_seq = predictor.predict(seq)
    preds = tgt_seq[:-1]

    res = {}
    for k, v in zip(fields, preds):
        res[k] = v
    return res


process_csv_dict(srcfile=opt.test_path, desfile=opt.pred_path, func=predict_subsets, fields=fields)

# test idea
# f1 = f1_mltc(
#     pfile=opt.pred_path,
#     lfile=opt.test_path,
#     fields=fields,
#     vocab=[int(i) for i in myconf.label.vocab.split(',')]
#     )
# print('f1:', f1)

# submit
for k in fields:
    print(k)
    generate(pfile=opt.pred_path, desfile=opt.submit_path, k=k, default=-2)