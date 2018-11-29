# -*- coding: utf-8 -*-
# base
import os
import sys
import json
import gc
import pickle
import numpy as np
import pandas as pd
# model
import tensorflow as tf
from tensorflow import set_random_seed
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing import text, sequence
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.layers import *
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from gensim.models.keyedvectors import KeyedVectors
# my
from config import Config
from nlp.models.capsule import TextClassifier
from keras.utils.vis_utils import plot_model
from helper import logging, logging_csv

# conf-my
myconf = Config("conf/capsule_char.conf")
vocab = [int(i) for i in myconf.label.vocab.split(',')]
with open(myconf.model.pre, 'r', encoding="UTF-8") as f:
    pre = json.load(f)
fields = myconf.data.fields.split(',')
# conf-tf
tfconf = tf.ConfigProto()
tfconf.gpu_options.allow_growth = True
set_session(tf.Session(config=tfconf))
# conf-random
np.random.seed(int(myconf.random.train_seed))
set_random_seed(int(myconf.random.train_seed))
# log
if not os.path.exists(myconf.log.dir):
    os.makedirs(myconf.log.dir)
log = logging_csv(myconf.log.path) 


def get_label(arr, vocab=vocab):
    arr = list(arr)
    return vocab[arr.index(max(arr))]

def get_index(arr):
    arr = list(arr)
    return arr.index(max(arr))

def get_prob(arr):
    return list(arr)

def same(i, k):
    if i == k:
        return 1
    else:
        return 0


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_roc_auc = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(self.validation_data[0])
        # val_pre = list(map(get_label, pred))
        # val_tgt = list(map(get_label, self.validation_data[1]))
        val_pre = list(map(get_index, pred))
        val_tgt = list(map(get_index, self.validation_data[1]))
        _val_f1 = f1_score(val_tgt, val_pre, average="macro")
        idx = [0, 1, 2, 3]
        if len(idx) != 2:
            tmp = []
            for i in idx:
                val_tgt_i = []
                val_score_i = []
                for j,v in enumerate(val_tgt):
                    if same(i,v):
                        val_tgt_i.append(1)
                        val_score_i.append(pred[j][i])
                    else:
                        val_tgt_i.append(0)
                        val_score_i.append(1-pred[j][i])
                tmp.append(roc_auc_score(val_tgt_i, val_score_i))
            _val_roc_auc = np.mean(tmp)
        else:
            val_score = [pred[i][v] for i,v in enumerate(val_tgt)]
            _val_roc_auc = roc_auc_score(val_tgt, val_score)
        _val_recall = recall_score(val_tgt, val_pre, average="macro")
        _val_precision = precision_score(val_tgt, val_pre, average="macro")
        self.val_f1s.append(_val_f1)
        self.val_roc_auc.append(_val_roc_auc)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        max_f1 = max(self.val_f1s)
        max_roc_auc = max(self.val_roc_auc)
        print("f1:", _val_f1, "roc_auc:", _val_roc_auc, "precision:", _val_precision, "recall:", _val_recall)
        print("max f1:", max_f1)
        print("max roc_auc:", max_roc_auc)
        log([epoch, max_f1, _val_f1, max_roc_auc, _val_roc_auc, _val_precision, _val_recall])
        return


def embeddings(tokenizer, word_index, mode="char", src="w2v"):
    if src == "w2v":
        vec_path = myconf.embedding.c2v_w2v
    elif src == "glove":
        vec_path = myconf.embedding.c2v_glove
    elif src == "pinyin":
        vec_path = myconf.embedding.c2v_pinyin 

    if src != "glove":
        i2v = KeyedVectors.load_word2vec_format(vec_path, binary=True, encoding='utf8', unicode_errors='ignore')
    else:
        i2v = KeyedVectors.load_word2vec_format(vec_path)
    embeddings_matrix = np.zeros((len(word_index) + 1, i2v.vector_size))

    for word, i in word_index.items():
        embedding_vector = i2v[word] if word in i2v else None
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector

    return embeddings_matrix


def embeddings_mix(tokenizer, word_index):
    i2v = KeyedVectors.load_word2vec_format(myconf.embedding.c2v_w2v, binary=True, encoding='utf8', unicode_errors='ignore')
    p2v = KeyedVectors.load_word2vec_format(myconf.embedding.c2v_pinyin, binary=True, encoding='utf8', unicode_errors='ignore')
    embeddings_matrix = np.zeros((len(word_index) + 1, i2v.vector_size))

    for word, i in word_index.items():
        py = char2pinyin(word)
        embedding_vector = i2v[word] if word in i2v else None
        embedding_pinyin = p2v[py] if py in p2v else None
        if embedding_vector is not None:
            if embedding_pinyin is not None:
                embeddings_matrix[i] = embedding_vector + embedding_pinyin
            else:
                embeddings_matrix[i] = embedding_vector
        else:
            if embedding_pinyin is not None:
                embeddings_matrix[i] = embedding_pinyin

    return embeddings_matrix


def train():
    model_dir = myconf.model.dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    maxlen = int(myconf.model.maxlen)
    max_features = int(myconf.model.max_features)
    batch_size = int(myconf.model.batch_size)
    epochs = int(myconf.model.epochs)

    train_data = pd.read_csv(myconf.data.train)
    train_data[myconf.data.src_field] = train_data.apply(lambda x: eval(x[1]), axis=1)
    val_data = pd.read_csv(myconf.data.val)
    val_data[myconf.data.src_field] = val_data.apply(lambda x: eval(x[1]), axis=1)
    if myconf.data.combine == 'yes':
        train_data = pd.concat([train_data, val_data])

    X_train = train_data[myconf.data.src_field].values
    Y_train = [pd.get_dummies(train_data[k])[vocab].values for k in fields]

    tokenizer = text.Tokenizer(num_words=None)
    tokenizer.fit_on_texts(X_train)
    word_index = tokenizer.word_index
    if not os.path.exists(myconf.model.tokenizer):
        with open(myconf.model.tokenizer, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    emb_w2v = embeddings(tokenizer, word_index, src="w2v")

    list_tokenized_train = tokenizer.texts_to_sequences(X_train)
    input_train = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
    if myconf.data.combine != 'yes':
        X_val = val_data[myconf.data.src_field].values
        Y_val = [pd.get_dummies(val_data[k])[vocab].values for k in fields]
        list_tokenized_val = tokenizer.texts_to_sequences(X_val)
        input_val = sequence.pad_sequences(list_tokenized_val, maxlen=maxlen)

    for i, k in enumerate(fields):
        log([k, "max_f1", "f1", "max_roc_auc", "roc_auc", "precision", "recall"])
        print('\n', k)
        model = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
        if i == 0:
            plot_model(model, to_file='capsule.png', show_shapes=True, show_layer_names=False)
        # file_path = model_dir + k + "_{epoch:02d}.hdf5"
        file_path = model_dir + k + "_{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)

        # earlystop = EarlyStopping(monitor='val_f1', patience=3, restore_best_weights=True)
        # checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True, save_best_only=True)
        metrics = Metrics()
        callbacks_list = [checkpoint, metrics]
        # callbacks_list = [metrics, checkpoint, earlystop]
        if myconf.data.combine == 'yes':
            history = model.fit(input_train, Y_train[i], batch_size=batch_size, epochs=epochs,
                             validation_split=0.1, callbacks=callbacks_list, verbose=2)
        else:
            history = model.fit(input_train, Y_train[i], batch_size=batch_size, epochs=epochs,
                             validation_data=(input_val, Y_val[i]), callbacks=callbacks_list, verbose=2)
        del model
        del history
        gc.collect()
        K.clear_session()


def pred():
    with open(myconf.model.tokenizer, 'rb') as handle:
        maxlen = int(myconf.model.maxlen)
        model_dir = myconf.model.dir
        tokenizer = pickle.load(handle)
        word_index = tokenizer.word_index
        test = pd.read_csv(myconf.data.test)
        test[myconf.data.src_field] = test.apply(lambda x: eval(x[1]), axis=1)
        X_test = test[myconf.data.src_field].values
        list_tokenized_test = tokenizer.texts_to_sequences(X_test)
        input_test = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

        emb_w2v = embeddings(tokenizer, word_index, src="w2v")

        if myconf.submit.dataset == "A":
            submit = pd.read_csv(myconf.data.testa_download)
            submit_prob = pd.read_csv(myconf.data.testa_download)
        elif myconf.submit.dataset == "B":
            submit = pd.read_csv(myconf.data.testb_download)
            submit_prob = pd.read_csv(myconf.data.testb_download)

        for k, v in pre.items():
            print('\n', k)
            model = TextClassifier().model(emb_w2v, maxlen, word_index, 4)
            model.load_weights(os.path.join(model_dir, v))
            submit[k] = list(map(get_label, model.predict(input_test)))
            submit_prob[k] = list(map(get_prob, model.predict(input_test))) # sep=','
            # submit_prob[k] = list(model.predict(input_test)) # np.array sep=' '
            del model
            gc.collect()
            K.clear_session()

        submit.to_csv(myconf.data.submit, index=None)
        submit_prob.to_csv(myconf.data.submit_prob, index=None)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["-train", "-pred"]:
        raise ValueError("""usage: python run_capsule_char.py [-train / -pred]""")

    if sys.argv[1] == "-train":
        train()
    else:
        pred()