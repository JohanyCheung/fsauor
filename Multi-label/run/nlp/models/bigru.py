# -*- coding: utf-8 -*-
import keras
from keras import Model
from keras.layers import *
from .attention import Attention


class TextClassifier():

    def model(self, embeddings_matrix, maxlen, word_index, num_class):
        inp = Input(shape=(maxlen,))
        encode = Bidirectional(CuDNNGRU(128, return_sequences=True))
        encode2 = Bidirectional(CuDNNGRU(128, return_sequences=True))
        attention = Attention(maxlen)
        x_4 = Embedding(len(word_index) + 1,
                        embeddings_matrix.shape[1],
                        weights=[embeddings_matrix],
                        input_length=maxlen,
                        trainable=True)(inp)
        x_3 = SpatialDropout1D(0.2)(x_4)
        x_3 = encode(x_3)
        x_3 = Dropout(0.2)(x_3)
        x_3 = encode2(x_3)
        x_3 = Dropout(0.2)(x_3)
        avg_pool_3 = GlobalAveragePooling1D()(x_3)
        max_pool_3 = GlobalMaxPooling1D()(x_3)
        attention_3 = attention(x_3)
        x = keras.layers.concatenate([avg_pool_3, max_pool_3, attention_3], name="fc")
        if num_class == 2:
            output = Dense(num_class, activation="sigmoid")(x)
            loss = 'binary_crossentropy'
        else:
            output = Dense(num_class, activation="softmax")(x)
            loss = 'categorical_crossentropy'
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)
        model = Model(inputs=inp, outputs=output)
        model.compile(
            loss=loss,
            optimizer=adam,
            metrics=["categorical_accuracy"])
        return model


class TextClassifier2():

    def model(self, embeddings_matrix, maxlen, word_index, num_class):
        inp = Input(shape=(maxlen,))
        x_emb = Embedding(len(word_index) + 1,
                        embeddings_matrix.shape[1],
                        weights=[embeddings_matrix],
                        input_length=maxlen,
                        trainable=True)(inp)

        x = SpatialDropout1D(0.2)(x_emb)

        x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
        x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)

        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = concatenate([avg_pool, max_pool])

        x = Dense(1000)(x)
        x = BatchNormalization()(x)
        # x = Activation(activation="relu")(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        x = Dense(500)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        if num_class == 2:
            output = Dense(num_class, activation="sigmoid")(x)
            loss = 'binary_crossentropy'
        else:
            output = Dense(num_class, activation="softmax")(x)
            loss = 'categorical_crossentropy'
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)
        model = Model(inputs=inp, outputs=output)
        model.compile(
            loss=loss,
            optimizer=adam, # 'adam'
            metrics=["accuracy"])
        return model
