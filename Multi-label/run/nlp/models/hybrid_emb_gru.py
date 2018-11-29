# -*- coding: utf-8 -*-
import keras
from keras import Model
from keras.layers import *
from .attention import Attention


def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


def squash(x, axis=-1):
    """From hinton's paper
    (||x||^2 / (1 + ||x||^2)) * (x / ||x||)
    """
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm)
    return scale * x / K.sqrt(s_squared_norm + K.epsilon())

def squash_half(x, axis=-1):
    """Use 0.5 in stead of 1 in hinton's paper.
    if 1, the norm of vector will be zoomed out.
    if 0.5, the norm will be zoomed in while original norm is less than 0.5
    and be zoomed out while original norm is greater than 0.5.
    (||x||^2 / (0.5 + ||x||^2)) * (x / ||x||)
    """
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    return scale * x / K.sqrt(s_squared_norm + K.epsilon())

def squash_no_compression(x, axis=-1):
    """Use 0 in stead of 1 in hinton's paper without compression.
    x / ||x||
    """
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    return x / K.sqrt(s_squared_norm + K.epsilon())

def squash_1x(x, axis=-1):
    """x / (1 + ||x||)
    """
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    return x / (1 + K.sqrt(s_squared_norm + K.epsilon()))

def squash_e(x, axis=-1):
    """(1-e^(-||x||)) * (x / ||x||)
    """
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = 1 - K.exp(-K.sqrt(s_squared_norm))
    return scale * x / K.sqrt(s_squared_norm + K.epsilon())

def squash_tanh(x, axis=-1):
    """tanh||x|| * (x / ||x||)
    """
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.tanh(-K.sqrt(s_squared_norm))
    return scale * x / K.sqrt(s_squared_norm + K.epsilon())


class Capsule(Layer):
    """A Capsule Implement with Pure Keras
    """
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash_no_compression # squash_half
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


class TextClassifier():

    def model(self, maxlen, num_class, word_index, emb_w2v, emb_glove, emb_pinyin):
        inp = Input(shape=(maxlen,))

        x_w2v = Embedding(len(word_index) + 1,
                        emb_w2v.shape[1],
                        weights=[emb_w2v],
                        input_length=maxlen,
                        trainable=True)(inp)
        x_glove = Embedding(len(word_index) + 1,
                        emb_glove.shape[1],
                        weights=[emb_glove],
                        input_length=maxlen,
                        trainable=True)(inp)
        # x_pinyin = Embedding(len(word_index) + 1,
        #                 emb_pinyin.shape[1],
        #                 weights=[emb_pinyin],
        #                 input_length=maxlen,
        #                 trainable=True)(inp)
        x_emb = Concatenate()([x_w2v, x_glove])
        # x_emb = Concatenate()([x_w2v, x_glove, x_pinyin])
        x = SpatialDropout1D(0.2)(x_emb)

        x_rgru = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
        x_rgru_capsule = Capsule(num_capsule=10, dim_capsule=16, routings=5,
                          share_weights=True)(x_rgru)
        x_rgru_capsule = Flatten()(x_rgru_capsule)
        x_rgru_attention = Attention(maxlen)(x_rgru)
        x_rgru = Concatenate()([x_rgru_capsule, x_rgru_attention])

        x_remb_capsule = Capsule(num_capsule=10, dim_capsule=16, routings=5,
                          share_weights=True)(x)
        x_remb_capsule = Flatten()(x_remb_capsule)
        x_remb_attention = Attention(maxlen)(x)
        x_remb = Concatenate()([x_remb_capsule, x_remb_attention])

        x_2 = Concatenate()([x_remb, x_rgru])
        x_3 = Dense(256, activation=None)(x_2)
        x_4 = BatchNormalization()(x_3)
        x_5 = LeakyReLU()(x_4)
        x_6 = Dropout(0.2)(x_5)
        x_7 = Dense(256, activation=None)(x_6)
        x_8 = BatchNormalization()(x_7)
        x_9 = LeakyReLU()(x_8)
        x_10 = Dropout(0.2)(x_9)

        if num_class == 2:
            output = Dense(num_class, activation="sigmoid")(x_10)
            loss = 'binary_crossentropy'
        else:
            output = Dense(num_class, activation="softmax")(x_10)
            loss = 'categorical_crossentropy'
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)
        model = Model(inputs=inp, outputs=output)
        model.compile(
            loss=loss,
            optimizer=adam,
            metrics=["categorical_accuracy"])
        return model
