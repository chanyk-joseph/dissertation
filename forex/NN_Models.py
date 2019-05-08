import numpy as np

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import CuDNNLSTM
LSTM = CuDNNLSTM

class NN_Models:
    def get_LSTM_BERT(self, SEQ_LEN, FUTURE_PERIOD_PREDICT):
        class LayerNormalization(Layer):
            def __init__(self, eps=1e-6, **kwargs):
                self.eps = eps
                super(LayerNormalization, self).__init__(**kwargs)
            def build(self, input_shape):
                self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                            initializer=Ones(), trainable=True)
                self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                            initializer=Zeros(), trainable=True)
                super(LayerNormalization, self).build(input_shape)
            def call(self, x):
                mean = K.mean(x, axis=-1, keepdims=True)
                std = K.std(x, axis=-1, keepdims=True)
                return self.gamma * (x - mean) / (std + self.eps) + self.beta
            def compute_output_shape(self, input_shape):
                return input_shape

        class ScaledDotProductAttention():
            def __init__(self, d_model, attn_dropout=0.1):
                self.temper = np.sqrt(d_model)
                self.dropout = Dropout(attn_dropout)
            def __call__(self, q, k, v, mask):
                attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/self.temper)([q, k])
                if mask is not None:
                    mmask = Lambda(lambda x:(-1e+10)*(1-x))(mask)
                    attn = Add()([attn, mmask])
                attn = Activation('softmax')(attn)
                attn = self.dropout(attn)
                output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
                return output, attn

        class MultiHeadAttention():
            # mode 0 - big martixes, faster; mode 1 - more clear implementation
            def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
                self.mode = mode
                self.n_head = n_head
                self.d_k = d_k
                self.d_v = d_v
                self.dropout = dropout
                if mode == 0:
                    self.qs_layer = Dense(n_head*d_k, use_bias=False)
                    self.ks_layer = Dense(n_head*d_k, use_bias=False)
                    self.vs_layer = Dense(n_head*d_v, use_bias=False)
                elif mode == 1:
                    self.qs_layers = []
                    self.ks_layers = []
                    self.vs_layers = []
                    for _ in range(n_head):
                        self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                        self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                        self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
                self.attention = ScaledDotProductAttention(d_model)
                self.layer_norm = LayerNormalization() if use_norm else None
                self.w_o = TimeDistributed(Dense(d_model))

            def __call__(self, q, k, v, mask=None):
                d_k, d_v = self.d_k, self.d_v
                n_head = self.n_head

                if self.mode == 0:
                    qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
                    ks = self.ks_layer(k)
                    vs = self.vs_layer(v)

                    def reshape1(x):
                        s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
                        x = tf.reshape(x, [s[0], s[1], n_head, d_k])
                        x = tf.transpose(x, [2, 0, 1, 3])  
                        x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
                        return x
                    qs = Lambda(reshape1)(qs)
                    ks = Lambda(reshape1)(ks)
                    vs = Lambda(reshape1)(vs)

                    if mask is not None:
                        mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
                    head, attn = self.attention(qs, ks, vs, mask=mask)  
                        
                    def reshape2(x):
                        s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
                        x = tf.reshape(x, [n_head, -1, s[1], s[2]]) 
                        x = tf.transpose(x, [1, 2, 0, 3])
                        x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
                        return x
                    head = Lambda(reshape2)(head)
                elif self.mode == 1:
                    heads = []; attns = []
                    for i in range(n_head):
                        qs = self.qs_layers[i](q)   
                        ks = self.ks_layers[i](k) 
                        vs = self.vs_layers[i](v) 
                        head, attn = self.attention(qs, ks, vs, mask)
                        heads.append(head); attns.append(attn)
                    head = Concatenate()(heads) if n_head > 1 else heads[0]
                    attn = Concatenate()(attns) if n_head > 1 else attns[0]

                outputs = self.w_o(head)
                outputs = Dropout(self.dropout)(outputs)
                if not self.layer_norm: return outputs, attn
                # outputs = Add()([outputs, q]) # sl: fix
                return self.layer_norm(outputs), attn

        def build_model():
            inp = Input(shape = (SEQ_LEN, 1))
            
            # LSTM before attention layers
            x = Bidirectional(LSTM(128, return_sequences=True))(inp)
            x = Bidirectional(LSTM(64, return_sequences=True))(x) 
                
            x, slf_attn = MultiHeadAttention(n_head=3, d_model=300, d_k=64, d_v=64, dropout=0.1)(x, x, x)
                
            avg_pool = GlobalAveragePooling1D()(x)
            max_pool = GlobalMaxPooling1D()(x)
            conc = concatenate([avg_pool, max_pool])
            conc = Dense(64, activation="relu")(conc)
            x = Dense(FUTURE_PERIOD_PREDICT, activation="sigmoid")(conc)      

            model = Model(inputs = inp, outputs = x)
            model.compile(
                loss = "mean_squared_error", 
                #optimizer = Adam(lr = config["lr"], decay = config["lr_d"]), 
                optimizer = "adam")
            
            # Save entire model to a HDF5 file
            #model.save('my_model.h5')
            
            return model

        multi_head = build_model()
        return multi_head