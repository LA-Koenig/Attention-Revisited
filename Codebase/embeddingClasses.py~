# From the Attention Is Not Enough Repository
# This is a reorganization of the code written during the 2020 summer COMS REU



#idk how much of this I actually need
import numpy as np
import sys
import pickle, random, string
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf
import os.path

strategy = tf.distribute.OneDeviceStrategy('gpu:1')


class OuterPositionEmbedding(keras.layers.Layer):
    def __init__(self,maxlen,embed_dim, *args, **kwargs):
        super(OuterPositionEmbedding, self).__init__(*args,**kwargs)
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen,
                                              output_dim=embed_dim)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        
    def call(self,x):
        maxlen = tf.shape(x)[1]
        print(maxlen)
        positions = tf.range(start=0,limit=maxlen,delta=1)
        positions = self.pos_emb(positions)
        print(tf.shape(positions))
        print(tf.shape(x))
        return x + positions
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'maxlen': self.maxlen,
            'embed_dim': self.embed_dim 
        })
        return config
    
    
    
class OuterMaskedTokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, *args, **kwargs):
        super(OuterMaskedTokenAndPositionEmbedding, self).__init__(*args, **kwargs)
        self.token_emb = keras.layers.Embedding(input_dim=vocab_size,
                                                output_dim=embed_dim,
                                                mask_zero=True)
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen+1,
                                              output_dim=embed_dim,
                                              mask_zero=True)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=1, limit=maxlen+1, delta=1)
        positions = positions * tf.cast(tf.sign(x),tf.int32)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim
        })
        return config
    
    
# Inner model stuff
class InnerMaskedPositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(InnerMaskedPositionEmbedding, self).__init__()
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen+1,
                                              output_dim=embed_dim,
                                              mask_zero=True)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        
    def compute_output_shape(self, input_shape):
        return input_shape + (embed_dim,)
    
    def call(self, x):
        maxlen = tf.shape(x)[-2]
        positions = tf.range(start=1, limit=maxlen+1, delta=1)
        positions = positions * tf.cast(tf.sign(tf.math.count_nonzero(x,axis=2)),tf.int32)
        positions = self.pos_emb(positions)
        return x + positions

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
