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

# --- Functions to process trainging and testing data ---
# mapping function from characters to integers
def letter_to_int(char_array):
    # --- Create a dictionary for all the letters & start/stops ---
    alphabet = np.array([i for i in range(1, 31)]) # All letters plus STARTSENTENCE, STOPSENTENCE, start, stop
    mapping = dict()
    for i in range(len(alphabet) - 4):
        mapping[chr(ord('a') + i)] = alphabet[i]

    mapping['start'] = alphabet[26]
    mapping['stop']  = alphabet[27]
    mapping['STARTSETNENCE'] = alphabet[28]
    mapping['STOPSENTENCE']  = alphabet[29]
    
    # --- Map the characters in the input array to integers ---
    x_input = char_array
    x_input = [list(i) for i in x_input]
    X = []
    for word in x_input:
        X.append([mapping[sym] for sym in word])
    X = np.array(X)
    
    # --- Create Y, preY, postY ---
    Y = []
    for word in X:
        Y.append(np.concatenate((np.array([27]), word, np.array([28])), axis=0))

    Y = np.array(Y)
    preY  = Y[:, :-1]
    postY = Y[:, 1:]
    
    return X, Y, preY, postY, mapping

def int_to_letter(encoding, mapping):
    enc_shape = encoding.shape
    
    flat_encoding = encoding.flatten() # Flatten array to just one dimension
    
    # list out keys and values separately
    key_list = list(mapping.keys())
    val_list = list(mapping.values())

    integers = []
    for letter in flat_encoding:
        integers.append(key_list[val_list.index(letter)])
        
    integers = np.array(integers)
    integers = np.reshape(integers, enc_shape)
    return integers


def MaskedSparseCategoricalCrossentropy(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def MaskedSparseCategoricalAccuracy(real, pred):
    accuracies = tf.equal(tf.cast(real,tf.int64), tf.argmax(pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

# --- Make functions to decypher model's output ---
def word_accuracy(output,check):
    count = 0
    for i, j in zip(output, check):
            if(np.array_equal(i,j)):
                count +=1


    word_accuracy = count / len(output)
    return word_accuracy

def letter_accuracy(output, check):
    count = 0
    for i, j in zip(output, check):
            for x, y in zip(i,j):
                if(x == y):
                    count += 1


    letter_accuracy = count / (len(output[0]) * len(output))
    return letter_accuracy