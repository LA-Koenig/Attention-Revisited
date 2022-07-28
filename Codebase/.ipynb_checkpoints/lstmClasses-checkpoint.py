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