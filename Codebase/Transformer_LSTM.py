#!/usr/bin/env python3
# coding: utf-8

# # Preliminary

# In[1]:


import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
import sys

import numpy as np
import pickle, random, string
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

# Visualization
#from IPython.display import display

from pathlib import Path


# In[2]:


# Ugly patch becasue idk how to make this propperly work
#sys.path[0] += '/Codebase'
sys.path


# In[3]:


#my classes
#from testClass import *
from transformerClasses import *
from embeddingClasses import *
from lstmClasses import *
from resourceFunctions import *


# In[4]:


strategy = tf.distribute.OneDeviceStrategy('gpu:1')
#Unsure if this is needed...


# # Load outer Model

# In[5]:


# setting up the pathlib stuff 
pathBase = Path('.')
print([x for x in path.iterdir() if x.is_dir()])

path = pathBase / 'saved-models'
outPath = path / 'outer_encdec_intembed'

encPathJson = outPath / 'encoder_len5_J_10000_intembed.json'
decPathJson = outPath / 'decoder_len5_J_10000_intembed.json'
encPathH5 = outPath / 'encoder_len5_J_10000_intembed.h5'
decPathH5 = outPath / 'decoder_len5_J_10000_intembed.h5'


# In[6]:


with encPathJson.open() as encoder_file, decPathJson.open() as decoder_file:
    encoder_json = encoder_file.read()
    decoder_json = decoder_file.read()
    
outer_encoder = keras.models.model_from_json(encoder_json)
outer_decoder = keras.models.model_from_json(decoder_json)

outer_encoder.load_weights(encPathH5)
outer_decoder.load_weights(decPathH5)


# In[7]:


keras.utils.plot_model(outer_encoder, show_shapes=True)


# In[8]:


keras.utils.plot_model(outer_decoder, show_shapes=True)


# # Load Input Data

# In[9]:


# Load training and testing data

corpus = np.loadtxt(pathBase / 'data' / sys.argv[1], dtype=object)
trainingSet = np.loadtxt(pathBase / 'data' / sys.argv[2], dtype=object)
testingSet  = np.loadtxt(pathBase / 'data' / sys.argv[3], dtype=object)


# #corpus = np.loadtxt(sys.argv[1], dtype=object)
# corpus = (Path('..') / 'data' / 'len5_10000-train.txt' ).open() #open('../data/len5_10000-train.txt')
# corpus = np.loadtxt(corpus, dtype=object)

# trainingSet = (Path('..') / 'data' / 'SG-10-train.txt').open()
# testingSet  = (Path('..') / 'data' / 'SG-10-test.txt').open()

# trainingSet = np.loadtxt(trainingSet, dtype=str)
# testingSet  = np.loadtxt(testingSet, dtype=str)
#This is the same accross all files




# # Create Embeddings

# In[11]:


# --- Create a dictionary for all the letters & start/stops ---
alphabet = np.array([i for i in range(1, 31)]) # All letters plus STARTSENTENCE, STOPSENTENCE, start, stop
mapping = dict()
for i in range(len(alphabet) - 4):
    mapping[chr(ord('a') + i)] = alphabet[i]

mapping['start'] = alphabet[26]
mapping['stop']  = alphabet[27]
mapping['STARTSETNENCE'] = alphabet[28]
mapping['STOPSENTENCE']  = alphabet[29]


# In[12]:


# --- Map words from corpus to words to roles ---
encoded_mapping = {}
selected_words = {}
for letter in string.ascii_lowercase[:10]:
    # Store the letter with the word for use in testing
    rand_corpus_word = random.choice(corpus)

    word, Y, preY, postY = word_to_int(rand_corpus_word, mapping)
    
    selected_words[letter] = postY
    
    encoded_mapping[letter] = outer_encoder.predict(np.array([word]))


# In[13]:


#--- Pre input to make encodings at the sentence level --- 
roles   = trainingSet #argv[2]    
x_train = []
for sentence in roles:
    x_train.append([encoded_mapping[letter] for letter in sentence])
x_train = np.array(x_train) # shape (n, 3, 2, 1, 50)

LENGTH_IDK  = x_train.shape[-1] # Replacing '50' in the code

t1 = x_train[:,:,0,0,:] # new shape (n,3,50)
t2 = x_train[:,:,1,0,:] # " '' "
# 4 time steps. pre
pre_t1 = np.concatenate((np.zeros((x_train.shape[0],1,LENGTH_IDK)), t1), axis = 1) # Orig. (x_train.shape[0], 1, 50)
pre_t2 = np.concatenate((np.zeros((x_train.shape[0],1,LENGTH_IDK)), t2), axis = 1)
post_t1 = np.concatenate((t1, np.zeros((x_train.shape[0],1,LENGTH_IDK))), axis = 1)
post_t2 = np.concatenate((t2, np.zeros((x_train.shape[0],1,LENGTH_IDK))), axis = 1)



# In[14]:


# Start or stop tokens
s_s = {"start": [0,1], "stop": [1,0], "none": [0,0]}
pre_start = np.zeros((x_train.shape[0], 4, 2))
post_stop = np.copy(pre_start)
pre_start[:,0,:] = s_s["start"]
post_stop[:,3,:] = s_s["stop"]

#This is the same accross every file


# In[ ]:





# In[15]:


print("shapes:")
print("\n\tx_train: ", x_train.shape)

print("\n\tt1: ", t1.shape)
print("\tt2: ", t2.shape)

print("\n\tpre_t1: ", pre_t1.shape)
print("\tpre_t2: ", pre_t2.shape)

print("\n\tpost_t1: ", post_t1.shape)
print("\tpost_t2: ", post_t2.shape)

print("\n\tpre_start: ", pre_start.shape)
print("\tpost_stop: ", post_stop.shape)


# # Inner Transformer

# In[16]:


#Now I'm working from the Transformer_Transformer file


# In[17]:


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


# In[18]:


# --- Model parameters ---
length = 10
padded_length = 20


# In[19]:


# Note we are making these the same, but they don't -have- to be!
input_length = padded_length
output_length = padded_length


# In[20]:


# Vocabulary sizes...
encoder_vocab_size = 30 # a, b, c, ... z, start, stop, STARTSENTENCE, STOPSENTENCE
decoder_vocab_size = 30 # a, b, c, ... z, start, stop, STARTSENTENCE, STOPSENTENCE


# In[21]:


# Size of the gestalt, context representations...
embed_dim = 128  # Embedding size for each token (enc/dec inputs already embedded) 
#nothing should rely on this??

#the rest of this has been update to the paper's
num_heads = 4  # Number of attention heads
ff_dim = 4  # Hidden layer size in feed forward network inside transformer
stack = 1
wd = 0.01 #unsure what this is


# In[22]:


HIDDEN_SIZE = 300 #outer size????

#updated to paper
BATCH_SIZE  = 100
EPOCHS      = 1600


# In[23]:


# --- Construct inner encoder/decoder ---
with strategy.scope():
    # Encoder
    
    encoder_input_t1 = keras.layers.Input(shape=(None, t1.shape[2]), name="enc_token_1")
    encoder_input_t2 = keras.layers.Input(shape=(None, t1.shape[2]), name="enc_token_2")
    encoder_input = keras.layers.Concatenate()([encoder_input_t1, encoder_input_t2])
    

    encoder_mask_pos_embedding = InnerMaskedPositionEmbedding(maxlen=input_length,
                                                              embed_dim=encoder_input.shape[-1])(encoder_input)#(encoder_embedding)

    encoder_state = InnerTransformerBlock(embed_dim=encoder_input.shape[-1], 
                                          num_heads=num_heads,ff_dim=ff_dim)(encoder_mask_pos_embedding)
    
    encoder_model = keras.Model([encoder_input_t1, encoder_input_t2],encoder_state,name="InnerEncoder")
    

    
    # Decoder
    
    decoder_input_t1 = keras.layers.Input(shape=(None, pre_t1.shape[2]), name="dec_token_1")
    decoder_input_t2 = keras.layers.Input(shape=(None, pre_t2.shape[2]), name="dec_token_2")

    decoder_startstop = keras.layers.Input(shape=(None, 2), name="dec_start/stop")
    
    decoder_concat = keras.layers.Concatenate()([decoder_input_t1, decoder_input_t2])
    
    decoder_s_s_dense = keras.layers.Dense(decoder_concat.shape[-1], use_bias=False)(decoder_startstop)

    decoder_context_input = keras.layers.Input(shape=encoder_state.shape[1:], name='inner_enc_state')

    decoder_mask_pos_embedding = InnerMaskedPositionEmbedding(maxlen=pre_t1.shape[1],
                                                         embed_dim=decoder_concat.shape[-1])(decoder_concat)
    decoder_add = keras.layers.Add()([decoder_mask_pos_embedding, decoder_s_s_dense])

    decoder_block = InnerMaskedTransformerBlock(embed_dim=decoder_add.shape[-1],
                                           num_heads=num_heads,
                                           ff_dim=ff_dim)

    decoder_hidden_output = decoder_block([decoder_add, decoder_context_input])
    

    decoder_dense_t1 = keras.layers.Dense(post_t1.shape[-1], activation='linear')(decoder_hidden_output)
    decoder_dense_t2 = keras.layers.Dense(post_t2.shape[-1], activation='linear')(decoder_hidden_output)

    decoder_dense_startstop = keras.layers.Dense(2, activation='sigmoid', name="start/stop")(decoder_hidden_output)
    
    
    
    #moved the following line for better organization. Was above. 
    decoder_inputs = [decoder_context_input, decoder_input_t1, decoder_input_t2, decoder_startstop]

    decoder_outputs = [decoder_dense_t1, decoder_dense_t2, decoder_dense_startstop]

    decoder_model = keras.Model(decoder_inputs,decoder_outputs,name="InnerDecoder")

    
    # Tie encoder and decoder into one model
    # with strategy.scope():
    #model = keras.Model([encoder_input]+ decoder_inputs, decoder_outputs)
    coupled_inputs = [keras.layers.Input(encoder_model.inputs[0].shape[1:]),
                      keras.layers.Input(encoder_model.inputs[1].shape[1:]),
                      keras.layers.Input(decoder_model.inputs[1].shape[1:]),
                      keras.layers.Input(decoder_model.inputs[2].shape[1:]),
                      keras.layers.Input(decoder_model.inputs[3].shape[1:])]                     
    coupled_outputs = decoder_model([encoder_model(coupled_inputs[0:2])] + coupled_inputs[2:])
    model = keras.Model(coupled_inputs, coupled_outputs)

    # --- Compile and fit model ---
    model.compile(loss = [keras.losses.MSE,keras.losses.MSE,keras.losses.binary_crossentropy],
               optimizer=keras.optimizers.Adam(),
               metrics=['accuracy'])
    
    model_input = {"enc_token_1": t1, "enc_token_2": t2, "dec_token_1": pre_t1, "dec_token_2": pre_t2, "dec_start/stop": pre_start}
    model_target = {"token_1": post_t1, "token_2": post_t2, "start/stop": post_stop}

    #with strategy.scope():
    history = model.fit([t1,t2,pre_t1,pre_t2,pre_start],
                        [post_t1, post_t2, post_stop],
                         batch_size=BATCH_SIZE,
                         epochs=EPOCHS,
                         verbose=0)


# In[24]:


keras.utils.plot_model(encoder_model, expand_nested=True, show_shapes=True)


# In[25]:


keras.utils.plot_model(decoder_model, expand_nested=True, show_shapes=True)


# In[26]:


keras.utils.plot_model(model, expand_nested=True, show_shapes=True)


# In[27]:


history.history.keys()


# In[28]:


plt.figure(1)  
# summarize history for accuracy 
plt.subplot(211)  
plt.plot(history.history['InnerDecoder_accuracy'])  
plt.plot(history.history['InnerDecoder_1_accuracy']) 
plt.plot(history.history['InnerDecoder_2_accuracy']) 
plt.title('model accuracy')  
plt.ylabel('reshape_2_accuracy')  
plt.xlabel('epoch')  
# summarize history for loss  
plt.subplot(212)  
plt.plot(history.history['InnerDecoder_loss'])
plt.plot(history.history['InnerDecoder_1_loss'])  
plt.plot(history.history['InnerDecoder_2_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.tight_layout()
plt.show()  


# In[29]:


model.evaluate([t1,t2,pre_t1,pre_t2,pre_start],[post_t1, post_t2, post_stop])


# # Testing

# In[34]:


with strategy.scope():
    x_test = []
    correct_result = [] # used to get accuracy at end
    roles = testingSet #argv[3]
    for sentence in roles:
        x_test.append([encoded_mapping[letter] for letter in sentence])
        correct_result.append([selected_words[letter] for letter in sentence])

    x_test = np.array(x_test) # shape (n, 3, 2, 1, 50)
    correct_result = np.array(correct_result)

    t1_test = x_test[:,:,0,0,:] # new shape (n,3,50)
    t2_test = x_test[:,:,1,0,:] # " '' "
    # 4 time steps. pre
    pre_t1_test = np.concatenate((np.zeros((x_test.shape[0],1,LENGTH_IDK)), t1_test), axis = 1)
    pre_t2_test = np.concatenate((np.zeros((x_test.shape[0],1,LENGTH_IDK)), t2_test), axis = 1)

    # Start tokens
    pre_start_test = np.zeros((x_test.shape[0], 4, 2))
    pre_start_test[:,0,:] = s_s["start"]
    #Changed these to start and stop to be consistent with above
    
    outer_result = np.empty((len(x_test),3,6)) 

        #from the LSTM/LSTM file.

    for i, sentence in enumerate(x_test):
        context = encoder_model.predict({"enc_token_1": t1_test[i:i+1], "enc_token_2": t2_test[i:i+1]})
        dec_t1 = np.zeros((1,4,LENGTH_IDK))
        dec_t2 = np.zeros((1,4,LENGTH_IDK))
        dec_s_s = pre_start[0:1,0:4,:]
        inner_result = np.zeros([4,2,LENGTH_IDK])
        output_length = 3

        # obtain the result from the inner decoder
        for x in range(output_length+1):
            out1, out2, out3= decoder_model.predict({"inner_enc_state": context, 
                                             "dec_token_1": dec_t1,
                                             "dec_token_2": dec_t2,
                                             "dec_start/stop": dec_s_s})
            #context = h#[h,c]
            dec_t1[:,x+1:x+2,:] = out1[:,x:x+1,:]
            dec_t2[:,x+1:x+2,:] = out2[:,x:x+1,:]
            dec_s_s[:,x+1:x+2,:] = np.round(out3[:,x:x+1,:])
        inner_result[:,0,:] = out1[0:1]
        inner_result[:,1,:] = out2[0:1]

        # obtain the result from the outer decoder
        output_length = 5
        for word in range(3):
            context = []
            context.append(inner_result[word,0:1,:])
            context.append(inner_result[word,1:2,:])
            token = np.array(mapping["start"])
            token = token.reshape([1, 1, 1])
            for letter in range(output_length + 1):
                out, h, c = outer_decoder.predict([token] + context)
                token = Out_to_int(out)
                context = [h,c]
                outer_result[i, word, letter] = token


# In[35]:


# --- Get letter and word level accuracy ---
word_accuracy = 0
letter_accuracy = 0
for answer, response in zip(correct_result, outer_result):
    # check each word
    for word in range(3):
        if np.array_equal(answer[word,:], response[word,:]):
            word_accuracy += 1
            letter_accuracy += 6
        #check each letter
        else:
            for letter in range(6):
                if np.array_equal(answer[word,letter], response[word,letter]):
                    letter_accuracy += 1
                    
word_accuracy /= float(correct_result.shape[0] * 3)
letter_accuracy /= float(correct_result.shape[0] * 3 * 6)


# In[36]:


print('-----------------------------')
print('''   Generalization Accuracy
-----------------------------
word_accuracy: %f
letter_accuracy: %f
'''%((word_accuracy*100), (letter_accuracy*100)))


# In[37]:

# --- Write results to file ---
#my_file = str(sys.argv[2][0:2]) + "_output.txt"
#I'll think about how best to rework this when I start running it in a batch

#until then
my_file = "compiled_output.txt"

file_object = open(my_file, 'a')
# Append 'hello' at the end of file

file_object.write("\n\n========================================================\n")
file_object.write("Transformer_LSTM tested with " + sys.argv[2] + '\n')

file_object.write("word_accuracy: " + str(word_accuracy*100) + '\n')

file_object.write("letter_accuracy: " + str(letter_accuracy*100) + '\n')
# Close the file
file_object.close()




