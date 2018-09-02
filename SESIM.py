
# coding: utf-8

# In[1]:


# coding: utf-8

from __future__ import print_function, unicode_literals
import os
import sys
import time
import pickle
from absl import logging

import numbers
import numpy as np
import pandas as pd
from numpy import dot
from pandas import DataFrame 
from numpy.linalg import norm
from itertools import combinations

import keras
import jieba
import jieba.posseg as pseg
from word2vec import KeyedVectors
import synonyms
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.externals import joblib
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.core import Dense, Dropout, Activation, Flatten

# 配置tensorflow利用显存方式
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth=True 
#config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))


# In[4]:


# ---------------------------------------------
def _load_stopwords(file_path):
    words = open(file_path, 'r', encoding='utf-8')
    stopwords = words.readlines()
    _stopwords = set()
    for w in stopwords:
        _stopwords.add(w.strip())
    return _stopwords


# ---------------------------------------------
def _load_w2v(model_file, binary=True):
    if not os.path.exists(model_file):
        print("os.path : ", os.path)
        raise Exception("Word2vec file [%s] does not exist." % model_file)
    return KeyedVectors.load_word2vec_format(
        model_file, binary=binary, unicode_errors='ignore')



# ---------------------------------------------
# internal variables init
# ---------------------------------------------
num_r = 10
num_c = 10
_vectors = None
_stopwords = set()
_cache_nearby = dict()
_cache_scores = dict()
    
try: 
    curdir = os.path.dirname(os.path.abspath(__file__)) 
    
    # load user defined dict
    _file_name = os.path.join(curdir, 'data', 'userdict.txt')
    print(">> SESIM on loading user defined dict [%s] ..." % _file_name)
    jieba.load_userdict(_file_name)
    
    # load stop words
    _file_name = os.path.join(curdir, 'data', 'stopwords.txt')
    print(">> SESIM on loading stopwords [%s] ..." % _file_name)
    _stopwords = _load_stopwords(_file_name)
    
    # load user defined similarity
    _file_name = os.path.join(curdir, 'data', 'sim_pairs.csv')
    print(">> SESIM on loading user defined similarity pairs [%s] ..." % _file_name)
    _sim_pairs = pd.read_csv(_file_name, sep='\t', header=None, names=['score','term1','term2'])
    
    _file_name = os.path.join(curdir, 'data', 'words.vector')
    #_file_name = "./data/GoogleNews-vectors-negative300.bin"
    print(">> SESIM on loading word2vectors [%s] ..." % _file_name)
    _vectors = _load_w2v(_file_name)      
    
    #load cached scores
    _cache_scores = joblib.load(os.path.join(curdir, 'data', '_cache_scores.pkl'))
    
except:     
    # load user defined dict
    _file_name = "./data/userdict.txt"
    print(">> SESIM on loading user defined dict [%s] ..." % _file_name)
    jieba.load_userdict(_file_name)
    
    # load stop words
    _file_name = "./data/stopwords.txt"
    print(">> SESIM on loading stopwords [%s] ..." % _file_name)
    _stopwords = _load_stopwords(_file_name)
    
    # load user defined similarity
    _file_name = "./data/sim_pairs.csv"
    print(">> SESIM on loading user defined similarity pairs [%s] ..." % _file_name)
    _sim_pairs = pd.read_csv(_file_name, sep='\t', header=None, names=['score','term1','term2'])
    
    # word embedding
    # ---------------------------------------------
    _file_name = "./data/words.vector"
    #_file_name = "./data/GoogleNews-vectors-negative300.bin"
    print(">> SESIM on loading word2vectors [%s] ..." % _file_name)
    _vectors = _load_w2v(_file_name)
    
    #load cached scores
    try:
        _cache_scores = joblib.load('./data/_cache_scores.pkl')
    except: pass


# In[17]:


# shared functions for both training and test

# model definition
# ---------------------------------------------
def gen_model(input_dim, N_nodes, r_droupout):
    model = Sequential()
    model.add(Dense(N_nodes, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dropout(r_droupout))
    model.add(Dense(N_nodes))
    model.add(Activation('relu'))
    model.add(Dropout(r_droupout))
    model.add(Dense(N_nodes))
    model.add(Activation('relu'))
    model.add(Dropout(r_droupout))
    model.add(Dense(N_nodes))
    model.add(Activation('relu'))
    model.add(Dropout(r_droupout))
    model.add(Dense(N_nodes))
    model.add(Activation('relu'))
    model.add(Dropout(r_droupout))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(), metrics=['accuracy'])
    return model


# To get similarity score of words w1 and w2
# ---------------------------------------------
def get_sesim_word(w1, w2): 
    tscore = 0
    # read from cache
    if ((w1,w2) in _cache_scores):
        tscore = _cache_scores[(w1,w2)]
    elif ((w2,w1) in _cache_scores):   
        tscore = _cache_scores[(w2,w1)]
    else:
        print('query:\t{}\t{}'.format(w1,w2))
        tscore = synonyms.compare(w1, w2)
        if np.isnan(tscore):
            if(w1 in _stopwords or w2 in _stopwords): tscore = 0
            if w1 == w2: tscore = 1
            idx   = (_sim_pairs['term1']==w1) & (_sim_pairs['term2']==w2)            
            if(sum(idx)): tscore = _sim_pairs['score'][idx].values[0]
            idx = (_sim_pairs['term1']==w2) & (_sim_pairs['term2']==w1)
            if(sum(idx)): tscore = _sim_pairs['score'][idx].values[0] 
        # put into cache
        _cache_scores[(w1,w2)] = tscore
        _cache_scores[(w2,w1)] = tscore
    return tscore


# To get cov-similarity feature of sentences s1 and s2
# ---------------------------------------------
def get_feat(s1, s2):
    mat_cor = np.zeros(num_r*num_c).reshape(num_r,num_c)
    try:
        words1 = jieba.lcut(s1)
        words2 = jieba.lcut(s2)
    except:
        print('Jieba cut error! \ts1: {},\ts2: {}'.format(s1, s2))
    idx1 = 0
    for word1 in words1:
        idx2 = 0
        for word2 in words2:
            if idx1>=num_r:
                continue
            if idx2>=num_c:
                continue
            mat_cor[idx1,idx2] = get_sesim_word(word1, word2)
            idx2 = idx2 + 1
        idx1 = idx1 + 1
    data = mat_cor.reshape(1,-1)
    data[np.isnan(data)]= 0
    return data    


# To get similarity score between sentences s1 and s2
# ---------------------------------------------
def get_sesim_sentence(s1, s2):  
    data = get_feat(s1, s2)
    with graph.as_default():
        sid_probs = model.predict(data, verbose=0)
    return sid_probs[0]


# To get similarities of a sentence s1 and a batch sentences s2s
# ---------------------------------------------
def get_sesim_sentence_batch(s1, s2s):
    data = pd.DataFrame()
    for s2 in s2s:
        data = data.append(pd.DataFrame(get_feat(s1,s2).reshape(1,-1)))
    with graph.as_default():
        sid_probs = model.predict(data, verbose=0)
    return sid_probs


# To generate sts (score, s1, s2) triples from "# sentences groups" formate file
# ---------------------------------------------
def s2sts(filename):
    group = []
    sentences_semantic = []
    with open(filename, encoding="utf-8") as file:
        lines = [l.strip() for l in file]
    for line in lines:
        if len(line)==0:
            continue
        if line[0] == '#':
            if len(group)>0:
                sentences_semantic.append(group)
            group = []
            continue
        group.append(line)
    sentences_semantic.append(group)
    #print("\n".join(str(line) for line in sentences_semantic))    
    data_STS = pd.DataFrame(columns=['score','s1','s2'])
    n_groups = len(sentences_semantic)
    for idxg in range(n_groups):
        group = sentences_semantic[idxg]
        idxs = range(len(group))
        combs = combinations(idxs, 2)
        for icomb in combs:
            data_STS = data_STS.append({'score':5, 's1':group[icomb[0]],                                        's2':group[icomb[1]]},ignore_index=True)
            idxg_t = np.random.choice(n_groups, 1)[0]
            if idxg_t == idxg: 
                idxg_t = idxg_t - 1
            data_STS = data_STS.append({'score':0,'s1':group[icomb[0]],                                        's2':sentences_semantic[idxg_t][0]},                                       ignore_index=True)
    return data_STS


# To generate cov similarity matrix for training from sts data
# ---------------------------------------------
def sts2mat(df):
    s1s = df[['s1']]
    s2s = df[['s2']]
    scores = df[['score']]
    data = pd.DataFrame()
    for idx in range(s1s.shape[0]):
        score = scores.iat[idx,0]
        s1 = s1s.iat[idx,0]
        s2 = s2s.iat[idx,0]
        data = data.append(pd.DataFrame(np.append(get_feat(s1, s2), score).reshape(1,-1) ))
    data = data.values
    data[np.isnan(data)]=0
    return data


# In[8]:


# generate open-domain training data
# ---------------------------------------------
if 1:
    data_STS = pd.read_csv("./data/STSBenchmark/sts-train.csv",
                           sep='\t',
                           header=None, 
                           names=['genre','filename','year','id','score','s1','s2'])
    data = sts2mat(data_STS)
    joblib.dump(data, './data/STSBenchmark_train_'+str(num_r)+'x'+str(num_c)+'.pkl')
    print('STSBenchmark data: {}'.format(data.shape))
    
    data_STS = pd.read_csv("./data/STSBenchmark/sts-test.csv",
                           sep='\t',
                           header=None,
                           names=['genre','filename','year','id','score','s1','s2'])
    data = sts2mat(data_STS)
    joblib.dump(data, './data/STSBenchmark_test_'+str(num_r)+'x'+str(num_c)+'.pkl')
    print('STSBenchmark data: {}'.format(data.shape))


# generate specific-domain training data
# ---------------------------------------------

joblib.dump(_cache_scores, './data/_cache_scores.pkl')


# In[ ]:


# load data for training process
# ---------------------------------------------
data_train  = joblib.load('./data/STSBenchmark_train_'+str(num_r)+'x'+str(num_c)+'.pkl')
data_test   = joblib.load('./data/STSBenchmark_test_'+str(num_r)+'x'+str(num_c)+'.pkl')
data_sts = np.append(data_train, data_test, axis =0)

# In[ ]:


# data load 
per_train = 0.9
if 1:
    x0 = data_sts[:,0:-1]
    y0 = data_sts[:,-1].astype(int)
    y0[y0<2.5] = 0
    y0[y0>2.5] = 1
    x1, y1  = shuffle(x0, y0)
    X_train = x1[0:int(x0.shape[0] * per_train)]
    X_test  = x1[int(x0.shape[0] * per_train)::]
    Y_train = y1[0:int(x0.shape[0] * per_train)]
    Y_test  = y1[int(x0.shape[0] * per_train)::]


# In[ ]:


N_batch_SC = 128
N_epoch_SC = 512
en_verbose = 0

# 1-dimensional MSE linear regression in Keras
model = Sequential()
model.add(Dense(1, input_dim=x0.shape[1]))
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=N_epoch_SC, verbose=en_verbose, validation_data=(X_test, Y_test))
print('\n1-dimensional MSE linear regression:')
score_train = model.evaluate(X_train, Y_train, verbose=0)
print('Train accuracy:\t{}'.format(score_train[1]))
score_test = model.evaluate(X_test, Y_test, verbose=0)
print('Test accuracy:\t{}'.format(score_test[1]))
score_whole = model.evaluate(x0, y0, verbose=0)
print('Whole accuracy:\t{}'.format(score_whole[1]))

# 2-class logistic regression in Keras
model = Sequential()
model.add(Dense(1, activation='sigmoid', input_dim=x0.shape[1]))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=N_epoch_SC, verbose=en_verbose, validation_data=(X_test, Y_test))
print('\n2-class logistic regression:')
score_train = model.evaluate(X_train, Y_train, verbose=0)
print('Train accuracy:\t{}'.format(score_train[1]))
score_test = model.evaluate(X_test, Y_test, verbose=0)
print('Test accuracy:\t{}'.format(score_test[1]))
score_whole = model.evaluate(x0, y0, verbose=0)
print('Whole accuracy:\t{}'.format(score_whole[1]))

# logistic regression with L1 and L2 regularization
from keras.regularizers import l1_l2
reg = l1_l2(l1=0.01, l2=0.01)
model = Sequential()
model.add(Dense(1, activation='sigmoid', W_regularizer=reg, input_dim=x0.shape[1]))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=N_epoch_SC, verbose=en_verbose, validation_data=(X_test, Y_test))
print('\nlogistic regression with L1 and L2 regularization:')
score_train = model.evaluate(X_train, Y_train, verbose=0)
print('Train accuracy:\t{}'.format(score_train[1]))
score_test = model.evaluate(X_test, Y_test, verbose=0)
print('Test accuracy:\t{}'.format(score_test[1]))
score_whole = model.evaluate(x0, y0, verbose=0)
print('Whole accuracy:\t{}'.format(score_whole[1]))


# In[ ]:


# CNN model par
N_nodes = X_train.shape[1]
r_droupout = 0.1
N_batch_SC = 128
N_epoch_SC = 2048
en_verbose = 0

N_tries = 5
thres_score = 0.1

the_one = thres_score
for k in range(N_tries):
    # model train
    model = gen_model(X_train.shape[1], N_nodes, r_droupout) #input_dim, N_nodes, r_droupout
    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(), metrics=['accuracy'])
    # model training
    start   = time.time()
    history = model.fit(X_train, Y_train,
                        batch_size=N_batch_SC, epochs=N_epoch_SC,
                        verbose=en_verbose, validation_data=(X_test, Y_test))    
    end     = time.time()
    print('time elapse training:\t', end - start, 'sec')
    
    # model evaluation
    start = time.time()
    score_train = model.evaluate(X_train, Y_train, verbose=0)
    end = time.time()
    print('time elapse:\t{} sec'.format(end - start))
    print('Train accuracy:\t{}'.format(score_train[1]))
    start = time.time()
    score_test = model.evaluate(X_test, Y_test, verbose=0)
    end = time.time()
    print('time elapse:\t{} sec'.format(end - start))
    print('Test accuracy:\t{}'.format(score_test[1]))
    start = time.time()
    score_whole = model.evaluate(x0, y0, verbose=0)
    end = time.time()
    print('time elapse:\t{} sec'.format(end - start))
    print('Whole accuracy:\t{}'.format(score_whole[1]))
    
    # serialize moodel weights to HDF5
    if (score_whole[1]>thres_score):
        model_file_name = "./data/model_sts_%.4f_%.4f_%.4f.h5"%(score_whole[1], score_train[1], score_test[1])
        model.save_weights(model_file_name)
        print(model_file_name, "saved to disk")
    if (score_whole[1]>the_one):
        the_one = score_whole[1]
        print('-> model update with accuracy:\t{}'.format(the_one))
        model_file_name = "./data/model_sts.h5"
        model.save_weights(model_file_name)