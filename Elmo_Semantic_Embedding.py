#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Required Libraries

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import re
from sklearn import preprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#!python -m spacy download en_core_web_md #you will need to install this on first load
import spacy
from spacy.lang.en import English
from spacy import displacy
from IPython.display import HTML
import logging
logging.getLogger('tensorflow').disabled = True #OPTIONAL - to disable outputs from Tensorflow

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import h5py

from numpy import array, hstack, vstack


# In[ ]:


#mount google drive 
from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


#reading data from the drive
data = pd.read_excel('/content/drive/My Drive/Colab Notebooks/data.xlsx')
data.head()


# In[ ]:


category = set(data['NODELABEL']);category


# In[ ]:


category_data =  data[data['NODELABEL']=='I have coupon related queries for this order'] 
category_data.head()


# In[ ]:


data.dtypes


# In[ ]:


#data1 = data.iloc[1950:2000,]
sentence = (data1['CUST_TEXT'].astype('str')).values.tolist()



#loading spacy model
import en_core_web_sm
nlp = en_core_web_sm.load()

#loading ELMo model from tensor hub
url = "https://tfhub.dev/google/elmo/2"
embed = hub.Module(url)

#creating word embeddings using ELMo model
embeddings = embed(
    sentence,
    signature="default",
    as_dict=True)["default"]


# In[ ]:


#creating session to store output for graph creation
%%time
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.tables_initializer())
  x = sess.run(embeddings)


# In[ ]:


# Writing array into hdf file
with h5py.File(('/content/drive/My Drive/Colab Notebooks/data/entire_data.h5'), 'w') as hf:
    hf.create_dataset("array",  data=data)


# In[ ]:


#Reading array from hdf file stored in drive
with h5py.File('/content/drive/My Drive/Colab Notebooks/data/array_first_half.h5', 'r') as hf:
    clust = hf['array'][:]
# len(clust)

