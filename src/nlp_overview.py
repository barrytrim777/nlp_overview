#!/usr/bin/env python
# coding: utf-8

# # nlp_overview.ipynb

# # 0. Comments

# Date: 01.12.21
# 
# Author: Barry Trim
# 
# Overview: Overview of NLP 
# 
# 
# Commments
# - 01.12.21: Initial Script
# 
# 

# # 1. Set-up

# ### 1.1 Import Statements

# In[1]:


# Importing Python Libraries to use in the code

####################################################################################################
# General modules

import pandas as pd
import numpy as np
import math
import random
import itertools

####################################################################################################


####################################################################################################
# System Interaction modules

import re
import os
import sys

####################################################################################################


####################################################################################################
# Plotting Libraries

import matplotlib.pyplot as plt
import matplotlib as mpl # new
import seaborn as sns #
sns.set(style="ticks") # Set Seaborn formatting style
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

####################################################################################################


####################################################################################################
# Datetime functions

import time
import datetime
from dateutil.parser import parse

####################################################################################################


####################################################################################################
# ML modules #######################################################################################

# Machine Learning Libraries - Sckit Learn

# Datasets
from sklearn.datasets import make_regression, fetch_20newsgroups


####################################################################################################


####################################################################################################
# NLP Modules

# nltk - NLP
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
stemmer = SnowballStemmer(language='english')

# vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vader_sentiment = SentimentIntensityAnalyzer()

# spaCy - NLP
import spacy
import en_core_web_sm
spacy_en = spacy.load('en_core_web_sm')


# SentenceTransformer - NLP
from sentence_transformers import SentenceTransformer # Embeddings
import umap # Dimenstionality Reduction
import hdbscan # Clustering

####################################################################################################



# ### 1.1 Config

# In[2]:


# Set-up project folders


# Set variables
var_sep = os.sep # Operating System directory separator
var_current_dir = os.getcwd() # find current directory
var_current_dir = var_current_dir

    
# Configure folders
var_folder_input = (var_current_dir+var_sep+"input"+var_sep)
var_folder_output = (var_current_dir+var_sep+"output"+var_sep)
var_folder_scripts = (var_current_dir+var_sep+"scripts"+var_sep)
var_folder_config = (var_current_dir+var_sep+"config"+var_sep)


# # 2. Functions

# In[ ]:





# # 3.0 Code

# ### 3.1 Import Data

# In[3]:


# import news dataset
data_pp = fetch_20newsgroups(subset='all')['data']


# In[4]:


# Number of elements in corpus
var_doc_len = len(data_pp)

# Word Count - for infor
var_text = ' '.join(data_pp)
var_text_list = var_text.split(' ')
var_word_cnt = len(var_text_list)

print('Number of documents: ', var_doc_len)
print('Number of words is:  ', var_word_cnt)
print('Avg words per doc:   ', round(var_word_cnt/var_doc_len, 2))


# In[5]:


# Filter data and unpack fields
arr_data = pd.DataFrame(data_pp)
arr_data.columns = ['text_orig']
arr_data['index_num'] = arr_data.index

# Import target data
data_target = fetch_20newsgroups(subset='all')['target']
arr_data['target'] = data_target


# In[6]:


# Add target names
data_target_names = fetch_20newsgroups(subset='all')['target_names']
arr_target_names = pd.DataFrame(data_target_names)
arr_target_names.columns = ['target_names']
arr_target_names['target'] = arr_target_names.index
arr_data = pd.merge(arr_data, arr_target_names, on=['target'])
arr_data.sort_values(by=['index_num'], inplace=True)
arr_data.reset_index(drop=True, inplace=True)


# In[7]:


# Display Data
arr_data


# In[8]:


# Display Data
print(arr_data.at[0, 'text_orig'])


# In[9]:


# Take the first 1000 records of the data
arr_data = arr_data.head(1000)


# ### 3.3 Remove Stop Characthers and email addresses

# In[ ]:


# Remove stop chars, 
# control characthers needed esaping '\'
lst_stop_chars = ['<', '>', '-', '_', '\|', ':', '\*', '\^']
arr_data['text_new'] = arr_data[['text_orig']].replace(regex=lst_stop_chars, value='')

# Remove all email addresses
arr_data['text_new'] = (arr_data[['text_new']].replace(regex='\S*@\S*\s?', value=''))

# Remove all URLs - needs to be refined
arr_data['text_new'] = (arr_data[['text_new']].replace(regex='https?://\S+', value=''))


# In[11]:


var_num = 777

print('******************* BREAK contents_replace ****************************')
print('')
print(arr_data.at[var_num, 'text_orig'])
print('')
print('******************* BREAK contents_replace_email****************************')
print('')
print(arr_data.at[var_num, 'text_new'])


# ### 3.4 POS Tagging

# In[12]:


# Get some text

var_text = """Arup is a British multinational professional services firm 
headquartered in London which provides design, engineering, architecture, 
planning, and advisory services across every aspect of the built environment. 
The firm employs approximately 16,000 staff in over 90 offices across 35 
countries around the world."""

# Get text from newsdata
# var_text = arr_data.at[30, 'text_new']

print(var_text)


# In[13]:


# Get POS and Lemmas etc
doc = spacy_en(var_text)

for token in doc:
    print(token, token.lemma_, token.pos_)
    # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)


# In[14]:


# Define POS elements to include
lst_pos = ['NOUN', 'VERB', 'DET', 'PUNCT']

# Run the filter
var_text_new = [x.lemma_ for x in doc if x.pos_ in lst_pos]

# Join the list into a string
print(' '.join(var_text_new))


# In[15]:


# Find noun phrases (noun chunks) in the text
for chunk in doc.noun_chunks:
    print(chunk.text)


# ### 3.5 Sentiment

# Vader (the NLTK Sentiment Library) works well on social media comments and reviews which are short and can contain emojis. Consider breaking up large documents into smaller documents for sentiment scoring.

# In[16]:


# Simple example of sentiment scoring example

var_text_positive = 'I love everyone and am overjoyed'
var_text_negative = 'I absolutely hate everyone and am very bitter'

print('positive: ', vader_sentiment.polarity_scores(var_text_positive))
print('positive compound: ', vader_sentiment.polarity_scores(var_text_positive)['compound'])
print()

print('negative: ', vader_sentiment.polarity_scores(var_text_negative))
print('negative compound: ', vader_sentiment.polarity_scores(var_text_negative)['compound'])


# ### 3.6 Embedding, Doc Clustering and Topic Modelling

# Text Embedding is converting ‘Text’ to numerical vectors which represent the words. Text Embedding causes models to function better as embedding models (such as BERT) can take into account content indicate similarity of meaning between two different words (e.g. ‘car’ and ‘automobile’ meaning the same thing.
# 

# In[17]:


# Text Embedding

# Define the model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# Generate the embeddings
lst_embeddings = model.encode(arr_data['text_new'], show_progress_bar=True)


# In[18]:


# View embedding
print('Embedding Length: ', len(lst_embeddings[0]))
print('')
print(lst_embeddings[0])


# In[19]:


# Dimensionality Reduction
# parameters need tuning for optimal results

# Dimensionality Reduction - for clustering (10 dimensions)
# n_components determines the number of dimensions in the output vector
umap_embeddings = umap.UMAP(n_neighbors=20, n_components=10, metric='cosine', random_state=1).fit_transform(lst_embeddings)

# Dimensionality Reduction - for plotting (2 dimensions)
umap_embeddings_plot = umap.UMAP(n_neighbors=20, n_components=2, metric='cosine', random_state=1).fit_transform(lst_embeddings)


# In[20]:


# View example of umap_embeddings
print('Embedding Length: ', len(umap_embeddings[0]))
print('')
print(umap_embeddings[0])


# In[21]:


# Cluster the documents
cluster = hdbscan.HDBSCAN(min_cluster_size=20, metric ='euclidean', cluster_selection_method='eom').fit(umap_embeddings)


# In[22]:


# Print the diffent number of clusters
np.unique(cluster.labels_)


# In[23]:


# Clustering probabilites
cluster.probabilities_


# In[ ]:


# Add cluster ID and probability to df
arr_data['cluster_id'] = list(cluster.labels_)
arr_data['cluster_prob'] = list(cluster.probabilities_)

# Add plotting values
arr_data['x'] = [x[0] for x in umap_embeddings_plot]
arr_data['y'] = [x[1] for x in umap_embeddings_plot]


# ### 3.7 Plotting

# In[25]:


# Add plotting colours to the df
lst_colors = (px.colors.qualitative.Dark24 + px.colors.qualitative.Light24 + px.colors.qualitative.Alphabet) * 15
lst_colors = ['#D2D2D2'] + lst_colors # Add light colour for Topic -1 (unmapped)
arr_colours = pd.DataFrame(lst_colors)
arr_colours['cluster_id'] = arr_colours.index - 1
arr_colours.columns = ['colour', 'cluster_id']
arr_data = pd.merge(arr_data, arr_colours, on=['cluster_id'])


# In[26]:


# Generate standalone html plot of the topic model using plotly

# Format hover text
arr_data['text_new_plot'] = arr_data['text_new'].str.wrap(100)
arr_data['text_new_plot'] = arr_data['text_new_plot'].apply(lambda x: x.replace('\n', '<br>')) # replace with hmtl line break


# Generate plot
fig_scatter = go.Figure(data=go.Scattergl(x = arr_data['x'], y = arr_data['y'],
                                text = arr_data['cluster_id'],   
                                hovertext = arr_data['text_new_plot'],
                                textposition='bottom center',
                                mode = 'markers',
                                name = '',
                                hovertemplate = "<b>Topic:</b> %{text}" + "<br><b>Text:</b> %{hovertext}",
                                marker=dict(size=10, opacity=0.5, color='rgba(0,0,0,0)', line=dict(color=arr_data['colour'], width=2)
                                            ) 
                               )
               )


fig_scatter.update_layout(autosize=True,
                    title={'text':'ScatterPlot',
                           'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                    margin=go.layout.Margin(l=20, r=20, b=20, t=100, pad=10),
                    xaxis = {'showgrid': False, 'zeroline': False, 'visible': False},
                    yaxis = {'showgrid': False, 'zeroline': False, 'visible': False},

    )


# Save the plot the figure
date_time_now = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
plot(fig_scatter, filename = var_folder_output + date_time_now + '_topic_model.html', auto_open=True)

