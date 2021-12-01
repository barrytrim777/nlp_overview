# nlp_overview.ipynb

# 0. Comments

Date: 01.12.21

Author: Barry Trim

Overview: Overview of NLP 


Commments
- 01.12.21: Initial Script



# 1. Set-up

### 1.1 Import Statements


```python
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



```

### 1.1 Config


```python
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

```

# 2. Functions


```python

```

# 3.0 Code

### 3.1 Import Data


```python
# import news dataset
data_pp = fetch_20newsgroups(subset='all')['data']

```


```python
# Number of elements in corpus
var_doc_len = len(data_pp)

# Word Count - for infor
var_text = ' '.join(data_pp)
var_text_list = var_text.split(' ')
var_word_cnt = len(var_text_list)

print('Number of documents: ', var_doc_len)
print('Number of words is:  ', var_word_cnt)
print('Avg words per doc:   ', round(var_word_cnt/var_doc_len, 2))

```

    Number of documents:  18846
    Number of words is:   5937230
    Avg words per doc:    315.04



```python
# Filter data and unpack fields
arr_data = pd.DataFrame(data_pp)
arr_data.columns = ['text_orig']
arr_data['index_num'] = arr_data.index

# Import target data
data_target = fetch_20newsgroups(subset='all')['target']
arr_data['target'] = data_target

```


```python
# Add target names
data_target_names = fetch_20newsgroups(subset='all')['target_names']
arr_target_names = pd.DataFrame(data_target_names)
arr_target_names.columns = ['target_names']
arr_target_names['target'] = arr_target_names.index
arr_data = pd.merge(arr_data, arr_target_names, on=['target'])
arr_data.sort_values(by=['index_num'], inplace=True)
arr_data.reset_index(drop=True, inplace=True)
```


```python
# Display Data
arr_data

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text_orig</th>
      <th>index_num</th>
      <th>target</th>
      <th>target_names</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>From: Mamatha Devineni Ratnam &lt;mr47+@andrew.cm...</td>
      <td>0</td>
      <td>10</td>
      <td>rec.sport.hockey</td>
    </tr>
    <tr>
      <th>1</th>
      <td>From: mblawson@midway.ecn.uoknor.edu (Matthew ...</td>
      <td>1</td>
      <td>3</td>
      <td>comp.sys.ibm.pc.hardware</td>
    </tr>
    <tr>
      <th>2</th>
      <td>From: hilmi-er@dsv.su.se (Hilmi Eren)\nSubject...</td>
      <td>2</td>
      <td>17</td>
      <td>talk.politics.mideast</td>
    </tr>
    <tr>
      <th>3</th>
      <td>From: guyd@austin.ibm.com (Guy Dawson)\nSubjec...</td>
      <td>3</td>
      <td>3</td>
      <td>comp.sys.ibm.pc.hardware</td>
    </tr>
    <tr>
      <th>4</th>
      <td>From: Alexander Samuel McDiarmid &lt;am2o+@andrew...</td>
      <td>4</td>
      <td>4</td>
      <td>comp.sys.mac.hardware</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>18841</th>
      <td>From: jim.zisfein@factory.com (Jim Zisfein) \n...</td>
      <td>18841</td>
      <td>13</td>
      <td>sci.med</td>
    </tr>
    <tr>
      <th>18842</th>
      <td>From: rdell@cbnewsf.cb.att.com (richard.b.dell...</td>
      <td>18842</td>
      <td>12</td>
      <td>sci.electronics</td>
    </tr>
    <tr>
      <th>18843</th>
      <td>From: westes@netcom.com (Will Estes)\nSubject:...</td>
      <td>18843</td>
      <td>3</td>
      <td>comp.sys.ibm.pc.hardware</td>
    </tr>
    <tr>
      <th>18844</th>
      <td>From: steve@hcrlgw (Steven Collins)\nSubject: ...</td>
      <td>18844</td>
      <td>1</td>
      <td>comp.graphics</td>
    </tr>
    <tr>
      <th>18845</th>
      <td>From: chriss@netcom.com (Chris Silvester)\nSub...</td>
      <td>18845</td>
      <td>7</td>
      <td>rec.autos</td>
    </tr>
  </tbody>
</table>
<p>18846 rows × 4 columns</p>
</div>




```python
# Display Data
print(arr_data.at[0, 'text_orig'])
```

    From: Mamatha Devineni Ratnam <mr47+@andrew.cmu.edu>
    Subject: Pens fans reactions
    Organization: Post Office, Carnegie Mellon, Pittsburgh, PA
    Lines: 12
    NNTP-Posting-Host: po4.andrew.cmu.edu
    
    
    
    I am sure some bashers of Pens fans are pretty confused about the lack
    of any kind of posts about the recent Pens massacre of the Devils. Actually,
    I am  bit puzzled too and a bit relieved. However, I am going to put an end
    to non-PIttsburghers' relief with a bit of praise for the Pens. Man, they
    are killing those Devils worse than I thought. Jagr just showed you why
    he is much better than his regular season stats. He is also a lot
    fo fun to watch in the playoffs. Bowman should let JAgr have a lot of
    fun in the next couple of games since the Pens are going to beat the pulp out of Jersey anyway. I was very disappointed not to see the Islanders lose the final
    regular season game.          PENS RULE!!!
    
    



```python
# Take the first 1000 records of the data
arr_data = arr_data.head(1000)

```

### 3.3 Remove Stop Characthers and email addresses


```python
# Remove stop chars, 
# control characthers needed esaping '\'
lst_stop_chars = ['<', '>', '-', '_', '\|', ':', '\*', '\^']
arr_data['text_new'] = arr_data[['text_orig']].replace(regex=lst_stop_chars, value='')

# Remove all email addresses
arr_data['text_new'] = (arr_data[['text_new']].replace(regex='\S*@\S*\s?', value=''))

# Remove all URLs - needs to be refined
arr_data['text_new'] = (arr_data[['text_new']].replace(regex='https?://\S+', value=''))

```


```python
var_num = 777

print('******************* BREAK contents_replace ****************************')
print('')
print(arr_data.at[var_num, 'text_orig'])
print('')
print('******************* BREAK contents_replace_email****************************')
print('')
print(arr_data.at[var_num, 'text_new'])

```

    ******************* BREAK contents_replace ****************************
    
    From: gregof@JSP.UMontreal.CA (Grego Filippo)
    Subject: Info wanted on Tseng Labs ET4000 VLB
    Organization: Universite de Montreal
    Lines: 9
    
    Hi fellow netters,
    
    does anybody have any info on Tseng Labs ET4000 VLB card:
    price, speed, compatibility with existing and up-comming softwares,
    performance compared to others cards ( is it an S3 based card ?)....
    
    Thank you..
    
    
    
    
    ******************* BREAK contents_replace_email****************************
    
    From (Grego Filippo)
    Subject Info wanted on Tseng Labs ET4000 VLB
    Organization Universite de Montreal
    Lines 9
    
    Hi fellow netters,
    
    does anybody have any info on Tseng Labs ET4000 VLB card
    price, speed, compatibility with existing and upcomming softwares,
    performance compared to others cards ( is it an S3 based card ?)....
    
    Thank you..
    
    
    


### 3.4 POS Tagging


```python
# Get some text

var_text = """Arup is a British multinational professional services firm 
headquartered in London which provides design, engineering, architecture, 
planning, and advisory services across every aspect of the built environment. 
The firm employs approximately 16,000 staff in over 90 offices across 35 
countries around the world."""

# Get text from newsdata
# var_text = arr_data.at[30, 'text_new']

print(var_text)
```

    Arup is a British multinational professional services firm 
    headquartered in London which provides design, engineering, architecture, 
    planning, and advisory services across every aspect of the built environment. 
    The firm employs approximately 16,000 staff in over 90 offices across 35 
    countries around the world.



```python
# Get POS and Lemmas etc
doc = spacy_en(var_text)

for token in doc:
    print(token, token.lemma_, token.pos_)
    # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
```

    Arup arup NOUN
    is be AUX
    a a DET
    British british ADJ
    multinational multinational ADJ
    professional professional ADJ
    services service NOUN
    firm firm NOUN
    
     
     SPACE
    headquartered headquarter VERB
    in in ADP
    London London PROPN
    which which PRON
    provides provide VERB
    design design NOUN
    , , PUNCT
    engineering engineering NOUN
    , , PUNCT
    architecture architecture NOUN
    , , PUNCT
    
     
     SPACE
    planning planning NOUN
    , , PUNCT
    and and CCONJ
    advisory advisory ADJ
    services service NOUN
    across across ADP
    every every DET
    aspect aspect NOUN
    of of ADP
    the the DET
    built build VERB
    environment environment NOUN
    . . PUNCT
    
     
     SPACE
    The the DET
    firm firm NOUN
    employs employ VERB
    approximately approximately ADV
    16,000 16,000 NUM
    staff staff NOUN
    in in ADP
    over over ADP
    90 90 NUM
    offices office NOUN
    across across ADP
    35 35 NUM
    
     
     SPACE
    countries country NOUN
    around around ADP
    the the DET
    world world NOUN
    . . PUNCT



```python
# Define POS elements to include
lst_pos = ['NOUN', 'VERB', 'DET', 'PUNCT']

# Run the filter
var_text_new = [x.lemma_ for x in doc if x.pos_ in lst_pos]

# Join the list into a string
print(' '.join(var_text_new))
```

    arup a service firm headquarter provide design , engineering , architecture , planning , service every aspect the build environment . the firm employ staff office country the world .



```python
# Find noun phrases (noun chunks) in the text
for chunk in doc.noun_chunks:
    print(chunk.text)
```

    Arup
    a British multinational professional services firm
    London
    which
    design
    engineering
    architecture
    planning
    advisory services
    every aspect
    the built environment
    The firm
    approximately 16,000 staff
    over 90 offices
    35 
    countries
    the world


### 3.5 Sentiment

Vader (the NLTK Sentiment Library) works well on social media comments and reviews which are short and can contain emojis. Consider breaking up large documents into smaller documents for sentiment scoring.


```python
# Simple example of sentiment scoring example

var_text_positive = 'I love everyone and am overjoyed'
var_text_negative = 'I absolutely hate everyone and am very bitter'

print('positive: ', vader_sentiment.polarity_scores(var_text_positive))
print('positive compound: ', vader_sentiment.polarity_scores(var_text_positive)['compound'])
print()

print('negative: ', vader_sentiment.polarity_scores(var_text_negative))
print('negative compound: ', vader_sentiment.polarity_scores(var_text_negative)['compound'])


```

    positive:  {'neg': 0.0, 'neu': 0.336, 'pos': 0.664, 'compound': 0.836}
    positive compound:  0.836
    
    negative:  {'neg': 0.541, 'neu': 0.459, 'pos': 0.0, 'compound': -0.7956}
    negative compound:  -0.7956


### 3.6 Embedding, Doc Clustering and Topic Modelling

Text Embedding is converting ‘Text’ to numerical vectors which represent the words. Text Embedding causes models to function better as embedding models (such as BERT) can take into account content indicate similarity of meaning between two different words (e.g. ‘car’ and ‘automobile’ meaning the same thing.



```python
# Text Embedding

# Define the model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# Generate the embeddings
lst_embeddings = model.encode(arr_data['text_new'], show_progress_bar=True)
```


    HBox(children=(HTML(value='Batches'), FloatProgress(value=0.0, max=32.0), HTML(value='')))


    



```python
# View embedding
print('Embedding Length: ', len(lst_embeddings[0]))
print('')
print(lst_embeddings[0])

```

    Embedding Length:  768
    
    [ 3.11651736e-01 -4.15521234e-01  4.59602594e-01 -1.21574461e-01
     -7.82123744e-01 -1.30180627e-01 -1.68112531e-01 -6.15227759e-01
      7.42287636e-01  3.71210963e-01  2.18754858e-01  5.41139901e-01
     -5.09341657e-01  1.11398935e+00  4.44403321e-01 -5.20721257e-01
      4.19572711e-01 -1.01445481e-01 -2.66003549e-01  2.55932450e-01
     -7.73899972e-01  4.10386741e-01  5.92998147e-01  9.32648063e-01
     -9.18336630e-01 -5.03186956e-02  4.65151757e-01 -4.28315967e-01
      4.63159472e-01  5.07030964e-01  5.78824937e-01  4.04971987e-01
     -5.70110261e-01 -1.81191087e-01  3.62696826e-01 -1.44147411e-01
      5.43858767e-01  2.34094337e-01  1.07747352e+00 -3.51341546e-01
     -2.24507377e-01 -8.55968595e-01 -1.86308119e-02  7.46259391e-01
      3.64485919e-01 -6.22368336e-01  5.58727562e-01 -1.28041700e-01
     -6.42184615e-01  8.83252770e-02  3.57173562e-01  2.69428760e-01
      2.05855578e-01  4.11779433e-02  1.55159205e-01 -4.73711312e-01
     -5.31442165e-02 -5.13362169e-01  6.85164750e-01  4.30066913e-01
      4.50583935e-01 -1.85458824e-01 -6.39908969e-01 -1.00309968e+00
     -4.62398455e-02 -3.69612873e-01 -6.94503844e-01  1.33078456e+00
     -2.99303353e-01 -3.85404646e-01  3.68702561e-01 -8.80081654e-02
      9.46023911e-02  2.47561440e-01  8.39081049e-01 -1.00489759e+00
      4.22721744e-01 -3.75251681e-01 -5.75513005e-01 -3.35222512e-01
     -1.55082747e-01  5.33235490e-01  2.60431588e-01 -1.95595533e-01
      4.71138991e-02  5.76438308e-01  1.08260465e+00 -1.69039443e-01
      4.51420128e-01  4.53448206e-01 -5.41999698e-01 -3.86571959e-02
      6.08852386e-01 -8.49256337e-01  7.75225088e-02  3.07606429e-01
      5.47214866e-01 -6.30895853e-01  1.18453950e-01 -2.45245248e-01
     -2.56473362e-01  7.10554898e-01  4.46303517e-01  7.29206145e-01
      2.35104755e-01 -8.65973234e-01 -1.32430196e-01  1.77506611e-01
     -1.64803267e-01 -7.62004256e-02  5.93815804e-01  1.34699667e+00
      4.30720389e-01 -5.32624006e-01 -3.43696207e-01 -1.66960919e+00
      3.31132829e-01 -4.77033943e-01  6.28794968e-01  6.44660532e-01
      5.44351041e-01 -2.11217090e-01  7.95624435e-01 -5.07535756e-01
     -1.72874808e-01 -1.22764759e-01 -4.23299313e-01  1.76151201e-01
     -1.15262344e-03  4.69367266e-01 -3.73718068e-02  2.30161637e-01
      2.87378699e-01 -9.23996940e-02  8.54950678e-03 -7.45458066e-01
     -7.32571363e-01 -2.47866303e-01 -4.72078860e-01 -1.39972031e-01
     -1.35436952e+00 -3.40091914e-01  9.07955885e-01  1.35486111e-01
     -2.01192200e-01 -4.85055000e-01 -1.11232126e+00 -5.26532531e-05
      2.06812501e-01 -3.50890785e-01 -8.84102821e-01  4.92045879e-02
      4.94755447e-01 -2.59245634e-01  5.14497221e-01  3.04004192e-01
      1.81112897e+00 -4.24471438e-01  7.24247754e-01 -9.79404598e-02
     -8.13962296e-02 -2.85158604e-01 -4.40126300e-01  3.74583244e-01
      1.10895348e+00 -6.37093961e-01  1.64499819e-01 -8.29807758e-01
     -6.67364299e-01  5.99320009e-02 -1.08364746e-01 -3.30818057e-01
      7.90323794e-01 -3.11763257e-01  1.45349240e+00 -2.11464420e-01
      3.99957970e-02  2.37928122e-01 -2.81392664e-01 -2.91522920e-01
      1.27109140e-01  2.17169542e-02 -3.09063613e-01  5.32574594e-01
     -3.89464855e-01 -6.23474598e-01  3.91959637e-01 -2.08190158e-01
      7.94455528e-01  8.31457436e-01  2.60437042e-01 -6.23986274e-02
     -6.57352507e-01  3.84552777e-01  8.67438674e-01  2.06114411e-01
      2.41218433e-01  9.77489129e-02 -3.20184082e-02  4.22901988e-01
     -8.79476905e-01 -6.97677732e-01 -8.73999298e-02  4.71671015e-01
     -5.21948218e-01 -2.06212878e-01  5.54463506e-01  6.65569961e-01
      5.62519789e-01 -5.43430746e-02  3.90323639e-01  6.84948862e-01
      6.12166002e-02 -4.24716175e-01 -1.38916865e-01 -7.84073293e-01
     -5.28921708e-02 -8.36153269e-01 -3.79708678e-01 -1.23509079e-01
      2.78755277e-01 -5.89713812e-01 -2.63947099e-01  4.18084532e-01
      1.20499527e+00  1.25423610e-01 -7.02963591e-01 -4.08887386e-01
     -1.61859300e-02 -4.68471736e-01  2.44523026e-03  5.25971018e-02
     -3.22683781e-01  7.50237405e-02 -1.10036775e-01 -1.93552837e-01
     -7.86377966e-01 -1.58863842e+00  9.37331736e-01 -8.98534894e-01
     -2.02004179e-01  1.95726171e-01  6.20328546e-01  7.11373448e-01
      6.99355364e-01  5.56580722e-01  7.85519034e-02 -4.98936266e-01
     -4.52289343e-01  4.46341068e-01 -8.41682673e-01  1.09913394e-01
     -9.60817933e-01 -8.78694534e-01  8.63803476e-02 -4.39817548e-01
      7.02057958e-01 -6.19709015e-01 -9.96111147e-03  2.20180795e-01
     -2.48484071e-02 -9.80575860e-01  6.08431220e-01 -1.08955897e-01
     -8.48861635e-01  5.62354811e-02 -8.23802471e-01 -5.90499997e-01
     -9.66210291e-02  3.68068755e-01 -1.02362655e-01 -7.15084851e-01
      2.21275046e-01  8.78970325e-02 -3.24264735e-01 -7.25689769e-01
      4.00464356e-01 -6.43147230e-01  3.54217798e-01  3.93295705e-01
     -2.42628098e-01  2.38374144e-01  3.65805715e-01 -7.93296337e-01
     -1.94896102e-01 -2.81814158e-01  1.25117445e+00 -5.08349478e-01
      1.11519853e-02 -9.41341877e-01  2.20144317e-01  1.44737586e-01
     -1.25244856e-02  1.13914847e+00  1.08865447e-01 -2.60557950e-01
      2.79060453e-01 -1.37410676e-02  3.75708103e-01  5.77616915e-02
     -6.45174742e-01  2.00331002e-01  7.15709329e-02  5.99921703e-01
     -3.80234513e-03  7.11846799e-02  7.97372162e-01 -8.77719522e-01
      3.16973291e-02 -7.24983335e-01  3.16573739e-01 -3.12468886e-01
      6.74143851e-01  7.59640455e-01  7.94548988e-01  2.78206527e-01
     -1.22600496e-01 -3.12583148e-01 -5.13063192e-01 -5.08404016e-01
      5.75289249e-01 -4.06834602e-01  8.14080596e-01  5.82935035e-01
      6.01389781e-02  9.97469783e-01 -3.35824430e-01  8.79049659e-01
     -2.51872957e-01  7.16491416e-03  3.13188940e-01 -2.23345801e-01
      2.45483547e-01  3.36491495e-01  1.83014408e-01  1.62187368e-01
      2.85138726e-01  4.81545508e-01 -9.00731832e-02  6.32277071e-01
      6.80203736e-01  8.66331309e-02 -6.09346151e-01 -2.20020667e-01
     -4.36110705e-01  6.98814869e-01 -1.04596472e+00  1.45546839e-01
      2.95717418e-01  2.91427970e-01 -3.88154209e-01  5.73667228e-01
      6.99787378e-01 -3.77584547e-01 -6.29637897e-01  2.24031895e-01
     -4.30278599e-01  4.79380995e-01  1.22864866e+00 -3.18867683e-01
     -6.44435361e-02  4.31855798e-01  8.38174462e-01 -1.75463527e-01
     -2.90182501e-01 -1.23636536e-01 -1.46743035e+00 -4.63689387e-01
     -7.60436505e-02  7.61196136e-01 -1.77588761e-01  7.98241019e-01
     -4.80729610e-01 -1.08737335e-01  5.10360003e-01  8.80518109e-02
      4.92901981e-01  1.31866753e-01 -8.98481667e-01 -3.82592261e-01
      4.12557200e-02  1.41015165e-02 -2.96780705e-01  4.77032334e-01
     -5.18299222e-01 -6.38897300e-01 -8.53641629e-01  1.49714917e-01
     -3.49935085e-01  3.05372536e-01  1.33573607e-01 -1.92558151e-02
     -3.81281488e-02 -5.45620136e-02 -8.29473555e-01 -7.46539712e-01
     -5.63867688e-01  2.19964206e-01  1.33425891e-01 -7.14244843e-01
     -3.64056736e-01  9.79070783e-01  2.33054131e-01  1.46192694e+00
      5.17964065e-01 -4.41580206e-01  2.25105181e-01  5.85252583e-01
      7.47360528e-01 -8.30267787e-01 -1.68476813e-02 -2.28284895e-01
     -3.69999021e-01  3.00301701e-01  5.68127930e-01 -3.63269776e-01
     -1.34183073e+00 -4.32651460e-01 -9.02406633e-01 -5.30575871e-01
      5.91963589e-01  7.81108364e-02 -4.79659485e-03  4.04638976e-01
     -4.46796268e-01 -1.47181526e-01 -5.02037182e-02 -6.50471747e-01
     -6.18146062e-01  1.77728176e-01 -5.56358874e-01 -2.34010741e-01
      1.32300246e+00  2.58930117e-01 -7.81419516e-01  3.04489344e-01
      4.36591685e-01  2.06155404e-01  2.97191679e-01 -1.85026526e-01
     -3.57103646e-01  2.09728569e-01 -2.29540080e-01  4.87949818e-01
     -8.28458786e-01 -4.52105671e-01  4.07075942e-01 -3.45836639e-01
     -2.69781560e-01  3.54931384e-01 -3.05045485e-01 -6.82934970e-02
      3.88019115e-01  6.82896197e-01  8.41041863e-01 -3.81105691e-01
     -9.58249792e-02 -2.77579486e-01 -9.66868401e-02  1.96925864e-01
     -5.65374255e-01 -1.58296704e-01  5.03619909e-01  4.39118147e-01
      1.45239204e-01  9.02416170e-01  3.56754810e-01  9.46067274e-02
      2.38603413e-01 -1.15153515e+00 -9.19734955e-01  2.92576134e-01
      7.40949512e-01 -3.99046838e-02 -5.45569420e-01 -9.09802854e-01
      5.35654008e-01 -1.67269319e-01 -7.98973888e-02 -6.61265194e-01
      6.42117321e-01 -6.47528529e-01 -2.67920256e-01 -3.10875505e-01
      1.70729190e-01  2.94792116e-01 -6.98643208e-01  4.11545217e-01
     -9.32988465e-01 -6.53961152e-02 -4.06087637e-02  9.75798249e-01
     -1.56012595e-01  1.09020853e+00  7.26734936e-01  7.27764606e-01
     -7.38196492e-01  1.04447806e+00  1.34140298e-01 -4.53314960e-01
     -1.91054001e-01  3.31987709e-01  1.67182431e-01  5.22994325e-02
      1.34425148e-01 -6.70732737e-01 -5.63641638e-02  2.87036091e-01
      6.99240088e-01  9.40100074e-01 -5.64689375e-02 -3.01179886e-01
     -7.35860288e-01 -8.10188413e-01  3.55702460e-01  5.37746012e-01
     -3.39046955e-01  3.39274377e-01 -3.96772087e-01  4.55441862e-01
     -6.10379100e-01 -5.14163673e-01  8.02443564e-01 -5.43212116e-01
      4.74471241e-01  2.90717065e-01 -1.18287945e+00 -1.36041880e+00
     -1.60563603e-01  4.74504590e-01 -1.94779575e-01 -4.77612704e-01
      1.66230083e-01  1.27218282e+00 -7.83967137e-01  5.59186280e-01
     -1.90726295e-01 -5.38330317e-01  1.18475175e+00  6.90512508e-02
      2.37355143e-01  4.58604544e-01  6.98619127e-01 -8.52260441e-02
     -2.66287595e-01  4.91694435e-02 -3.24853629e-01 -5.82822800e-01
     -3.45359415e-01  4.13164526e-01  4.26057667e-01 -1.77581608e-01
      6.62115455e-01 -7.74923444e-01  6.91561878e-01 -2.44732082e-01
      1.79688826e-01  2.20674261e-01  4.55730975e-01  2.76844919e-01
     -5.29150248e-01 -2.16634884e-01  3.88040543e-01 -7.08286285e-01
      2.19358698e-01 -4.72052813e-01 -5.56204915e-01  2.67678127e-02
     -1.09302437e+00  3.61109018e-01  4.43266481e-02  5.47809005e-01
     -6.96592152e-01 -1.53724062e+00  2.47413903e-01 -3.17731470e-01
     -1.26416171e+00 -2.24450380e-01 -2.83317447e-01  6.43212676e-01
     -9.01712626e-02  3.78751159e-01 -3.14119756e-01 -3.05347681e-01
     -1.75387979e-01 -9.53902006e-02  6.05944209e-02 -2.89864875e-02
     -1.28440276e-01 -2.64110297e-01 -5.46473980e-01 -7.17729628e-01
      4.23347503e-01 -2.29610428e-01  7.98629642e-01 -3.04720014e-01
     -3.12822722e-02  7.53043294e-01  5.69033146e-01 -5.79780757e-01
      2.78818429e-01 -7.12325156e-01  7.07295954e-01  1.64810196e-01
      5.29469967e-01 -2.25001015e-02  5.89295387e-01 -3.15621167e-01
     -2.23849937e-01  3.42291534e-01  6.20197654e-01 -2.25577638e-01
      1.09127194e-01  1.27399653e-01 -1.04244852e+00 -4.25330810e-02
      2.28368372e-01 -2.82427996e-01 -4.86113667e-01 -3.72290820e-01
      1.84391439e-01  6.03126109e-01 -7.64910281e-01  6.15824163e-01
     -3.91209684e-02  5.52764416e-01 -7.77239263e-01  6.36030555e-01
      5.41082501e-01 -2.88283616e-01  1.68443456e-01 -2.29198501e-01
      3.02547514e-01 -8.23701084e-01 -1.15562630e+00  8.57779205e-01
     -8.26887131e-01  1.37180552e-01  1.67508051e-01 -1.10833213e-01
     -6.28509045e-01 -5.47035456e-01 -4.15061414e-02  8.92039299e-01
      1.07085943e+00 -5.80086946e-01 -5.40398002e-01  3.08304012e-01
     -2.69565821e-01  6.21446550e-01 -6.61990821e-01  9.99443412e-01
     -5.11192083e-02  5.33920884e-01  3.06737065e-01  1.01705813e+00
      3.80663201e-02 -6.33274615e-01 -2.13012323e-01 -7.01264620e-01
     -5.70829988e-01 -5.60387075e-01 -3.12974334e-01  4.17373925e-02
      5.17509103e-01  4.54566091e-01  1.43117696e-01 -3.74367774e-01
     -3.87715846e-01 -4.54097092e-01 -3.29247773e-01  1.05615604e+00
     -1.54652321e+00 -1.88887715e-01  3.37956071e-01  5.19114375e-01
     -7.68878579e-01  6.26906008e-03  3.05545181e-01  3.71665433e-02
     -1.35920596e+00  2.45626017e-01 -6.88377976e-01  4.65401337e-02
     -2.82167912e-01 -3.63258541e-01  5.64713590e-02  1.01691246e+00
      4.36786920e-01 -5.58752537e-01  4.18222904e-01  1.71297505e-01
     -3.08321834e-01 -4.57338125e-01  1.10875927e-02 -2.78492987e-01
      3.71936597e-02 -7.68101931e-01 -1.25169683e+00  8.05762261e-02
     -2.07736000e-01 -1.18499964e-01 -3.27433467e-01 -2.05663532e-01
     -1.11825936e-01  2.28551775e-01  1.28253028e-02  6.94504678e-01
     -3.60441089e-01 -9.92668867e-01  8.06335986e-01 -7.83893347e-01
     -3.22375894e-01 -4.59250510e-01  2.89244682e-01 -1.09507469e-02
     -1.14834464e+00  3.18470895e-01 -5.88264577e-02  4.19766456e-01
     -9.15395856e-01 -1.69079989e-01 -6.02216423e-01 -3.49897742e-01
     -8.57860029e-01  8.40671137e-02 -2.68011987e-01  7.55660832e-02
     -2.47089818e-01 -4.69282806e-01 -5.26433587e-01  3.48675326e-02
      4.22718853e-01 -9.15037543e-02 -1.11383781e-01 -3.61893214e-02
     -4.93998379e-01  5.88897645e-01  1.71037674e-01 -6.92062736e-01
     -1.60189986e-01 -1.14765968e-02 -4.88495171e-01 -1.76822424e+00
      1.61109433e-01  4.82677162e-01  2.38364115e-01  3.70894045e-01
     -4.28817347e-02 -1.16501875e-01 -1.35070920e-01  7.91680932e-01
      3.11765850e-01 -1.72707945e-01  8.20030347e-02 -4.75795031e-01
     -4.67029303e-01 -6.53111756e-01 -6.90162659e-01 -1.67404324e-01
     -2.46469021e-01  8.85856748e-01  6.61219656e-01 -6.32712170e-02
     -1.52404547e-01  6.02636412e-02  2.81475961e-01  1.17374331e-01]



```python
# Dimensionality Reduction
# parameters need tuning for optimal results

# Dimensionality Reduction - for clustering (10 dimensions)
# n_components determines the number of dimensions in the output vector
umap_embeddings = umap.UMAP(n_neighbors=20, n_components=10, metric='cosine', random_state=1).fit_transform(lst_embeddings)

# Dimensionality Reduction - for plotting (2 dimensions)
umap_embeddings_plot = umap.UMAP(n_neighbors=20, n_components=2, metric='cosine', random_state=1).fit_transform(lst_embeddings)

```


```python
# View example of umap_embeddings
print('Embedding Length: ', len(umap_embeddings[0]))
print('')
print(umap_embeddings[0])

```

    Embedding Length:  10
    
    [4.747404  6.2907195 8.761942  7.7304115 4.0579996 6.548015  5.444745
     4.3544    4.0735064 5.865413 ]



```python
# Cluster the documents
cluster = hdbscan.HDBSCAN(min_cluster_size=20, metric ='euclidean', cluster_selection_method='eom').fit(umap_embeddings)

```


```python
# Print the diffent number of clusters
np.unique(cluster.labels_)

```




    array([-1,  0,  1,  2])




```python
# Clustering probabilites
cluster.probabilities_

```




    array([0.8356093 , 1.        , 1.        , 1.        , 1.        ,
           1.        , 1.        , 0.998292  , 1.        , 1.        ,
           0.95309127, 0.7969983 , 1.        , 1.        , 0.76346394,
           0.93221751, 1.        , 0.        , 0.        , 0.        ,
           1.        , 0.94990749, 0.98331318, 0.        , 1.        ,
           1.        , 0.66825979, 0.        , 1.        , 1.        ,
           0.        , 0.985194  , 1.        , 0.        , 0.        ,
           1.        , 1.        , 0.84675549, 1.        , 0.908934  ,
           0.88097526, 1.        , 0.97387638, 0.        , 1.        ,
           1.        , 1.        , 0.        , 0.79040183, 0.93804263,
           1.        , 1.        , 0.        , 1.        , 0.        ,
           0.99802888, 0.        , 1.        , 0.96978327, 1.        ,
           0.88991694, 1.        , 0.9906081 , 0.91149776, 0.86532684,
           0.        , 0.97422937, 0.99976243, 1.        , 1.        ,
           1.        , 0.80980246, 1.        , 1.        , 1.        ,
           1.        , 0.        , 1.        , 0.        , 0.74168537,
           0.        , 1.        , 1.        , 0.        , 1.        ,
           1.        , 1.        , 1.        , 0.8867628 , 0.88954845,
           0.72702876, 1.        , 0.83728708, 1.        , 0.82706732,
           1.        , 1.        , 1.        , 0.8227774 , 1.        ,
           0.94414256, 1.        , 1.        , 0.98537544, 1.        ,
           0.66409427, 0.985653  , 0.95454273, 0.59712658, 1.        ,
           0.99645785, 1.        , 1.        , 1.        , 0.82306218,
           1.        , 1.        , 1.        , 0.88991694, 0.96671883,
           1.        , 0.92661421, 1.        , 0.        , 0.75342305,
           0.97253732, 1.        , 0.9277466 , 1.        , 1.        ,
           0.76878533, 1.        , 0.77087995, 0.84739   , 1.        ,
           0.91040309, 1.        , 0.        , 1.        , 0.95484997,
           1.        , 1.        , 0.        , 0.        , 0.        ,
           0.78513101, 1.        , 1.        , 0.63774011, 1.        ,
           0.96735565, 0.74952238, 1.        , 1.        , 0.83738139,
           1.        , 0.        , 1.        , 1.        , 1.        ,
           1.        , 1.        , 0.72012517, 1.        , 1.        ,
           0.        , 1.        , 0.94912361, 0.95608592, 0.86384198,
           1.        , 0.81922556, 1.        , 1.        , 1.        ,
           1.        , 0.85538505, 1.        , 0.        , 1.        ,
           0.        , 0.78065259, 0.78873697, 0.        , 1.        ,
           1.        , 1.        , 1.        , 1.        , 1.        ,
           1.        , 1.        , 0.97574934, 0.97116116, 1.        ,
           0.85870366, 0.90949348, 1.        , 0.97399918, 1.        ,
           1.        , 0.88629339, 0.        , 0.87451002, 1.        ,
           1.        , 1.        , 0.93953353, 1.        , 0.80350367,
           1.        , 0.88395549, 0.        , 1.        , 0.        ,
           0.        , 1.        , 0.        , 0.97049454, 0.        ,
           0.88120506, 1.        , 0.91304373, 1.        , 0.84798791,
           0.        , 1.        , 0.91147284, 1.        , 0.88991694,
           0.        , 1.        , 0.        , 1.        , 0.80187477,
           1.        , 1.        , 1.        , 1.        , 0.86489476,
           0.91146317, 0.        , 0.        , 1.        , 1.        ,
           1.        , 1.        , 1.        , 0.        , 0.87675853,
           1.        , 0.90238277, 1.        , 1.        , 0.        ,
           1.        , 1.        , 1.        , 1.        , 1.        ,
           1.        , 0.        , 0.9623689 , 0.99199667, 1.        ,
           1.        , 1.        , 1.        , 1.        , 1.        ,
           0.        , 0.        , 1.        , 1.        , 0.        ,
           0.92490413, 0.86602064, 0.9492907 , 0.84112326, 0.92146324,
           0.        , 0.77976692, 1.        , 0.88991694, 1.        ,
           1.        , 0.9367941 , 0.81643112, 1.        , 1.        ,
           1.        , 0.        , 1.        , 1.        , 1.        ,
           1.        , 1.        , 0.93881707, 1.        , 0.        ,
           1.        , 1.        , 1.        , 0.        , 0.99675723,
           1.        , 0.87752828, 0.86269508, 1.        , 0.79983624,
           1.        , 0.        , 0.83022725, 0.        , 1.        ,
           1.        , 1.        , 0.89244951, 1.        , 0.94553542,
           0.84055173, 1.        , 0.        , 1.        , 0.        ,
           0.        , 1.        , 0.        , 0.95903247, 1.        ,
           0.        , 0.85450028, 1.        , 0.        , 1.        ,
           1.        , 0.84094441, 0.96905938, 0.98923477, 1.        ,
           1.        , 0.58604525, 0.95254077, 1.        , 0.7830469 ,
           1.        , 0.908934  , 1.        , 0.        , 0.        ,
           1.        , 0.58120003, 0.        , 1.        , 1.        ,
           0.85512398, 0.772667  , 1.        , 0.        , 1.        ,
           1.        , 0.88748736, 1.        , 0.83054567, 0.91105997,
           0.73261469, 1.        , 0.96182151, 1.        , 1.        ,
           1.        , 1.        , 0.        , 0.79175295, 0.        ,
           1.        , 0.        , 0.        , 1.        , 1.        ,
           0.        , 0.88036399, 0.90323   , 0.        , 0.90263989,
           1.        , 0.63512289, 1.        , 0.        , 1.        ,
           1.        , 1.        , 0.        , 0.80045848, 1.        ,
           0.81129271, 1.        , 1.        , 1.        , 1.        ,
           0.99456662, 0.88569164, 0.        , 0.        , 0.99976243,
           0.93505686, 0.87839063, 0.90076488, 0.        , 0.93832051,
           0.88991694, 1.        , 0.91308911, 0.        , 0.        ,
           0.        , 1.        , 0.79381695, 1.        , 1.        ,
           1.        , 1.        , 0.94829464, 1.        , 0.81370532,
           1.        , 0.97734003, 1.        , 0.        , 0.        ,
           1.        , 1.        , 0.66685418, 0.        , 1.        ,
           1.        , 1.        , 1.        , 1.        , 0.        ,
           1.        , 1.        , 0.        , 0.91670345, 1.        ,
           0.91178213, 0.        , 1.        , 0.        , 1.        ,
           1.        , 0.87366261, 0.79477862, 1.        , 0.90045978,
           1.        , 1.        , 1.        , 0.98837428, 0.        ,
           1.        , 1.        , 1.        , 1.        , 1.        ,
           1.        , 1.        , 0.96197625, 0.82258506, 1.        ,
           1.        , 0.77752077, 1.        , 1.        , 0.        ,
           1.        , 1.        , 0.8432469 , 1.        , 0.98840638,
           0.87839063, 1.        , 0.78870671, 1.        , 1.        ,
           0.        , 0.        , 0.90285471, 1.        , 0.87451279,
           0.        , 1.        , 0.95483791, 1.        , 0.        ,
           1.        , 0.        , 0.98420571, 0.86432144, 1.        ,
           0.        , 1.        , 1.        , 1.        , 0.87803933,
           1.        , 0.94374693, 0.        , 0.        , 0.        ,
           1.        , 0.85062003, 1.        , 0.        , 1.        ,
           0.95817839, 1.        , 1.        , 1.        , 1.        ,
           0.        , 0.        , 1.        , 0.98717344, 1.        ,
           0.85991524, 0.96732923, 0.87083332, 1.        , 0.88991694,
           0.        , 0.        , 0.95939956, 0.93073875, 1.        ,
           0.88991694, 1.        , 0.83634358, 0.95864798, 1.        ,
           0.98768817, 0.87496285, 0.95542552, 1.        , 0.        ,
           1.        , 1.        , 1.        , 1.        , 0.96334568,
           1.        , 1.        , 1.        , 0.92764605, 0.85290001,
           0.90141211, 0.85844156, 1.        , 1.        , 0.82662511,
           0.        , 0.84033576, 1.        , 0.75833207, 1.        ,
           0.8822133 , 1.        , 0.        , 1.        , 1.        ,
           0.89619319, 1.        , 1.        , 0.93454865, 0.86723317,
           0.74916728, 0.88991694, 1.        , 0.84377168, 1.        ,
           0.        , 1.        , 1.        , 0.92496312, 0.78730063,
           1.        , 0.94493213, 1.        , 0.89566214, 1.        ,
           0.77741812, 0.92162031, 0.92379744, 0.96136059, 0.65894487,
           0.80955774, 0.91091428, 1.        , 1.        , 0.        ,
           1.        , 0.94892669, 1.        , 0.        , 0.99789389,
           0.91990711, 0.94820137, 1.        , 1.        , 1.        ,
           1.        , 0.        , 1.        , 1.        , 1.        ,
           1.        , 0.        , 0.86910236, 1.        , 1.        ,
           1.        , 1.        , 1.        , 0.8571577 , 1.        ,
           0.89499399, 0.89487876, 1.        , 1.        , 0.        ,
           0.7184005 , 1.        , 0.        , 0.        , 0.86978078,
           1.        , 0.99699542, 0.78704971, 0.89880273, 0.85986329,
           1.        , 0.        , 0.86200585, 0.        , 0.        ,
           0.92162031, 0.        , 1.        , 1.        , 0.93126799,
           1.        , 1.        , 1.        , 1.        , 0.76896419,
           0.89453882, 0.98770906, 1.        , 1.        , 1.        ,
           0.93757772, 1.        , 0.        , 0.        , 1.        ,
           0.94169673, 0.88991694, 0.90686385, 1.        , 1.        ,
           0.99183535, 1.        , 1.        , 0.        , 0.        ,
           0.95505726, 1.        , 0.99378399, 0.7732116 , 0.        ,
           0.        , 0.        , 0.        , 1.        , 0.99126102,
           0.88118561, 0.88991694, 0.89449668, 0.        , 1.        ,
           1.        , 0.        , 1.        , 0.        , 0.9193292 ,
           1.        , 0.        , 1.        , 1.        , 0.89694649,
           1.        , 0.        , 0.        , 1.        , 0.        ,
           1.        , 0.        , 0.93628113, 0.87625082, 0.86615573,
           0.88991694, 1.        , 0.84901864, 0.        , 0.        ,
           0.95286948, 1.        , 1.        , 1.        , 1.        ,
           0.        , 1.        , 0.990829  , 0.        , 1.        ,
           1.        , 0.9487945 , 1.        , 0.90683885, 0.9887377 ,
           1.        , 0.96009641, 0.        , 0.        , 0.78228499,
           0.87593243, 0.88572323, 0.75630357, 0.        , 0.        ,
           0.        , 1.        , 0.89183459, 1.        , 0.88991694,
           1.        , 1.        , 1.        , 1.        , 1.        ,
           1.        , 0.        , 1.        , 1.        , 0.92710378,
           1.        , 1.        , 0.91391766, 0.        , 0.83683651,
           0.89252349, 0.        , 0.        , 1.        , 1.        ,
           0.80918411, 1.        , 1.        , 0.97002571, 0.86544138,
           0.        , 1.        , 1.        , 1.        , 0.77364856,
           0.87778986, 1.        , 1.        , 1.        , 0.97144984,
           0.8208196 , 0.93498166, 0.95880203, 1.        , 0.        ,
           0.        , 1.        , 0.96650654, 1.        , 1.        ,
           0.77822376, 1.        , 0.73659246, 1.        , 1.        ,
           0.86517975, 0.9374977 , 1.        , 0.        , 0.        ,
           0.        , 1.        , 1.        , 1.        , 1.        ,
           0.        , 1.        , 1.        , 0.        , 1.        ,
           1.        , 1.        , 0.92059499, 0.        , 0.        ,
           0.        , 1.        , 1.        , 1.        , 1.        ,
           0.78196891, 1.        , 1.        , 0.        , 1.        ,
           0.7420706 , 1.        , 1.        , 0.        , 0.97020602,
           0.78045212, 0.99375031, 0.        , 1.        , 1.        ,
           1.        , 0.        , 1.        , 1.        , 0.83922967,
           1.        , 1.        , 1.        , 0.68249003, 1.        ,
           1.        , 0.78798532, 0.        , 1.        , 0.81097314,
           1.        , 1.        , 1.        , 0.        , 1.        ,
           1.        , 0.88991694, 0.959651  , 0.96454615, 0.75406635,
           0.93777316, 1.        , 0.91200777, 1.        , 1.        ,
           0.93360186, 1.        , 0.89854845, 1.        , 0.82769429,
           0.94905591, 0.98695901, 1.        , 0.9372658 , 1.        ,
           1.        , 0.        , 1.        , 0.80830496, 1.        ,
           0.        , 0.90336065, 1.        , 1.        , 0.9502046 ,
           1.        , 0.93838806, 0.99031739, 0.96112631, 0.        ,
           0.96150881, 0.        , 1.        , 0.        , 1.        ,
           1.        , 1.        , 0.93190489, 1.        , 0.85388172,
           1.        , 0.76389665, 1.        , 0.98371285, 0.        ,
           0.8967203 , 1.        , 0.93336613, 1.        , 0.92377618,
           0.        , 1.        , 0.        , 1.        , 1.        ,
           1.        , 0.89262043, 0.        , 0.81450053, 0.98771091,
           1.        , 0.64635585, 0.99263572, 1.        , 1.        ,
           1.        , 1.        , 1.        , 0.        , 0.75991998,
           0.81508628, 0.78468559, 0.90591145, 1.        , 1.        ,
           1.        , 0.        , 0.86620093, 1.        , 0.96150714,
           1.        , 0.        , 0.98767175, 1.        , 1.        ,
           0.        , 1.        , 0.8668776 , 0.84153007, 1.        ,
           1.        , 1.        , 1.        , 0.        , 0.76774546,
           1.        , 0.        , 0.        , 0.83741056, 0.98615141,
           1.        , 1.        , 0.        , 0.59430149, 0.93886863,
           1.        , 1.        , 0.88991694, 0.93522755, 1.        ,
           1.        , 1.        , 0.92146292, 0.        , 1.        ,
           0.        , 0.        , 0.86607902, 0.        , 0.        ,
           1.        , 1.        , 1.        , 0.88991694, 0.        ,
           1.        , 0.94600474, 1.        , 0.        , 1.        ,
           1.        , 1.        , 0.88991694, 0.89332269, 0.95167332,
           1.        , 0.99247752, 1.        , 1.        , 0.        ])




```python
# Add cluster ID and probability to df
arr_data['cluster_id'] = list(cluster.labels_)
arr_data['cluster_prob'] = list(cluster.probabilities_)

# Add plotting values
arr_data['x'] = [x[0] for x in umap_embeddings_plot]
arr_data['y'] = [x[1] for x in umap_embeddings_plot]

```

### 3.7 Plotting


```python
# Add plotting colours to the df
lst_colors = (px.colors.qualitative.Dark24 + px.colors.qualitative.Light24 + px.colors.qualitative.Alphabet) * 15
lst_colors = ['#D2D2D2'] + lst_colors # Add light colour for Topic -1 (unmapped)
arr_colours = pd.DataFrame(lst_colors)
arr_colours['cluster_id'] = arr_colours.index - 1
arr_colours.columns = ['colour', 'cluster_id']
arr_data = pd.merge(arr_data, arr_colours, on=['cluster_id'])

```


```python
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
```

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)





    '/Users/barry.trim/Documents/03_vs_code/08_python/nlp_overview/src/output/211201_160634_topic_model.html'


