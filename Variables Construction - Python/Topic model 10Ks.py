# -*- coding: utf-8 -*-


import os
import pandas as pd
import pickle

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import random

from six import iteritems
import warnings

import matplotlib.pyplot as plt


os.chdir('/ML data and analysis/10ks')


# =============================================================================
# PREPARING THE CORPUS FOR THE TOPIC MODEL
# Import a bigrammed transformation of all the 10Ks
# Create a corpus dictionary
# =============================================================================

bigram_10k_filepath = 'bigram_transformed_10ks_all.txt'
with open(bigram_10k_filepath) as f:
    data = f.readlines()
    
dictionary_filepath = 'master_dictionary.dict'

if True: # Use a False statement to load a previously built dictionary
     dictionary = corpora.Dictionary(line.lower().split() for line in data)
     once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if 
                 docfreq == 1]
     dictionary.filter_tokens(once_ids)  # remove stop words and words that appear only once     
     dictionary.filter_extremes(no_below=50, no_above=0.3, keep_n=None)
     dictionary.compactify()  # remove gaps in id sequence after words that were removed
     dictionary.save(dictionary_filepath)
        
dictionary = corpora.Dictionary.load(dictionary_filepath)


# =============================================================================
# EVALUATING THE TOPIC MODELS BASED ON THE NUMBER OF TOPICS
# =============================================================================

# Using coherence and perplexity metrics to choose the number of topics
random.Random(5).shuffle(data)
train_data = data[:int((len(data)+1)*.80)] #Select 80% for training set
test_data = data[int((len(data)+1)*.80):] #Remaining 20% data for test set

train_data_token = [text.lower().split() for text in train_data]
train_data_corpus = [dictionary.doc2bow(text) for text in train_data_token]

test_data_token = [text.lower().split() for text in test_data]
test_data_corpus = [dictionary.doc2bow(text) for text in test_data_token]


def compute_coherence_values(
        train_data_token, train_data_corpus, test_data_corpus, dictionary, k):
    """ This function measures coherence and perplexity score of a
    trained LDA model
    """
    
    print ("***\n***\nTraining an LDA model with {} topics \n\n".format(k))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
    
    lda_model = gensim.models.LdaModel(corpus=train_data_corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=10000,
                                           passes=5,
                                           alpha='auto',
                                           eta='auto')
    
    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=train_data_token, corpus=train_data_corpus, coherence='c_v')   
    coherence = coherence_model_lda.get_coherence()
    
    perplexity = lda_model.log_perplexity(test_data_corpus)
    
    return ([coherence, perplexity])

# Selecting a range for the number of topics to be tested for
topics_range = range(40, 180, 10)

# Running the function for different numbers of topics
coherence_values = {}
for n_topics in topics_range:
    coherence_values[n_topics] = compute_coherence_values(
        train_data_token, train_data_corpus, test_data_corpus, dictionary, 
        n_topics)
 
coherence_df = pd.DataFrame(coherence_values).transpose().reset_index()
coherence_df.columns = ['n_topics', 'coherence', 'perplexity']
coherence_df.to_pickle('coherence_values.pkl')

coherence_df = pd.read_pickle('coherence_values.pkl')

# plotting coherence and perplexity metrics
plt.rc('font',family='Times New Roman')
plt.rcParams['figure.dpi'] = 900
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Number of topics')
ax1.set_ylabel('Coherence', color=color)
ax1.plot(coherence_df['n_topics'], coherence_df['coherence'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Log perplexity', color=color)  
ax2.plot(coherence_df['n_topics'], coherence_df['perplexity'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout() 
plt.savefig('topic_model_evaluation_10ks.jpg', dpi=300)
plt.show()


# =============================================================================
# TRAINING THE MAIN MODEL - NUMBER OF TOPICS = 70
# =============================================================================

# Taking the bigrammed corpus into vector space using the dictionary

# Term Document Frequency
corpus = [dictionary.doc2bow(text.lower().split()) for text in data]

with open('corpus.pkl', 'wb') as handle:
    pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

lda_model = gensim.models.LdaModel(corpus=corpus,
                                       id2word=dictionary,
                                       num_topics=70, 
                                       random_state=100,
                                       chunksize=10000,
                                       passes=10,
                                       alpha='auto',
                                       eta='auto')

lda_model.save('LDA70.mm')


# =============================================================================
# TRANSFORMING THE CORPUS INTO TOPIC WEIGHT VECTORS USING THE TRAINED LDA MODEL
# =============================================================================

lda_corpus = [dict(lda_model[x]) for x in corpus]
lda_corpus_df = pd.DataFrame(lda_corpus)
lda_corpus_df = lda_corpus_df.fillna(0)

# index file loading -- the index file maps 10Ks in the corpus to firm information
# such as GVKEY and CUSIP and filing date etc.
index_file = pd.read_pickle('../index_file.pkl')

# Merging the new index file with the corpus lda file
lda_corpus_df2 = pd.merge(lda_corpus_df.reset_index(), index_file[['CUSIPH','fdate','index_reset']],
                          left_on=['index'], right_on=['index_reset'], how='inner'
                          ).drop(columns=['index', 'index_reset']).set_index(['CUSIPH','fdate'])
lda_corpus_df2 = lda_corpus_df2.reset_index().sort_values(by=['CUSIPH','fdate']).dropna(
        subset=['CUSIPH'])

lda_corpus_df2['year'] = lda_corpus_df2['fdate'].astype(str).str.slice(stop=4).astype(int)

lda_corpus_df2.to_csv('lda70_corpus.csv')


