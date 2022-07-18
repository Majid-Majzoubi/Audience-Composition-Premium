#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import pandas as pd
import numpy as np
import re
import pickle

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.callbacks import PerplexityMetric, ConvergenceMetric, CoherenceMetric


import spacy
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
%matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from datetime import datetime                    

from scipy.spatial.distance import cosine

from datetime import datetime

os.chdir('/ML data and analysis/transcripts')

# =============================================================================
# IMPORTING THE MAIN FILES
# =============================================================================
speaker1 = pd.read_csv('speaker detail 1.csv')
speaker2 = pd.read_csv('speaker detail 2.csv')
speaker3 = pd.read_csv('speaker detail 3.csv')
speaker4 = pd.read_csv('speaker detail 4.csv')
speaker5 = pd.read_csv('speaker detail 5.csv')
speaker6 = pd.read_csv('speaker detail 6.csv')

speaker = pd.concat([speaker1, speaker2, speaker3, speaker4, speaker5, speaker6])
del [speaker1, speaker2, speaker3, speaker4, speaker5, speaker6]


speaker['transcriptcomponentid'] = speaker['transcriptcomponentid'].apply(
    pd.to_numeric, errors='coerce').astype('Int64')
speaker.dropna(subset=['transcriptcomponentid', 'transcriptid'], inplace=True)

speaker = speaker[pd.to_numeric(speaker['transcriptid'], errors='coerce').notnull()]
speaker['transcriptid'] = speaker['transcriptid'].astype('str')
speaker['transcriptid_isnumber'] = speaker['transcriptid'].str.isnumeric()
speaker = speaker[speaker['transcriptid_isnumber'] == True]
speaker['transcriptid'] = speaker['transcriptid'].astype('float64')
del speaker['transcriptid_isnumber']
speaker.to_pickle('speaker detail all')

# Loading files cleaned from previous steps
speaker = pd.read_pickle('speaker detail all')
calls = pd.read_csv('Trascripts details.csv')


# =============================================================================
# PREPROCESS QUESTIONS
# =============================================================================

questions = speaker[speaker.transcriptcomponenttypename== 'Question'][[
    'transcriptcomponentid', 'componenttextpreview']]
questions.dropna(inplace=True)
questions.drop_duplicates(subset='transcriptcomponentid', inplace=True)
questions.set_index('transcriptcomponentid', inplace=True)

data = questions.componenttextpreview.values.tolist()


# Cleaning the questions
def sent_to_words(sentences):
    """ This lowercases, tokenizes, de-accents
    """
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  

data_words = list(sent_to_words(data))

# Remove numbers, but not words that contain numbers.
data_words = [[token for token in doc if not token.isnumeric()] for doc in data_words]

# Remove words that are only one character.
data_words = [[token for token in doc if len(token) > 1] for doc in data_words]

# Build the bigram model
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
bigram_mod = gensim.models.phrases.Phraser(bigram)

# Define functions for stopwords, bigrams, and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
print('lemmatizing data')
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

with open(r'topic modeling/data_lemmatized.pkl', 'wb') as handle:
    pickle.dump(data_lemmatized, handle, protocol=pickle.HIGHEST_PROTOCOL)   
    
# =============================================================================
# Create the Dictionary and Corpus needed for Topic Modeling
# =============================================================================
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)
id2word.filter_extremes(no_below=50, no_above=0.5)

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_lemmatized]

with open(r'topic modeling/corpus.pkl', 'wb') as handle:
    pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

# =============================================================================
# Finding the best number of topics for the topic model - RESULT:30
# =============================================================================
# Using coherence metric to choose the number of topics
random.Random(5).shuffle(data_lemmatized)
train_data_token = data_lemmatized[:int((len(data_lemmatized)+1)*.25)] #25% of datafor training
test_data_token = data_lemmatized[int((len(data_lemmatized)+1)*.25):
                                  int((len(data_lemmatized)+1)*.30)] #5% of data for test set

train_data_corpus = [id2word.doc2bow(text) for text in train_data_token]
test_data_corpus = [id2word.doc2bow(text) for text in test_data_token]


def compute_coherence_values(
        train_data_token, train_data_corpus, test_data_corpus, dictionary, k):
    """ This function measures coherence and perplexity score of a
    trained LDA model
    """
    print ("***\n***\nTraining an LDA model with {} topics \n\n".format(k))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
    start_time = datetime.now()
    lda_model = gensim.models.LdaModel(corpus=train_data_corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100000,
                                           passes=5,
                                           alpha='auto',
                                           eta='auto')  
    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=train_data_token, corpus=train_data_corpus, 
        coherence='c_v')   
    coherence = coherence_model_lda.get_coherence()   
    perplexity = lda_model.log_perplexity(test_data_corpus)
    finish_time = datetime.now()
    time_to_finish = finish_time - start_time
    return ([coherence, perplexity, time_to_finish])

# Selecting a range for the number of topics to be tested for
topics_range = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 
                90, 100, 110 ,120, 130, 140, 150, 160, 170, 180, 190, 200]

# Running the function for different numbers of topics
coherence_values = {}
for n_topics in topics_range:
    coherence_values[n_topics] = compute_coherence_values(
        train_data_token, train_data_corpus, test_data_corpus, id2word, n_topics)
 
coherence_df = pd.DataFrame(coherence_values).transpose().reset_index()
coherence_df = coherence_df.iloc[:, :-1]
coherence_df.columns = ['n_topics', 'coherence', 'perplexity']
coherence_df.to_pickle('topic modeling/coherence_values_10_200.pkl')

# plotting coherence and perplexity metrics
plt.rc('font',family='Times New Roman')
plt.rcParams['figure.dpi'] = 900
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Number of topics')
ax1.set_ylabel('Coherence', color=color)
ax1.plot(coherence_df.head(26)['n_topics'], coherence_df.head(26)['coherence'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Log perplexity', color=color)  
ax2.plot(coherence_df.head(26)['n_topics'], coherence_df.head(26)['perplexity'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.savefig('topic_model_evaluation_transcripts.jpg', dpi=300)
plt.show()


# =============================================================================
# Building the topic model
# =============================================================================
print('creating LDA')
# Build LDA model

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=30, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=10000,
                                           passes=10,
                                           alpha='auto',
                                           eta = 'auto',
                                           per_word_topics=True,
                                           eval_every=1)

with open(r'topic modeling/lda_model130.pkl', 'wb') as handle:
    pickle.dump(lda_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(r'topic modeling/lda_model30.pkl', 'rb') as handle:
    lda_model = pickle.load(handle)


# =============================================================================
# Measuring "heterogeneity in evaluative schema of the audiences a firm engages with"
# =============================================================================
calls.dropna(subset=['companyid', 'mostimportantdateutc'], inplace=True)
calls['companyid'] = calls['companyid'].astype('int64')
calls['mostimportantdateutc'] = calls['mostimportantdateutc'].astype('int64')
calls['year'] = calls['mostimportantdateutc'] // 10000

calls_IDs = calls[['companyid', 'year', 'transcriptid'
                   ]].drop_duplicates().set_index(['companyid', 'year'])

speaker_IDs = speaker[['transcriptid','transcriptcomponentid']].drop_duplicates().set_index(
    'transcriptid').squeeze()

calls_IDs = calls_IDs.squeeze()

def get_firmyear(companyID, Year):
    # get list of all transcriptIDs
    try:
        transcriptIDs= list(calls_IDs.loc[companyID, Year].values)
    except:
        print('No transrcipts found')
        transcriptIDs = []    
    
    # get list of all components
    transcript_components = []
    
    for ID in transcriptIDs:
        try:
            components = list(speaker_IDs.loc[ID].values)
            transcript_components.extend(components) 
        except:
            pass    
    return transcript_components

map_table = questions.reset_index().reset_index().set_index('transcriptcomponentid').iloc[:,0]

def convert_id_row(componentIDS): # Change test to questions
    #row_numbers = list(map_table.loc[componentIDS].values)
    row_numbers = list(map_table.loc[map_table.index.intersection(componentIDS)].values)   
    return row_numbers

def get_topics(componentIDs):
    row_numbers = convert_id_row(componentIDs)
    lda_corpus = [dict(lda_model[corpus[x]][0]) for x in row_numbers]
    lda_corpus_df = pd.DataFrame(lda_corpus)
    lda_corpus_df = lda_corpus_df.fillna(0)
    return lda_corpus_df
    
def calculate_diversity(companyID, Year):
     componentIDs = get_firmyear(companyID, Year)
     if len(componentIDs)==0:
         return np.nan
     else:
         row_numbers = get_topics(componentIDs)
         topics = get_topics(componentIDs)
         #topics = pd.DataFrame([[1,2,3], [1,3,5], [5,5,5], [7,7,8]])
         topics_n = topics.shape[0]
    
         if (topics_n)<=1:
             return np.nan
         else:
             mean_topics = topics.mean()
             topics_dmeaned = topics-mean_topics
             topics_dmeaned_sq = topics_dmeaned.pow(2)
             topics_dmeaned_sum = topics_dmeaned_sq.sum()
             topics_dmeaned_sum_div = topics_dmeaned_sum/(topics_n-1)
             topics_dmeaned_sum_sqrt = topics_dmeaned_sum_div.pow(0.5)
             diversity = topics_dmeaned_sum_sqrt.sum()
             return diversity
        
firm_years = calls[['companyid', 'year']].groupby(
    ['companyid', 'year']).first().reset_index()


diversity_measures = {}
for index, row in firm_years.head(100).iterrows():
    diversity_measures[(row['companyid'], row['year'])] = calculate_diversity(
        row['companyid'], row['year'])
    

# =============================================================================
# Develop diversity measure for select firms
# =============================================================================
reg_firms = pd.read_pickle(r'../predicted_values_distances_lda70_full.pkl')

reg_firms = reg_firms[['cusip', 'year']].drop_duplicates()

mapping_cusip = pd.read_csv('identifiers_cusip.csv')
mapping_cusip['cusip'] = mapping_cusip['cusip'].str.slice(0,-1)
mapping_cusip.startdate = mapping_cusip.startdate.dropna()

mapping_cusip['startdate_num'] = mapping_cusip['startdate'].str.isnumeric()
mapping_cusip = mapping_cusip[mapping_cusip['startdate_num'] == True]
mapping_cusip['startdate'] = mapping_cusip['startdate'].astype('int')
mapping_cusip['startdate'] = mapping_cusip['startdate'] // 10000

mapping_cusip['enddate_num'] = mapping_cusip['enddate'].str.isnumeric()
mapping_cusip = mapping_cusip[mapping_cusip['enddate_num'] == True]
mapping_cusip['enddate'] = mapping_cusip['enddate'].astype('int')
mapping_cusip['enddate'] = mapping_cusip['enddate'] // 10000

reg_firms_id = reg_firms.merge(mapping_cusip, on='cusip', how='inner')
reg_firms_id_s = reg_firms_id.query(
    'year<=enddate & year>=startdate')

reg_firms_id_s.drop_duplicates(['cusip', 'year'], inplace=True)
reg_firms_id_s.drop_duplicates(['companyid', 'year'], inplace=True)

reg_firms_id_s = reg_firms_id_s[['cusip', 'year', 'companyid']]

reg_firms_id_s.reset_index(inplace=True)
reg_firms_id_s.columns = ['n', 'cusip', 'year', 'companyid']


diversity_measures = {}
for index, row in reg_firms_id_s.iterrows():
    diversity_measures[(row['cusip'], row['year'])] = calculate_diversity(
        row['companyid'], row['year'])    
    print(datetime.now(), "     firm number:  ", row['n'])
    
diversity_measures_df = pd.DataFrame.from_dict(diversity_measures, orient='index')

diversity_measures_df.columns = ['diversity_engagement']
diversity_measures_df.reset_index(inplace=True)
diversity_measures_df[['cusip', 'year']] = pd.DataFrame(
    diversity_measures_df['index'].tolist(), index=diversity_measures_df.index)
del diversity_measures_df['index']
diversity_measures_df.to_pickle('diversity_engagement_lda30.pkl')


# =============================================================================
# Develop controls
# =============================================================================
# Number of questions asked
n_questions = speaker[speaker['transcriptcomponenttypename']=='Question'].groupby(
    'transcriptid')['transcriptcomponentid'].nunique()

# Number of people asking questions
n_persons = speaker[speaker['transcriptcomponenttypename']=='Question'].groupby(
    'transcriptid')['transcriptpersonid'].nunique()

# Number of words in the the Q&A
n_question_words = speaker[speaker['transcriptcomponenttypename']=='Question'].groupby(
    'transcriptid')['word_count'].sum()

# Number of words in the presentation
n_present_words = speaker[speaker['transcriptcomponenttypename']=='Presenter Speech'].groupby(
    'transcriptid')['word_count'].sum()

# total number of words spoken
# Number of words in the presentation
n_total_words = speaker.groupby(
    'transcriptid')['word_count'].sum()

# merging together
transcripts_controls = pd.concat([n_questions, n_persons, n_question_words,
                                  n_present_words, n_total_words], axis=1)

transcripts_controls.columns = ['n_questions', 'n_persons', 'n_question_words',
                                  'n_present_words', 'n_total_words']


def calculate_controls(companyID, Year):
    try:
        transcriptIDs= list(calls_IDs.loc[companyID, Year].values)
    except:
        print('No transrcipts found')
        transcriptIDs = []  
    
    if len(transcriptIDs)==0:
        return np.nan
    else:
         controls = transcripts_controls.loc[
             transcripts_controls.index.intersection(transcriptIDs).values]
         return list(controls.sum().values)
         
control_measures = {}
for index, row in reg_firms_id_s.iterrows():
    control_measures[(row['cusip'], row['year'])] = calculate_controls(
        row['companyid'], row['year'])    
    print(datetime.now(), "     firm number:  ", row['n'])

control_measures_na = {k: v for k, v in control_measures.items() if v == v}

control_measures_df = pd.DataFrame.from_dict(control_measures_na, orient='index')
control_measures_df.dropna(inplace=True)
control_measures_df.columns = ['n_questions', 'n_persons', 'n_question_words',
                                  'n_present_words', 'n_total_words']
                                      
control_measures_df.reset_index(inplace=True)

control_measures_df[['cusip', 'year']] = pd.DataFrame(
    control_measures_df['index'].tolist(), index=control_measures_df.index)

del control_measures_df['index']

diversity_controls = pd.merge(diversity_measures_df, control_measures_df,
                              on=['cusip', 'year'], how='left')

diversity_controls.to_pickle('diversity_engagement_controls_lda30.pkl')
