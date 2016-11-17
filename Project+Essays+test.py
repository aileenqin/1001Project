
# coding: utf-8

# In[10]:

import pandas as pd
import numpy as np
essays = pd.read_csv('C:/Users/Aileen/Downloads/opendata_essays000.gz', escapechar='\\', names=['_projectid', '_teacherid', 'title', 'short_description', 'need_statement', 'essay', 'thankyou_note', 'impact_letter'])


# In[11]:

essays.drop(['_projectid', '_teacherid', 'thankyou_note', 'impact_letter'], axis = 1, inplace = True, errors = 'ignore')


# In[13]:

new


# In[12]:

a = np.random.randint(0,len(essays),size = 6000)
#projects1 = projects[:30]
new = essays.iloc[a,:]['essay']


# In[14]:

import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction


# In[15]:

#nltk.download()
#Words that don't covey significant meaning
stopwords = nltk.corpus.stopwords.words('english')

#Stemming: breaking a word down into its root
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')


# In[16]:

#define a tokenizer and stemmer which returns the set of stems in the text that it is passed

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


# In[17]:

#functions to iterate over the list of essays to create two vocabularies: 
# one stemmed and one only tokenized
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in new:#['short_description']:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)


# In[18]:

#create DataFrame with the stemmed vocabulary as the index and the tokenized words as the column. 
#The benefit of this is it provides an efficient way to look up a stem and return a full token
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')


# In[19]:

from sklearn.feature_extraction.text import TfidfVectorizer
#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(new)#['short_description']) #fit the vectorizer to synopses
print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)
dist


# In[20]:

from sklearn.cluster import KMeans

'''Number of clusters??'''
num_clusters = 7
km = KMeans(n_clusters=num_clusters)
get_ipython().magic('time km.fit(tfidf_matrix)')
clusters = km.labels_.tolist()

from sklearn.externals import joblib

#uncomment the below to save your model 
#since I've already run my model I am loading from the pickle

joblib.dump(km,  'doc_cluster.pkl')

km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()


# In[21]:

pd.value_counts(pd.Series(clusters))# = new['cluster']#.value_counts()


# In[22]:

from __future__ import print_function

print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0], end=',')
    print() #add whitespace
    print() #add whitespace
    
    #print("Cluster %d titles:" % i, end='')
    #for title in new.ix[i]['title'].values.tolist():
     #   print(' %s,' % title, end='')
    #print() #add whitespace
    #print() #add whitespace


# In[ ]:

import string
def strip_proppers(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word.islower()]
    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()


# In[ ]:

from nltk.tag import pos_tag

def strip_proppers_POS(text):
    tagged = pos_tag(text.split()) #use NLTK's part of speech tagger
    non_propernouns = [word for word,pos in tagged if pos != 'NNP' and pos != 'NNPS']
    return non_propernouns


# In[ ]:

from gensim import corpora, models, similarities 

#remove proper names
get_ipython().magic("time preprocess = [strip_proppers(doc) for doc in new]#['short_description']]")

#tokenize
get_ipython().magic('time tokenized_text = [tokenize_and_stem(text) for text in preprocess]')

#remove stop words
get_ipython().magic('time texts = [[word for word in text if word not in stopwords] for text in tokenized_text]')


# In[ ]:

#create a Gensim dictionary from the texts
dictionary = corpora.Dictionary(texts)

#remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
dictionary.filter_extremes(no_below=1, no_above=0.8)

#convert the dictionary to a bag of words corpus for reference
corpus = [dictionary.doc2bow(text) for text in texts]

get_ipython().magic('time lda = models.LdaModel(corpus, num_topics=5, id2word=dictionary,update_every=5,chunksize=10000,passes=100)')

lda.show_topics()


# In[ ]:

topics_matrix = lda.show_topics(formatted=False, num_words=20)
topics_matrix = np.array(topics_matrix)

topic_words = topics_matrix[:,:,1]
for i in topic_words:
    print([str(word) for word in i])
    print()


# In[ ]:

topics_matrix

