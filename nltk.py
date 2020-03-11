#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.book import *


# In[2]:


from gensim.models import Word2Vec
from nltk.corpus import stopwords
from string import punctuation
import pprint as pp


# ### Word and sentence tokenizer

# Word tokenizer breaks down the entire text strings into tokens and put them into a list.
# Use pos_tag to break a block of text down into its parts of speech 

# In[3]:


from nltk.tokenize import word_tokenize, sent_tokenize
w = word_tokenize('The novel corona virus has infected more than 108,000 people around the world')
print(w)

# break down text into parts of speech
print(nltk.pos_tag(w))
print(nltk.pos_tag(w, tagset = 'universal'))


# Sentence tokenizer breaks text down into a list of sentences

# In[4]:


s1 = 'Coronavirus: On the front line in Wuhan. COVID-19 is the name of the virus' # 2 sentences
print(sent_tokenize(s1))
s2 = 'Coronavirus. On the front line in Wuhan. COVID-19 is the name of the virus' # 3 sentences
print(sent_tokenize(s2))


# ### Corpora in other languages

# In[19]:


from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch', 'Hungarian_Magyar']
cfd = nltk.ConditionalFreqDist((lang, len(word))
                               for lang in languages
                               for word in udhr.words(lang + '-Latin1'))
cfd.plot(cumulative=True)


# ### Word2Vec

# Used to convert a document to a vector representation. Similar words will have similar values - floating points values.                                                                                                       

# In[20]:


from gensim.models import Word2Vec
caesar_vec = Word2Vec(gutenberg.sents('shakespeare-caesar.txt'))
macbeth_vec = Word2Vec(gutenberg.sents('shakespeare-macbeth.txt'))

print(caesar_vec.wv.most_similar('pray', topn=10)) # top 10 similar words 'pray' in caesar text
print(macbeth_vec.wv.most_similar('pray', topn=10)) # top 10 similar words 'pray' in macbeth text


# In[21]:


bible = gutenberg.sents('bible-kjv.txt')
stopw = stopwords.words('english')
biblew = [[w.lower() for w in s if w not in punctuation and w not in stopw] for s in bible]
print(len(biblew))

bible_vec = Word2Vec(biblew)
pp.pprint(bible_vec.wv.most_similar('satan', topn=10))
pp.pprint(bible_vec.wv.most_similar('devil', topn=10))


# ### Stemming and Lemmatization

# Stemming: Normalization process in which sentences or words are reduced to their base forms by getting rid of the word endings. Results you get may or may not be a actual word.

# In[22]:


from nltk.stem import PorterStemmer
st = PorterStemmer()

words = ['are', 'signs', 'infection', 'coronavirus', 'corona', 'symptoms', 
         'shortness', 'short', 'breaths', 'breathing', 'played', 'plays']
for word in words:
    print(word, st.stem(word)) # prints out the word and the stem version of the word


# Lemmatizer returns the root word back

# In[23]:


from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()
words = ['are', 'signs', 'infection', 'coronavirus', 'corona', 'symptoms',
      'shortness', 'short', 'breaths', 'breathing', 'played', 'plays']

for word in words:
    print(word, lm.lemmatize(word)) # prints out the word and the lem version of the word

