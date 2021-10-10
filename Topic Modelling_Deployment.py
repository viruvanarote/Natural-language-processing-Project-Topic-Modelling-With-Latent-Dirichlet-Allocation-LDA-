from bs4 import BeautifulSoup as bs
import requests
import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)
import numpy as np
import pandas as pd
import streamlit as st 
import matplotlib.pyplot as plt

st.title('Model Deployment: Topic Modelling')

st.sidebar.header('User Input Parameters :: GOOGLE NEWS')


link = 'https://news.google.com/news/rss'

page = requests.get(link)
#page.content

soup = bs(page.content,'html.parser')
news = soup.find_all('title',)
#news

statement = []
for i in range(0,len(news)):
    statement.append(news[i].get_text())

statement.pop(0)

import pandas as pd
df = pd.DataFrame()
df['x'] = statement
#df.head()	

data = df.x.values.tolist()
#data

import re
# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

data = pd.Series(data)

import re
data_processed = data.apply(lambda x: re.sub(r'[^a-zA-Z/s]+',' ',x).lower())
#data_processed

import nltk
#nltk.download('stopwords')
#nltk.download('punkt')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(["the", "india", "times", "hindustan times", "indian express", "hindustan", "indian",'gsmarenacom', "express", "ndtv",'news18','news','DNA India','moneycontrol','news18','gsmarena','com'])
len(stop_words)

from nltk import word_tokenize

data_final = data_processed.apply(lambda x:' '.join([word for word in word_tokenize(x) if word not in stop_words and len(word)> 2]))

#data_final[0:2]

words_list = []
for sentence in data_final:
    words_list.extend(nltk.word_tokenize(sentence))
freq_dist = nltk.FreqDist(words_list)
freq_dist.most_common(20)
#freq_dist.keys()

words_list

# creating a temporary dataframe and plotting the graph
#%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
temp = pd.DataFrame(freq_dist.most_common(30),  columns=['word', 'count'])
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='word', y='count', 
            data=temp, ax=ax)
plt.title("Top words")
plt.xticks(rotation='vertical');
st.pyplot(fig)
#%config InlineBackend.figure_format = 'retina'

from wordcloud import WordCloud
import wordcloud
# creation of wordcloud
wcloud_fig = WordCloud( stopwords=set(wordcloud.STOPWORDS),
                      colormap='viridis', width=3000, height=2000).generate_from_frequencies(freq_dist)
#plotting the wordcloud
plt.figure(figsize=(16,10), frameon=True)

plt.imshow(wcloud_fig, interpolation  = 'bilinear')
plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

import gensim
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim import models

words_sent = [x.split() for x in data_final]
#words_sent
bigram = Phrases(words_sent, min_count=3, threshold=100)
trigram = Phrases(bigram[words_sent], threshold=100)  

bigram_phraser = Phraser(bigram)
trigram_phraser = Phraser(trigram)

# bow = [bigram_phraser[word] for word in words_sent] # creating bigram
bow = [trigram_phraser[bigram_phraser[words]] for words in words_sent] 

#words_sent

bow[0:10]

import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

doc = nlp(' '.join(bow[2]))
for token in doc:
    print(token ,'=>', token.pos_)

def lemmatization(texts, tags=['NOUN', 'ADJ', 'VERB', 'ADV','PROPN']): # filter noun and adjective(for topic modelling we need only this filter)
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in tags])
    return texts_out

bag_of_words = lemmatization(bow)
#bag_of_words

from gensim import corpora
id2word = corpora.Dictionary(bag_of_words)
print(id2word)

corpus_matrix = [id2word.doc2bow(sent) for sent in bag_of_words]

corpus_matrix[0]

import gensim
LDA = gensim.models.ldamodel.LdaModel

# Build LDA model
lda_model = LDA(corpus=corpus_matrix,id2word=id2word,
                                    num_topics=10, 
                                    random_state=100,
                                           update_every=1,
                                           chunksize=200,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

for idx, topic in lda_model.print_topics(-1):
   st.write(print("Topic: {} \nWords: {}".format(idx, topic )))
   print("\n")