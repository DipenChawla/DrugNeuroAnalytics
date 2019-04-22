
import numpy as np
import pandas as pd
from pprint import pprint
import keras
import codecs
import re
import seaborn as sns
import pickle

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)

reviews = pd.read_csv("All_Labelled_Reviews.csv")
reviews['category_id'] = reviews['Category'].factorize()[0]
reviews = reviews[reviews.category_id != 3]
reviews.reset_index(drop = True, inplace = True)
category_id_df = reviews[['Category', 'category_id']].drop_duplicates().sort_values('category_id')
id_to_category = dict(category_id_df[['category_id', 'Category']].values)
category_to_id = dict(category_id_df.values)
reviews.head()

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
def analyze_sentiment(df):
    sentiments = []
    sid = SentimentIntensityAnalyzer()
    for i in range(df.shape[0]):
        line = df['Review'].iloc[i]
        sentiment = sid.polarity_scores(line)
        sentiments.append([sentiment['neg'], sentiment['pos'],
                           sentiment['neu'], sentiment['compound']])
    df[['neg', 'pos', 'neu', 'compound']] = pd.DataFrame(sentiments)
#     df['Negative'] = df['compound'] < -0.1
#     df['Positive'] = df['compound'] > 0.1
    return df

analyze_sentiment(reviews).head()

from sklearn.model_selection import train_test_split

list_corpus = reviews["Review"].tolist()
list_labels = reviews["Category"].tolist()

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=word_tokenize, max_features=15000)
vector = vectorizer.fit_transform(list_corpus)

df2 = pd.DataFrame(vector.toarray())
df2.head()

df_final = pd.concat([df2, reviews], axis =1)
df_final = df_final.drop('Review', axis =1)
df_final = df_final.drop('Category', axis =1)
df_final = df_final.drop('Drug', axis = 1)
df_final_labels = pd.DataFrame(list_labels)
df_final = df_final.drop('category_id', axis = 1)
df_final.head()

X_train, X_test, y_train, y_test,indices_train, indices_test = train_test_split(df_final, df_final_labels,reviews.index, test_size=0.2,
                                                                                random_state=42)

from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
svc = LinearSVC(dual=False, multi_class='ovr', class_weight='balanced')
scores = cross_val_score(svc, X_train, y_train, scoring='f1_weighted', n_jobs=-1, cv=10)
print('Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100, np.std(scores) * 100))

svc.fit(X_train, y_train)
review = 'felt like a zombie'
list = [review]
test = pd.DataFrame(list, columns = ['Review'])
df1 = analyze_sentiment(test)
review = vectorizer.transform([review])
testt = pd.DataFrame(review.toarray())
df_finaltest = pd.concat([testt, df1], axis =1)
df_finaltest = df_finaltest.drop('Review', axis = 1)
print(svc.predict(df_finaltest))


pickle.dump(svc, open('model.pkl','wb'))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Loading model to compare the results
# model = pickle.load(open('model.pkl','rb'))
# print(model.predict(tfidf_vectorizer.transform([review])))
