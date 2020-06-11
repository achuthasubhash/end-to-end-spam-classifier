# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 05:40:50 2020

@author: DELL
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt

data=pd.read_csv("E:\\github\\my upload\\end to end spam classifier\\Spam-email-Classifier-Deployment-masteremails.csv")

data.head()

data['spam'].unique()

data.shape

import re
import nltk
nltk.download('stopwords') #contain irrelvant words  & ava in diff lang
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = [] #corpus collection of strings
for i in range(0,5728):
    review = re.sub('[^a-zA-Z]', ' ', str(data['text'][i])) #remove  except a-z & A-Z & create space b/w words
    review = review.lower() #capital to lower
    review = review.split()  #sentence to words
    ps = PorterStemmer()  # loved to love (diff kind of same word into standard word)
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #take words not in stopward
    review = ' '.join(review) #again to string sep by space
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 8000)
X = cv.fit_transform(corpus).toarray()
y = data['spam']

X.shape

y.shape


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Creating a pickle file for the CountVectorizer
pickle.dump(cv, open('cv-transform.pkl', 'wb'))
# Creating a pickle file for the Multinomial Naive Bayes model
filename = 'spam-email-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))