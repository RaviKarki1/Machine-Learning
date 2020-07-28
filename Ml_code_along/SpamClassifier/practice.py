# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 09:03:28 2020

@author: ravik
"""

#read in data
import pandas as pd

messages = pd.read_csv('smsspamcollection/SMSSpamCollection',
                       sep = '\t',
                       names = ['label', 'message'])

#data preprocessing for text data
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


corpus = []
for i in range(0, len(messages)):
    #select the texts only
    review = re.sub('[^a-zA-Z]',
                      ' ',
                      messages['message'][i])
    
    #change all to lowercase
    review = review.lower()
    #split the words
    review = review.split()
    #remove the stopwords
    review = [ps.stem(word) for word in review if not word  in stopwords.words('english')]
    #join the words to the list
    review = ' '.join(review)
    #append the words to the corpus
    corpus.append(review)


#text documents to a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(corpus).toarray()

#label into binary
y = pd.get_dummies(messages['label'])
y = y.iloc[:, 1].values

#split data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#fit to naiveBayes classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
pred = clf.fit(X_train, y_train)

y_pred = pred.predict(X_test)

#check the accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
