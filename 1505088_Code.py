# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 00:26:29 2020

@author: User
"""

import numpy as np
import pandas as pd
import numpy as np
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re


with open('Data/topics.txt') as f:
    Topics = f.readlines()
content = [x.strip() for x in Topics]


print(content)

#text = "Bangladesh, officially the People's Republic of Bangladesh, is a country in South Asia. It is the eighth-most populous country in the world, with a population exceeding 162 million people. It is not other countries. It is the best place to live in."
#print("===Raw Text:===\n", text)

def Binary_Vector(List1, List2): 
    check = [] 
    for m in List1:
        for n in List2:
            # if there is a match
            if m == n: 
                check.append(1)
            else:
                check.append(0)
                  
    return check

from bs4 import BeautifulSoup as bs
#t = ['Sample']
set_words = set()
all_texts = []
Topic_train = []
for c in content:
    print(c)
    count = 0
    with open("Data/Training/"+ c + '.xml','r',encoding='utf-8') as file:
        content = file.read()
        soup = bs(content , features= "lxml")
        
        for items in soup.findAll("row"):
            if count == 10: break
            text = items["body"]
            #Removing <p> and <a href>
            text = re.sub('<a.+?>.+?</a>', '', text)
            text = re.sub('<img.+?>', '', text)
            text = re.sub('</?p>', '', text)
            
            text = text.lower()
            #print("\n===After Lowercase:===\n", text)
            
            text = re.sub(r'[-+]?\d+', '', text)
            #print("\n===After Removing Numbers:===\n", text)
            
            
            text=text.translate((str.maketrans(' ',' ',string.punctuation)))
            #print("\n===After Removing Punctuations:===\n", text)
            
            
            text = word_tokenize(text)
            #print("\n===After Tokenizing:===\n", text)
            
            
            stop_words = set(stopwords.words('english'))
            text = [word for word in text if not word in stop_words]
            #print("\n===After Stopword Removal:===\n", text)
            
            
            lemmatizer=WordNetLemmatizer()
            text = [lemmatizer.lemmatize(word) for word in text]
            #print("\n===After Lemmatization:===\n", text)
            
            stemmer= PorterStemmer()
            text = [stemmer.stem(word) for word in text]
            #print("\n===After Stemming:===\n", text)
            set_words.update(text)
            all_texts.append(text)
            Topic_train.append(c)
            count = count + 1
        

all_words = list(set_words)

#Create Binary Vector of words
X_train = []
for word in all_texts:
    X = Binary_Vector(all_words, word)
    X_train.append(X)
    
print("\n",all_words)
print("\n\n", set_words)
print("\n\n", X_train)
print("\n\n", Topic_train)