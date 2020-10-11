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
from collections import Counter
#words=["i", "love", "you", "i", "you", "a", "are", "you", "you", "fine", "green"]
#print(most_common_words)

with open('Data/topics.txt') as f:
    Topics = f.readlines()
content = [x.strip() for x in Topics]

print(content)

def Binary_Vector(List1, List2): 
    check = [0] *len(List1)
    idx = 0
    for m in List1:
        for n in List2:
            # if there is a match
            if m == n:
                check[idx] = 1
                break
        idx += 1
    #print("check\n", check)
    return check

def Numeric_vector(List1, List2):
    check = [0] *len(List1)
    idx = 0
    for m in List1:
        for n in List2:
            # if there is a match
            if m == n:
                check[idx] += 1
                
        idx += 1
    #print("check\n", check)
    return check
    
def euclidean_distance(instance1, instance2):
    distance = 0
    for i in range(len(instance1)):
        distance += (instance1[i] - instance2[i])**2
    return np.sqrt(distance)

def hamming_distance(list1 , list2):
    distance = 0
    for i in range(len(list1)):
        distance += abs(list1[i] - list2[i])
    return distance


from bs4 import BeautifulSoup as bs
#t = ['Sample']
set_words = set()
all_texts_train = []
Topic_train = []
all_texts_validate = []
Topic_validate = []
all_texts_test = []
Topic_test = []
for c in content:
    print(c)
    count = 0
    with open("Data/Training/"+ c + '.xml','r',encoding='utf-8') as file:
        content = file.read()
        soup = bs(content , features= "lxml")
        
        for items in soup.findAll("row"):
            if count == 50: break
            text = items["body"]
            #Removing <tags> and <a href>
            text = re.sub('<a.+?>.+?</a>', '', text)
            text = re.sub('<img.+?>', '', text)
            text = re.sub('</?[a-z]+>', '', text)
            
            text = text.lower()
            #print("\n===After Lowercase:===\n", text)
            
            text = re.sub(r'[-+]?\d+', '', text)
            #print("\n===After Removing Numbers:===\n", text)
                        
            #text=text.translate((str.maketrans(' ',' ',string.punctuation)))
            text = re.sub('['+string.punctuation+']', ' ', text)
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
            
            #Dividing into train and test
            
            if count < 20:
                set_words.update(text)
                all_texts_train.append(text)
                Topic_train.append(c)
            elif count >= 20 and count < 30:
                all_texts_validate.append(text)
                Topic_validate.append(c)
            else:
                all_texts_test.append(text)
                Topic_test.append(c)
            #all_texts.append(text)
            #Topic_train.append(c)
            count = count + 1
        

all_words = list(set_words)

#Create Binary Vector of words
kNN_train = []
for word in all_texts_train:
    X = Binary_Vector(all_words, word)
    kNN_train.append(X)
    #print("\n Train ", X)

kNN_test = []
for word in all_texts_test:
    X = Binary_Vector(all_words, word)
    kNN_test.append(X)
    #print("\n Test ", len(X))

kNN_train_Euclid = []
for word in all_texts_train:
    X = Numeric_vector(all_words, word)
    kNN_train_Euclid.append(X)
    #print("\n Train ", word)
    
kNN_test_Euclid = []
for word in all_texts_test:
    X = Numeric_vector(all_words, word)
    kNN_test_Euclid.append(X)
    #print("\n Test ", word)

            
def prediction_kNN(X_train, Y_train, X_test, n_neighbors):
    allTestNeighbers=[]
    allPredictedOutputs =[]
    #allPredictedOutputs_euclid =[]
    
    #Determine Number of unique class lebels
# =============================================================================
#     uniqueOutputLabels = []
#     for label in Y_train:
#         if label not in uniqueOutputLabels:
#             uniqueOutputLabels.append(label)
#     uniqueOutputCount = len(uniqueOutputLabels)
# =============================================================================
    
    #calculate for earch test data points
    for testInput in X_test:
        allDistances = []
        #allDistances_euclid = []
        
        for trainInput, trainActualOutput in zip(X_train, Y_train):
            distance = hamming_distance( trainInput, testInput)
            #distance_euclid = euclidean_distance( trainInput, testInput)
            #print(len(trainInput))
            #print("\n Test " , len(testInput))
            allDistances.append((trainInput, trainActualOutput, distance))
            #allDistances_euclid.append((trainInput, trainActualOutput, distance_euclid))
        #Sort (in ascending order) the training data points based on distances from the test point     
        allDistances.sort(key=lambda x: x[2])
        #allDistances_euclid.sort(key=lambda x: x[2])
        
        
        #Assuming output labels are from 0 to uniqueOutputCount-1
        #voteCount = np.zeros(uniqueOutputCount)
        neighbors = []
        class_label = []
        #neighbors_euclid = []
        #class_label_euclid = []
        #print("\n AllDistances :", allDistances)
        for n in range(n_neighbors):
            neighbors.append(allDistances[n][0])
            class_label.append(allDistances[n][1])
            #voteCount[class_label] += 1
            #neighbors_euclid.append(allDistances_euclid[n][0])
            #class_label_euclid.append(allDistances_euclid[n][1])
        
        #Determine the Majority Voting (Equal weight considered)
        most_common_words= [word for word, word_count in Counter(class_label).most_common(1)]
        #most_common_words_euclid= [word for word, word_count in Counter(class_label_euclid).most_common(1)]
        #predictedOutput = np.argmax(voteCount)
        
        allTestNeighbers.append(neighbors)
        allPredictedOutputs.append(most_common_words)
        
    return allPredictedOutputs, allTestNeighbers

def performanceEvaluation(X_train, Y_train, X_test, Y_test, n_neighbors):
    totalCount = 0
    correctCount = 0
    
    for testInput, testActualOutput in zip(X_test, Y_test):
        predictedOutput,_ = prediction_kNN(X_train, Y_train, [testInput], n_neighbors)
        #print("\n Words", testActualOutput , predictedOutput[0][0])
        if predictedOutput[0][0] == testActualOutput:
            correctCount += 1
        totalCount += 1    
    
    print("\nHamming: \n n = ",n_neighbors,"Total Correct Count: ",correctCount," Total Wrong Count: ",totalCount-correctCount," Accuracy: ",(correctCount*100)/(totalCount))

def prediction_kNN_Euclid(X_train, Y_train, X_test, n_neighbors):
    allTestNeighbers=[]
    allPredictedOutputs =[]
    
    #calculate for earch test data points
    for testInput in X_test:
        allDistances = []
        
        for trainInput, trainActualOutput in zip(X_train, Y_train):
            distance_euclid = euclidean_distance( trainInput, testInput)
            #print(len(trainInput))
            #print("\n Test " , len(testInput))
            allDistances.append((trainInput, trainActualOutput, distance_euclid))
        #Sort (in ascending order) the training data points based on distances from the test point     
        allDistances.sort(key=lambda x: x[2])
        
        
        #Assuming output labels are from 0 to uniqueOutputCount-1
        #voteCount = np.zeros(uniqueOutputCount)
        neighbors = []
        class_label = []
        for n in range(n_neighbors):
            neighbors.append(allDistances[n][0])
            class_label.append(allDistances[n][1])
            
        
        #Determine the Majority Voting (Equal weight considered)
        most_common_words= [word for word, word_count in Counter(class_label).most_common(1)]
        
        
        allTestNeighbers.append(neighbors)
        allPredictedOutputs.append(most_common_words)
        
    return allPredictedOutputs, allTestNeighbers

def performanceEvaluation_Euclid(X_train, Y_train, X_test, Y_test, n_neighbors):
    totalCount_euclid = 0
    correctCount_euclid = 0
    
    for testInput, testActualOutput in zip(X_test, Y_test):
        predictedOutput_euclid,_ = prediction_kNN_Euclid(X_train, Y_train, [testInput], n_neighbors)
        #print("\n Words", testActualOutput , predictedOutput[0][0])
        
        if predictedOutput_euclid[0][0] == testActualOutput:
            correctCount_euclid += 1
        totalCount_euclid += 1
    
    print("\n Euclidean:\n n = ",n_neighbors,"Total Correct Count: ",correctCount_euclid," Total Wrong Count: ",totalCount_euclid-correctCount_euclid," Accuracy: ",(correctCount_euclid*100)/(totalCount_euclid))

for n in range(6):
    if n%2 == 0 : continue
    #print(n)
    performanceEvaluation(kNN_train, Topic_train, kNN_test , Topic_test, n)
    performanceEvaluation_Euclid(kNN_train_Euclid,Topic_train,kNN_test_Euclid,Topic_test,n)

#print("\n",all_words)
#print("\n\n", set_words)
#print("\n\n", X_train)
#print("\n\n", X_test)
# =============================================================================
# print("\n\n", Topic_train)
# print("\n\n", Topic_test)
# print("\nLength of X_train\n", len(kNN_train))
# print("\nLength of X_test\n", len(kNN_test))
# print("\nLength of Topic_train\n", len(Topic_train))
# =============================================================================
