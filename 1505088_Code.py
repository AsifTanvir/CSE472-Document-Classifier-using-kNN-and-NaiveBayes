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
from operator import add
from functools import reduce
#words=["i", "love", "you", "i", "you", "a", "are", "you", "you", "fine", "green"]
#print(most_common_words)

with open('Data/topics.txt') as f:
    Topics = f.readlines()
content = [x.strip() for x in Topics]

print(content)

def get_Norm(arr):
    n = sum(map(lambda x:x*x,arr))
    m = np.sqrt(n)
    return m

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
    
def IDF_CW_vector(list1 , list2):
    check = [0] * len(list1)
    idx = 0
    
    for m in list1:
        for n in list2:
            if m in n:
                check[idx] += 1
        idx += 1
    
    return check

def Calc_TF_IDF(list1 , list2 , D, C_W):
    check = [0] * len(list1)
    idx =0
    alpha = 0
    beta = 0
    N_dw = Numeric_vector(list1, list2)
    W_d = len(list2)
    TF = [x / W_d for x in N_dw]
    for i in list1:
        if D == C_W[idx]:
            alpha = 1
            beta = 2
        x = (D + alpha) / (C_W[idx] + beta)
        idf = np.log(x)
        check[idx] = TF[idx] * idf
        idx += 1
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

def TF_IDF_distance(list1, list2):
    norm1 = get_Norm(list1)
    norm2 = get_Norm(list2)
    distance = 0
    prod = np.dot(list1,list2)
    distance = (prod) / (norm1 * norm2)
    return distance

# =============================================================================
# #NB starts here
# =============================================================================
def Prob_W_C(Topic , Test, alpha, V):
    #check = [0] *len(Topic)
    tot_prob = 0
    Nc = len(Topic)
    for word in Test:
        Nw_C = Topic.count(word)
        V_alpha = alpha * V
        Pw_C = np.log((Nw_C + alpha) / (Nc + V_alpha))
        tot_prob += Pw_C
    return tot_prob

from bs4 import BeautifulSoup as bs
#t = ['Sample']
set_words = set()
all_texts_train = []
Topic_train = []
all_texts_validate = []
Topic_validate = []
all_texts_test = []
Topic_test = []
Words_Topic_NB = []
total_train_documents = 0
for c in content:
    print(c)
    count = 0
    with open("Data/Training/"+ c + '.xml','r',encoding='utf-8') as file:
        content = file.read()
        soup = bs(content , features= "lxml")
        
        for items in soup.findAll("row"):
            if count == 50: break
            text = items["body"]
            if len(text) == 0: continue
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
                total_train_documents += 1
            elif count >= 20 and count < 30:
                all_texts_validate.append(text)
                Topic_validate.append(c)
            else:
                all_texts_test.append(text)
                Topic_test.append(c)
            #all_texts.append(text)
            #Topic_train.append(c)
            count = count + 1
        #For NB
        #uniq_wrd = list(set_words)
        wrd = reduce(add, all_texts_train)
        Words_Topic_NB.append((wrd, c))

all_words = list(set_words)
#for l in Words_Topic_NB:
#print("\n NB:", len(Words_Topic_NB))

#Create Binary Vector of words
# =============================================================================
# kNN_train = []
# for word in all_texts_train:
#     if len(word) == 0: continue
#     X = Binary_Vector(all_words, word)
#     kNN_train.append(X)
#     #print("\n Train ", all_words , word, X)
# 
# kNN_test = []
# for word in all_texts_test:
#     if len(word) == 0: continue
#     X = Binary_Vector(all_words, word)
#     kNN_test.append(X)
#     #print("\n Test ", len(X))
# 
# kNN_train_Euclid = []
# for word in all_texts_train:
#     if len(word) == 0: continue
#     X = Numeric_vector(all_words, word)
#     kNN_train_Euclid.append(X)
#     #print("\n Train ", word)
#     
# kNN_test_Euclid = []
# for word in all_texts_test:
#     if len(word) == 0: continue
#     X = Numeric_vector(all_words, word)
#     kNN_test_Euclid.append(X)
#     #print("\n Test ", word)
# 
# C_w = IDF_CW_vector(all_words,all_texts_train)
# #print("\n Vector: ", C_w, len(C_w))
# kNN_train_TF_IDF = []
# for word in all_texts_train:
#     if len(word) == 0: continue
#     X = Calc_TF_IDF(all_words, word, total_train_documents , C_w)
#     kNN_train_TF_IDF.append(X)
#     #print("\n Train ", len(X))
# 
# kNN_test_TF_IDF = []
# for word in all_texts_test:
#     if len(word) == 0: continue
#     #print("\n Test ", word,len(word))
#     X = Calc_TF_IDF(all_words, word, total_train_documents , C_w)
#     kNN_test_TF_IDF.append(X)     
# =============================================================================
    
 
def prediction_kNN(X_train, Y_train, X_test, n_neighbors , Type):
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
        #print("\n n", n)
        allDistances = []
        #allDistances_euclid = []
        
        for trainInput, trainActualOutput in zip(X_train, Y_train):
            if Type == "Hamming" :
                distance = hamming_distance( trainInput, testInput)
            elif Type == "Euclidean":
                distance = euclidean_distance( trainInput, testInput)
            else:
                distance = TF_IDF_distance(trainInput, testInput)
            
            #print(trainInput)
            #print("\n Test " , distance)
            allDistances.append((trainInput, trainActualOutput, distance))
            #allDistances_euclid.append((trainInput, trainActualOutput, distance_euclid))
        #Sort (in ascending order) the training data points based on distances from the test point     
        if Type == "TF-IDF":
            allDistances.sort(key=lambda x: x[2], reverse=True)
        else:
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

def performanceEvaluation(X_train, Y_train, X_test, Y_test, n_neighbors, Type):
    totalCount = 0
    correctCount = 0
    
    for testInput, testActualOutput in zip(X_test, Y_test):
        predictedOutput,_ = prediction_kNN(X_train, Y_train, [testInput], n_neighbors, Type)
        #print("\n Words", [testInput] , X_train)
        if predictedOutput[0][0] == testActualOutput:
            correctCount += 1
        totalCount += 1    
    if Type == "Hamming":
        print("\nHamming: \n n = ",n_neighbors,"Total Correct Count: ",correctCount," Total Wrong Count: ",totalCount-correctCount," Accuracy: ",(correctCount*100)/(totalCount))
    elif Type == "Euclidean":
        print("\nEuclidean: \n n = ",n_neighbors,"Total Correct Count: ",correctCount," Total Wrong Count: ",totalCount-correctCount," Accuracy: ",(correctCount*100)/(totalCount))
    else:
        print("\nTF-IDF: \n n = ",n_neighbors,"Total Correct Count: ",correctCount," Total Wrong Count: ",totalCount-correctCount," Accuracy: ",(correctCount*100)/(totalCount))



# =============================================================================
# def prediction_kNN_Euclid(X_train, Y_train, X_test, n_neighbors):
#     allTestNeighbers=[]
#     allPredictedOutputs =[]
#     
#     #calculate for earch test data points
#     for testInput in X_test:
#         allDistances = []
#         
#         for trainInput, trainActualOutput in zip(X_train, Y_train):
#             distance_euclid = euclidean_distance( trainInput, testInput)
#             #print(len(trainInput))
#             #print("\n Test " , len(testInput))
#             allDistances.append((trainInput, trainActualOutput, distance_euclid))
#         #Sort (in ascending order) the training data points based on distances from the test point     
#         allDistances.sort(key=lambda x: x[2])
#         
#         
#         #Assuming output labels are from 0 to uniqueOutputCount-1
#         #voteCount = np.zeros(uniqueOutputCount)
#         neighbors = []
#         class_label = []
#         for n in range(n_neighbors):
#             neighbors.append(allDistances[n][0])
#             class_label.append(allDistances[n][1])
#             
#         
#         #Determine the Majority Voting (Equal weight considered)
#         most_common_words= [word for word, word_count in Counter(class_label).most_common(1)]
#         
#         
#         allTestNeighbers.append(neighbors)
#         allPredictedOutputs.append(most_common_words)
#         
#     return allPredictedOutputs, allTestNeighbers
# 
# def performanceEvaluation_Euclid(X_train, Y_train, X_test, Y_test, n_neighbors):
#     totalCount_euclid = 0
#     correctCount_euclid = 0
#     
#     for testInput, testActualOutput in zip(X_test, Y_test):
#         predictedOutput_euclid,_ = prediction_kNN_Euclid(X_train, Y_train, [testInput], n_neighbors)
#         #print("\n Words", testActualOutput , predictedOutput[0][0])
#         
#         if predictedOutput_euclid[0][0] == testActualOutput:
#             correctCount_euclid += 1
#         totalCount_euclid += 1
#     
#     print("\n Euclidean:\n n = ",n_neighbors,"Total Correct Count: ",correctCount_euclid," Total Wrong Count: ",totalCount_euclid-correctCount_euclid," Accuracy: ",(correctCount_euclid*100)/(totalCount_euclid))
# 
# =============================================================================

for n in range(6):
    if n%2 == 0 : continue
    #print(n)
    #performanceEvaluation(kNN_train, Topic_train, kNN_test , Topic_test, n, "Hamming")
    #performanceEvaluation(kNN_train_Euclid,Topic_train,kNN_test_Euclid,Topic_test,n, "Euclidean")
    #performanceEvaluation(kNN_train_TF_IDF, Topic_train, kNN_test_TF_IDF, Topic_test, n, "TF-IDF")

def prediction_NB(X_test, alpha, V):
    prob = 0
    allProbabilities = []
    for l in Words_Topic_NB:
        prob = Prob_W_C(l[0], X_test, alpha, V)
        #print("\n", prob)
        allProbabilities.append((prob, l[1]))
    allProbabilities.sort(key=lambda x: x[0], reverse=True)
    #print("\n",  allProbabilities)
    return allProbabilities[0][1]

def PerformanceNB(X_test, Y_test, alpha, V):
    totalCount = 0
    correctCount = 0
    
    for testInput, testActualOutput in zip(X_test, Y_test):
        predictedOutput = prediction_NB(testInput, alpha, V)
        #print("\n Words", predictedOutput , testActualOutput)
        if predictedOutput == testActualOutput:
            correctCount += 1
        totalCount += 1
    print("\n Naive Bayes:\n alpha = ",alpha,"Total Correct Count: ",correctCount," Total Wrong Count: ",totalCount-correctCount," Accuracy: ",(correctCount*100)/(totalCount))

V = len(all_words)
alpha = np.arange(0.1, 1.1, 0.1)
for alp in alpha:
    PerformanceNB(all_texts_validate, Topic_validate, alp, V)

# =============================================================================
# z = Prob_W_C(Words_Topic_NB[0][0], all_texts_train[10], 1, len(all_texts_train[0]))
# z1 = Prob_W_C(Words_Topic_NB[1][0], all_texts_train[10], 1, len(all_texts_train[0]))
# z2 = Prob_W_C(Words_Topic_NB[2][0], all_texts_train[10], 1, len(all_texts_train[0]))
# print(z, z1, z2)
# =============================================================================
#ki ek obostha
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
