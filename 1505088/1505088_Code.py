# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 00:26:29 2020

@author: User
"""

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
from scipy import spatial
from scipy import stats

with open('Data/topics.txt') as f:
    Topics = f.readlines()
all_topics = [x.strip() for x in Topics]

print(all_topics)

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
for c in all_topics:
    print(c)
    count = 0
    train_words_per_topic = []
    with open("Data/Training/"+ c + '.xml','r',encoding='utf-8') as file:
        content = file.read()
        soup = bs(content , features= "lxml")
        
        for items in soup.findAll("row"):
            if count == 1200: break
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
            
            if count < 500:
                set_words.update(text)
                all_texts_train.append(text)
                train_words_per_topic.append(text)
                Topic_train.append(c)
                total_train_documents += 1
            elif count >= 500 and count < 700:
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
        wrd = reduce(add, train_words_per_topic)
        Words_Topic_NB.append((wrd, c)) #All words in each topic

del wrd,train_words_per_topic
all_words = list(set_words)


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
 
## Creating a vector that contains the number of documents of each word occur
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
        if C_W[idx] == 0: continue
        if D == C_W[idx]:
            alpha = .1
            beta = .2
        x = (D + alpha) / (C_W[idx] + beta)
        idf = np.log(x)
        check[idx] = TF[idx] * idf
        idx += 1
    return check
    
def euclidean_distance(list1, list2):
    distance = 0
# =============================================================================
#     a = np.array(list1)
#     b = np.array(list2)
# =============================================================================
# =============================================================================
#     for i in range(len(instance1)):
#         distance += (instance1[i] - instance2[i])**2
#     return math.sqrt(distance)
# =============================================================================
    distance = np.linalg.norm(list1-list2 ,ord = 2)
    return distance

def hamming_distance(list1 , list2):
    distance = 0
# =============================================================================
#     a = np.array(list1)
#     b = np.array(list2)
# =============================================================================
# =============================================================================
#     for i in range(len(list1)):
#         distance += abs(list1[i] - list2[i])
# =============================================================================
    distance = np.linalg.norm(list1-list2 ,ord = 1)
    return distance

def TF_IDF_distance(list1, list2):
    norm1 = np.linalg.norm(list1 ,ord = 2)
    norm2 = np.linalg.norm(list2 ,ord = 2)
    distance = 0
    prod = np.dot(list1,list2)
    distance = (prod) / (norm1 * norm2)
    #distance = 1 - spatial.distance.cosine(list1, list2)
    return distance

# =============================================================================
# #NB starts here
# =============================================================================


def countFreq(arr):
    dict_word = {}
    n = len(arr)
    visited = [False for i in range(n)] 
    #count = [0] * n
  
    for i in range(n): 
          
        # Skip this element if already processed 
        if (visited[i] == True): 
            continue
  
        count = 1
        for j in range(i + 1, n, 1): 
            if (arr[i] == arr[j]): 
                visited[j] = True
                count += 1
        
        dict_word[arr[i]] = count
        #print(dict_word)
    return dict_word

def Prob_W_C(Topic , Test, alpha, V , frequency):
    #check = [0] *len(Topic)
    tot_prob = 0
    Nc = len(Topic)
    for word in Test:
        #Nw_C = Topic.count(word)
        if word in frequency:
            Nw_C = frequency[word]
        else:
            Nw_C = 0
        Pw_C = np.log((Nw_C + alpha) / (Nc + (alpha * V)))
        tot_prob += Pw_C
    return tot_prob


C_w = IDF_CW_vector(all_words,all_texts_train)
D = total_train_documents
# =============================================================================
# #Create Vectors of words
# =============================================================================
def Vectorize(Lists, Type):
    kNN = []
    #C_w = IDF_CW_vector(all_words,Lists)
    #D = len(Lists)
    for word in Lists:
        if len(word) == 0: continue
        if Type == "Hamming":
            X = Binary_Vector(all_words, word)
            kNN.append(X)
        elif Type == "Euclidean":
            X = Numeric_vector(all_words, word)
            kNN.append(X)
        else:
            X = Calc_TF_IDF(all_words, word, D , C_w)
            kNN.append(X)
    return kNN


# =============================================================================
# #Prediction of kNN
# =============================================================================
def prediction_kNN(X_train, Y_train, testInput, n_neighbors , Type):
    allPredictedOutputs =[]
    
    #calculate for earch test data points
    allDistances = []
    
    for trainInput, trainActualOutput in zip(X_train, Y_train):
        if Type == "Hamming" :
            distance = hamming_distance( trainInput, testInput)
        elif Type == "Euclidean":
            distance = euclidean_distance( trainInput, testInput)
        else:
            distance = TF_IDF_distance(trainInput, testInput)
        
        allDistances.append((trainInput, trainActualOutput, distance))
    #Sort (in ascending order) the training data points based on distances from the test point     
    if Type == "TF-IDF":
        allDistances.sort(key=lambda x: x[2], reverse=True)
    else:
        allDistances.sort(key=lambda x: x[2])
    
    
    neighbors = []
    class_label = []
    #print("\n AllDistances :", allDistances)
    for n in range(n_neighbors):
        neighbors.append(allDistances[n][0])
        class_label.append(allDistances[n][1])
    
    #Determine the Majority Voting (Equal weight considered)
    most_common_words= [word for word, word_count in Counter(class_label).most_common(1)]
    
    allPredictedOutputs.append(most_common_words)
        
    return allPredictedOutputs

def performanceEvaluation(X_train, Y_train, X_test, Y_test, n_neighbors, Type):
    totalCount = 0
    correctCount = 0
    
    for testInput, testActualOutput in zip(X_test, Y_test):
        predictedOutput = prediction_kNN(X_train, Y_train, testInput, n_neighbors, Type)
        #print("\n Words", predictedOutput[0][0], testActualOutput)
        if predictedOutput[0][0] == testActualOutput:
            correctCount += 1
        totalCount += 1
    if Type == "Hamming":
        print("\nHamming: \n k = ",n_neighbors,"Total Correct Count: ",correctCount," Total Wrong Count: ",totalCount-correctCount," Accuracy: ",(correctCount*100)/(totalCount))
    elif Type == "Euclidean":
        print("\nEuclidean: \n k = ",n_neighbors,"Total Correct Count: ",correctCount," Total Wrong Count: ",totalCount-correctCount," Accuracy: ",(correctCount*100)/(totalCount))
    else:
        print("\nTF-IDF: \n k = ",n_neighbors,"Total Correct Count: ",correctCount," Total Wrong Count: ",totalCount-correctCount," Accuracy: ",(correctCount*100)/(totalCount))
    return (correctCount*100)/(totalCount)


best_result = []
kNN_train = np.array(Vectorize(all_texts_train, "Hamming"))
#kNN_test = np.array(Vectorize(all_texts_test, "Hamming"))
kNN_validate = np.array(Vectorize(all_texts_validate, "Hamming"))
kNN_train_Euclid = np.array(Vectorize(all_texts_train, "Euclidean"))
#kNN_test_Euclid = np.array(Vectorize(all_texts_test, "Euclidean"))
kNN_validate_Euclid = np.array(Vectorize(all_texts_validate, "Euclidean"))
kNN_train_TF_IDF = np.array(Vectorize(all_texts_train, "TF-IDF"))
#kNN_test_TF_IDF = np.array(Vectorize(all_texts_test, "TF-IDF"))
kNN_validate_TF_IDF = np.array(Vectorize(all_texts_validate, "TF-IDF"))
for n in range(1,6,2):
    print(n)
    best_ham = performanceEvaluation(kNN_train, Topic_train, kNN_validate , Topic_validate, n, "Hamming")
    best_result.append((best_ham , n , "Hamming"))
    best_euclid = performanceEvaluation(kNN_train_Euclid,Topic_train,kNN_validate_Euclid,Topic_validate,n, "Euclidean")
    best_result.append((best_euclid , n , "Euclidean"))
    best_TF = performanceEvaluation(kNN_train_TF_IDF, Topic_train, kNN_validate_TF_IDF, Topic_validate, n, "TF-IDF")
    best_result.append((best_TF , n , "TF-IDF"))
    
best_result.sort(key=lambda x: x[0], reverse = True)
best_k = best_result[0][1]
best_type = best_result[0][2]
print("\n Best kNN : " , best_k , best_type)
    
# =============================================================================
# Generating array for all words frequency topic wise..
# Prediction for Naive Bayes
# =============================================================================
Frequency_of_words = []
for l in Words_Topic_NB:
    Frequency_of_words.append(countFreq(l[0]))

def prediction_NB(X_test, alpha, V):
    prob = 0
    allProbabilities = []
    idx = 0
    for l in Words_Topic_NB:
        prob = Prob_W_C(l[0], X_test, alpha, V , Frequency_of_words[idx])
        allProbabilities.append((prob, l[1]))
        idx += 1
    allProbabilities.sort(key=lambda x: x[0], reverse=True)
    return allProbabilities[0][1]

def PerformanceNB(X_test, Y_test, alpha, V):
    totalCount = 0
    correctCount = 0
    
    for testInput, testActualOutput in zip(X_test, Y_test):
        predictedOutput = prediction_NB(testInput, alpha, V)
        if predictedOutput == testActualOutput:
            correctCount += 1
        totalCount += 1
    print("\n Naive Bayes:\n alpha = ",alpha,"Total Correct Count: ",correctCount," Total Wrong Count: ",totalCount-correctCount," Accuracy: ",(correctCount*100)/(totalCount))
    return (correctCount*100)/(totalCount)

best_accuracy = []
alpha = np.arange(0.1, 1.1, 0.1)
V = len(all_words)
for alp in alpha:
    accuracy = PerformanceNB(all_texts_validate, Topic_validate, alp, V)
    best_accuracy.append(accuracy)
idx = np.argmax(best_accuracy)
best_alpha = alpha[idx]
print("\n Best Alpha: ", best_alpha)

# =============================================================================
# best_type = "TF-IDF"
# best_k = 5
# best_alpha = 0.1
# =============================================================================
# =============================================================================
#  All 50 Iterations.....
# =============================================================================
accuracy_kNN = []
accuracy_NB = []
train_kNN = Vectorize(all_texts_train, best_type)
test_kNN = Vectorize(all_texts_test, best_type)
V = len(all_words)
print(len(all_topics))
for i in range(50):
    train = []
    test = []
    
    train_iter = []
    topic_train_iter = []
    test_iter = []
    topic_test_iter = []
    for c in range(len(all_topics)):
        a = []
        slic = (c*500) + (i*10)
        a = train_kNN[slic: slic+10]
        train = train + a
        a = test_kNN[slic: slic+10]
        test = test + a
        
        a = all_texts_test[slic : slic+10]
        test_iter = test_iter + a
        a = Topic_train[slic : slic+10]
        topic_train_iter = topic_train_iter + a
        a = Topic_test[slic : slic+10]
        topic_test_iter = topic_test_iter + a
    #print("\n data: ", train_iter, test_iter , topic_train_iter , topic_test_iter)
    train_k = np.array(train)
    test_k = np.array(test)
    accuracy_kNN.append(performanceEvaluation(train_k, topic_train_iter, test_k, topic_test_iter, best_k, best_type))
    accuracy_NB.append(PerformanceNB(test_iter, topic_test_iter, best_alpha, V))
    print("\n Iteration: " , i , "\n Accuracy kNN: ", accuracy_kNN[i] , " Accuracy NB: " , accuracy_NB[i])


ttest = stats.ttest_rel(accuracy_kNN,accuracy_NB)
stat = ttest[0]
p_value = ttest[1]
Significance = [0.005, 0.01 , 0.05]
for s in Significance:
    if p_value < s:
        if stat < 0:
            print("\nNaive Bayes is better than kNN. P_value: ", p_value)
        else:
            print("\nkNN is better than Naive Bayes. P_value: ", p_value)
    else:
        if stat < 0:
            print("\nNaive Bayes is probably better than kNN. P_value: ", p_value)
        else:
            print("\nkNN is probably better than Naive Bayes. P_value: ", p_value)

