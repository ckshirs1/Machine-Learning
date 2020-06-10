#!/usr/bin/env python
# coding: utf-8

# In[47]:


import os
import pandas as pd
import math
import numpy as np


# List of the stop words

wordsToBeIgnored = ["I", "we", "he", "she", "it", "they", "to", "cc", "can", "could", "should",
 "would", "may", "might","must", "shall", "will", "ought", "'", ":", "Subject:", "from" , 
"a","about","above","after","again", "against", "all", "am", "an", "and", "any", "are", "aren't", 
"as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
"can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", 
"down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
"haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", 
"him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
"isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no",
"nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "-", "/", ".","'" 
"@", ","]


pathForHam = "./train/train/ham"
pathForSpam = "./train/train/spam"

pathForTestHam = "./test/ham"
pathForTestSpam = "./test/spam"

filesInHam = []
filesInSpam = []

testFilesInHam = []
testFilesInSpam = []

dfForBow = pd.DataFrame(columns=['Token', 'SpamWeight', 'HamWeight'])
resultsFile = open("./results.txt", "a") 

resultsFile.write("Output when stop words excluded\n")

#reading from the ham files
for r, d, f in os.walk(pathForHam):
    for file in f:
        if '.txt' in file:
            filesInHam.append(os.path.join(r, file))
   
for r, d, f in os.walk(pathForSpam):
    for file in f:
        if '.txt' in file:
            filesInSpam.append(os.path.join(r, file))

def initBowToZero():
    for i in range(len(dfForBow.index)):
        dfForBow.at[i, "SpamWeight"] = 0
        dfForBow.at[i, "HamWeight"] = 0
        
tokensOfSpamTraining = []        
for file in filesInSpam:
    f = open(file, "r", encoding="latin-1")
    tokens = f.read().split()
    tokensWithoutStopWords = []
    for token in tokens:
        if token not in wordsToBeIgnored:
            tokensWithoutStopWords.append(token)
    tokensOfSpamTraining.append(tokensWithoutStopWords)

tokensOfHamTraining = []    
for file in filesInHam:
    f = open(file, "r")
    tokens = f.read().split()
    tokensWithoutStopWords = []
    for token in tokens:
        if token not in wordsToBeIgnored:
            tokensWithoutStopWords.append(token)
    tokensOfHamTraining.append(tokensWithoutStopWords)

indexForBow = 0

def appendUniqueWords(tokens):
    global indexForBow
    for token in tokens:
        if token not in dfForBow["Token"].tolist():
            dfForBow.at[indexForBow, "Token"] = token
            dfForBow.at[indexForBow, "SpamWeight"] = 0
            dfForBow.at[indexForBow, "HamWeight"] = 0
            #print("appended unique word at:",indexForBow)
            indexForBow = indexForBow + 1
    #print("total unique words are:",len(dfForBow.index))        

#print(dfForBow["Token"].values.tolist())
       
def getX(tokens):
    x = []
    tokensFromDf = dfForBow["Token"].tolist()
    for tokenDf in tokensFromDf:
        if tokenDf in tokens: # if the token from tokensFromDf is found in tokens
            x.append(1)
        else:
            x.append(0)
    return x

def predict(x):
    # here we will have summation of wi * xi
    # first we will calculate summation for ham
    summationForHam = 0
    for i in range(len(dfForBow.index)):
        summationForHam = summationForHam + (dfForBow.at[i, "HamWeight"] * x[i])
    
    # here we will calculate summation for Spam
    summationForSpam = 0
    for i in range(len(dfForBow.index)):
        summationForSpam = summationForSpam + (dfForBow.at[i, "SpamWeight"] * x[i])
    
    print("summationForHam:",summationForHam," summationForSpam:",summationForSpam,"\n")
    
    # 0 for spam and 1 for ham
    if summationForSpam >=summationForHam:
        return 0
    
    else:
        return 1
    
def updateWeightsForHam(x, perceptronPrediction, learningRate):
    for i in range(len(dfForBow.index)):
        delta = x[i]*(1 - perceptronPrediction) * learningRate
        dfForBow.at[i, "HamWeight"] = dfForBow.at[i, "HamWeight"] + delta
            

def updateWeightsForSpam(x, perceptronPrediction, learningRate):
    for i in range(len(dfForBow.index)):
        delta =-x[i]*(0 -perceptronPrediction) * learningRate
        dfForBow.at[i, "SpamWeight"] = dfForBow.at[i, "SpamWeight"] + delta
            

#reading from the spam files
for r, d, f in os.walk(pathForTestSpam):
    for file in f:
        if '.txt' in file:
            testFilesInSpam.append(os.path.join(r, file))
            
            
#reading from the ham files
for r, d, f in os.walk(pathForTestHam):
    for file in f:
        if '.txt' in file:
            testFilesInHam.append(os.path.join(r, file))

tokensForSpamTesting = []
for file in testFilesInSpam:
    f = open(file, "r", encoding="latin-1")
    tokensForSpamTesting.append(f.read().split()) 
    
tokensForHamTesting = []
for file in testFilesInHam:
    f = open(file, "r", encoding="latin-1")
    tokensForHamTesting.append(f.read().split()) 
        
        
def testSpamData():
    resultForSpamDocs = []
    count = 0 
    for tokens in tokensForSpamTesting:
        print("test file in spam number:",count, " out of ", len(tokensForSpamTesting))
#       f = open(file, "r", encoding="latin-1")
#       fileContent = f.read()
#        tokens = fileContent.split()
        x = getX(tokens)
        resultForSpamDocs.append(predict(x))
#       print("perceptronPrediction:",perceptronPrediction)
        count = count + 1
    return getAccuracyForSpamDocs(resultForSpamDocs)
    
    
            
def testHamData():
    resultForHamDocs = []
    count = 0 
    for tokens in tokensForHamTesting:
        print("test file in ham number:",count, " out of ", len(tokensForHamTesting))
#       f = open(file, "r", encoding="latin-1")
#       fileContent = f.read()
#        tokens = fileContent.split()
        x = getX(tokens)
        resultForHamDocs.append(predict(x))
#       print("perceptronPrediction:",perceptronPrediction)
        count = count + 1
    return getAccuracyForHamDocs(resultForHamDocs)


def getAccuracyForHamDocs(resultForHamDocs):
    print("result for ham docs:",resultForHamDocs)
    count = 0 # count for spam
    for i in resultForHamDocs:
        if i == 1:
            count = count + 1
    print("Count in ham:",count)
    print("count of total in ham",len(resultForHamDocs))
    return (count)/(len(resultForHamDocs))

def getAccuracyForSpamDocs(resultForSpamDocs):
    print("result for spam docs:",resultForSpamDocs)
    count = 0 # count for spam
    for i in resultForSpamDocs:
        if i == 0:
            count = count + 1
    print("Count in spam:",count)
    print("count of total in spam",len(resultForSpamDocs))
    return (count)/(len(resultForSpamDocs))	

                        
def trainAndTestData(learningRate, iterations):
    #nitBowToZero()
    for i in range(iterations):
        count = 1
        for tokens in tokensOfHamTraining:
            if count <= 1000:
                print("For Hams, Iteration no: ",i,", file number:", count, " out of ",len(filesInHam))
    #           fileContent = open(file, "r").read()
    #            tokens = fileContent.split()
                if i == 0:
                    appendUniqueWords(tokens)
                x = getX(tokens)
                perceptronPrediction = predict(x)
                updateWeightsForHam(x, perceptronPrediction, learningRate)
            count = count + 1
        #print(dfForBow)
        #hamAccuracy = testHamData()
        #spamAccuracy = testSpamData()
        #string = "\nFor Ham data, number of iterations:"+str(i)+" and learning rate:"+str(learningRate)+", Accuracy is on spam test data: "+str(spamAccuracy)+", Accuracy is on ham test data: "+str(hamAccuracy)+"\nAccuracy is:"+str((spamAccuracy+hamAccuracy)/2)+"\n"
        #print(string)
        #resultsFile.write(string)
        
        count = 1
        for tokens in tokensOfSpamTraining:
            if count <= 1000:
                print("For Spams, Iteration no: ",i,", file number:",count," out of ",len(filesInSpam))
              # f = open(file, "r", encoding="latin-1")
              # fileContent = f.read()
              #  tokens = fileContent.split()
                if i == 0:
                    appendUniqueWords(tokens)
                x = getX(tokens)
                perceptronPrediction = predict(x)
                updateWeightsForSpam(x, perceptronPrediction, learningRate)
            count = count + 1
        print(dfForBow)
        spamAccuracy = testSpamData()
        hamAccuracy = testHamData()
        string = "\nNumber of iterations:"+str(i)+" and learning rate:"+str(learningRate)+" Accuracy is on spam test data: "+str(spamAccuracy)+" Accuracy is on ham test data: "+str(hamAccuracy)+"\nAccuracy is:"+str((spamAccuracy+hamAccuracy)/2)+"\n"
        print(string)
        resultsFile.write(string)
        

learningRates = [0.3, 0.7 ,1]


for l in range(len(learningRates)):
    trainAndTestData(learningRates[l], 8)
    initBowToZero()
resultsFile.close()



