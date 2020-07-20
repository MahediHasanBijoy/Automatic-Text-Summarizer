# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 02:45:27 2019

@author: Bijoy
"""

import json
import sys
from collections import OrderedDict
import math
from operator import itemgetter
import nltk
from nltk import sent_tokenize, word_tokenize, bigrams
from nltk.corpus import indian
from nltk.tag import tnt
from collections import Counter
import copy
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


def readData(fileName):
    '''
    :params: fileName
    :global: stringLines
    :local: dataFileOpen
    :return: none
    :Used to read data from the file
    '''
    global stringLines
    dataFileOpen = open(fileName, 'r', encoding = "utf-8")
    stringLines = dataFileOpen.read()
    dataFileOpen.close()
    
    
def loadFeatureVector():
    '''
    :global:featureVector,tfIdfWeight,sentPosWeight,bigramWeight,unkWordWeight,cueWordWeight,topicWeight, properNounWeight
    :local: file, featureVector, currentFeature, sentPosWeight, tfIdfWeight, bigramWeight, unkWordWeight, cueWordWeight, topicWeight, properNounWeight
    :return: none
    :Used to load the features from json file
    '''
    global featureVector,tfIdfWeight,sentPosWeight,topicWeight
    file = open('C:/Users/Bijoy/Python Code/My Thesis/featureWeightVector.json', 'r', encoding = "utf-8")
    featureVector = json.load(file)
    file.close()
    for items in featureVector:
        currentFeature = items[1]
        if(currentFeature == "Tf-Idf"):
            tfIdfWeight = items[0]
        elif(currentFeature == "Sentence Position Feature"):
            sentPosWeight = items[0]
        elif(currentFeature == "Topic Feature"):
            topicWeight = items[0]
            
            
def formDataDict():
    
    global stringLines, titleData, wordList, sentenceList, originalSentenceList, bigramCountList
    stringLines = stringLines.replace("Title:", "")
    stringLines = stringLines.replace(".", "")
    stringLines = stringLines.replace("?", " . ")
    stringLines = stringLines.replace("!", " . ")
    stringLines = stringLines.replace("\"", "")
    stringLines = stringLines.replace("।", " . ")
    stringLines = stringLines.replace("\'", "")
    stringLines = stringLines.replace("\n", "")
    stringLines = stringLines.replace(",", "")
    stringLines = stringLines.replace("’", "")
    stringLines = stringLines.replace("\\", "")
    stringLines = stringLines.replace("Text:", ".")
    originalFile = copy.deepcopy(stringLines)
    originalSentenceList = [sentenceList.strip() for sentenceList in originalFile.split(' . ')]
    del originalSentenceList[-1]
    removeStopWord()
    sentenceList = [sentenceList.strip() for sentenceList in stringLines.split(' . ')]
    titleData = sentenceList[0]
    del sentenceList[0]
    del sentenceList[-1]
    for sentence in (range(len(sentenceList))):
        sentenceList[sentence] = word_tokenize(sentenceList[sentence])
    wordList = list(set(word_tokenize(stringLines)))
    
    
def removeStopWord():
    
    global stringLines
    stringLinesWords = stringLines.split()
    stopWordsFile = open("C:/Users/Bijoy/Python Code/My Thesis/stopwords-bn.txt", 'r', encoding = "utf-8")
    stopWordsFileRead = stopWordsFile.readlines()
    for words in stopWordsFileRead:
        words = words.strip()
        for slwords in stringLinesWords:
            if words ==slwords:
                stringLinesWords.remove(slwords)
    stringLines = " ".join(stringLinesWords)+" " 
    
    
    
def generateStemWords(word):
    
    global suffixes
    
    if word.endswith("ই"):
        return word[:-len("ই")]
    elif word.endswith("ও"):
        return word[:-len("ও")]
    elif word.endswith("তো"):
        return word[:-len("তো")]
    elif word.endswith("কে"):
        return word[:-len("কে")]
    elif word.endswith("তে"):
        return word[:-len("তে")]
    elif word.endswith("রা"):
        return word[:-len("রা")]
    elif word.endswith("চ্ছি"):
        return word[:-len("চ্ছি")]
    elif word.endswith("চ্ছিল"):
        return word[:-len("চ্ছিল")]
    elif word.endswith("চ্ছে"):
        return word[:-len("চ্ছে")]
    elif word.endswith("চ্ছিস"):
        return word[:-len("চ্ছিস")]
    elif word.endswith("চ্ছিলেন"):
        return word[:-len("চ্ছিলেন")]
    elif word.endswith("চ্ছ"):
        return word[:-len("চ্ছ")]
    elif word.endswith("েছ"):
        return word[:-len("েছ")]
    elif word.endswith("েছে"):
        return word[:-len("েছে")]
    elif word.endswith("েছেন"):
        return word[:-len("েছেন")]
    elif word.endswith("রছ"):
        return word[:-len("রছ")]+"র"
    elif word.endswith("রব"):
        return word[:-len("রব")]+"র"
    elif word.endswith("েল"):
        return word[:-len("েল")]
    elif word.endswith("েলো"):
        return word[:-len("েলো")]
    elif word.endswith("ওয়া"):
        return word[:-len("ওয়া")]
    elif word.endswith("েয়ে"):
        return word[:-len("েয়ে")]+"া"
    elif word.endswith("য়"):
        return word[:-len("য়")]
    elif word.endswith("য়ে"):
        return word[:-len("য়ে")]
    elif word.endswith("য়েছিল"):
        return word[:-len("য়েছিল")]
    elif word.endswith("েয়েছিল"):
        return word[:-len("েয়েছিল")]+"া"
    elif word.endswith("েছিল"):
        return word[:-len("েছিল")]
    elif word.endswith("েয়েছিলেন"):
        return word[:-len("েয়েছিলেন")]+"া"
    elif word.endswith("েছিলেন"):
        return word[:-len("েছিলেন")]
    elif word.endswith("লেন"):
        return word[:-len("লেন")]
    elif word.endswith("দের"):
        return word[:-len("দের")]
    elif word.endswith("ের"):
        return word[:-len("ের")]
    elif word.endswith("েরটার"):
        return word[:-len("েরটার")]
    elif word.endswith("টার"):
        return word[:-len("টার")]
    elif word.endswith("টি"):
        return word[:-len("টি")]
    elif word.endswith("টির"):
        return word[:-len("টির")]
    elif word.endswith("েরটা"):
        return word[:-len("েরটা")]
    elif word.endswith("টা"):
        return word[:-len("টা")]
    elif word.endswith("টার"):
        return word[:-len("টার")]
    elif word.endswith("েরগুলো"):
        return word[:-len("েরগুলো")]
    elif word.endswith("েরগুলোর"):
        return word[:-len("েরগুলোর")] 
    elif word.endswith("গুলো"):
        return word[:-len("গুলো")]
    elif word.endswith("গুলোর"):
        return word[:-len("গুলোর")]  
    elif word.endswith("ার"):
        return word[:-len("ার")]
    elif word.endswith("েন"):
        return word[:-len("েন")]
    elif word.endswith("বেন"):
        return word[:-len("বেন")]
    elif word.endswith("িস"):
        return word[:-len("িস")]
    elif word.endswith("ছিস"):
        return word[:-len("ছিস")]
    elif word.endswith("ছিলি"):
        return word[:-len("ছিলি")]
    elif word.endswith("ছি"):
        return word[:-len("ছি")]
    elif word.endswith("ছে"):
        return word[:-len("ছে")]
    elif word.endswith("লি"):
        return word[:-len("লি")]
    elif word.endswith("বি"):
        return word[:-len("বি")]
    elif word.endswith("ে"):
        return word[:-len("ে")]
    return word


def stemmingForData(sentenceList):
    
    for sentence in range(len(sentenceList)):
        stringTemp = []

        for words in sentenceList[sentence]:
            temp_word = generateStemWords(words)
            stringTemp.append(temp_word)
        sentenceList[sentence] = stringTemp
        
        
def stemmingForTitle():
    
    global titleList
    titleList = titleData.split(" ")
    stringTemp = []
    for words in titleList:
        temp_word = generateStemWords(words)
        stringTemp.append(temp_word)
        titleList = stringTemp
        
        
def calculateIdf():
    
    global idf
    allWords = []
    for sentence in range(len(sentenceList)):
        allWords.extend(list(set(sentenceList[sentence])))
    idf = Counter(allWords)
    for items in idf:
        idf[items] = math.log(len(sentenceList) / idf[items])
        
        
def calculateFeatures():
    
    global featureProbablity
    for i in range(1,len(originalSentenceList)):
        featureProbablity[originalSentenceList[i]] = {}
    i = 1
    j = len(originalSentenceList)-1
    tfIdf = [0] * len(sentenceList)
    for sentences in sentenceList:
        countTopicFeature = 0
        for words in sentences:
            if words in titleList:
                countTopicFeature +=  1
        if (len(sentences) ==  0):
            sentences.append(" ")
        featureProbablity[originalSentenceList[i]]["topicFeature"] = countTopicFeature / len(sentences)


        tfNumerator = {}
        tfNumerator = Counter(sentences)
        for words in tfNumerator:
            tfNumerator[words] = (tfNumerator[words] / len(sentences) * idf[words])
            tfIdf[i - 1] +=  tfNumerator[words]
        featureProbablity[originalSentenceList[i]]["tfIdf"] = tfIdf[i - 1] / len(sentences)

        if (i <=  j):
            featureProbablity[originalSentenceList[i]]["sentencePositionFeature"] = (j - (len(sentenceList) / 2)) / ((len(sentenceList) / 2))
            featureProbablity[originalSentenceList[j]]["sentencePositionFeature"] = (j - (len(sentenceList) / 2)) / ((len(sentenceList) / 2))
        featureProbablity[originalSentenceList[i]]["class"] = 0
        i +=  1
        j -=  1
        
        
def sentenceRank():
    '''
    :global: rankSentences
    :local: i
    :return: none
    :Used to rank the sentences, based on the weights obtained from various algorithms
    '''
    global rankSentences
    rankSentences = {}
    i = 1
    for sentences in sentenceList:
        sentenceWeight = 0
        sentenceWeight += featureProbablity[originalSentenceList[i]]["topicFeature"] * topicWeight
        sentenceWeight += featureProbablity[originalSentenceList[i]]["tfIdf"] * tfIdfWeight
        sentenceWeight += featureProbablity[originalSentenceList[i]]["sentencePositionFeature"] * sentPosWeight
        rankSentences[i] = sentenceWeight
        i += 1
        



featureProbablity = OrderedDict()       
loadFeatureVector()
readData("C:/Users/Bijoy/Python Code/My Thesis/Dataset1/Documents/Document_"+str(194)+".txt")
formDataDict()
stemmingForData(sentenceList)
stemmingForTitle()
calculateIdf()
calculateFeatures()
sentenceRank()
sorted_x = OrderedDict(sorted(rankSentences.items(), key = itemgetter(1)))
answer = []

for key in sorted_x:
    answer.insert(0,key)
summary = []

for i in range(math.ceil(len(answer)*0.3)):
    summary.append(answer[i])
summary = sorted(summary)
summaryText = []

del originalSentenceList[0]
for i in summary:
   summaryText.append(originalSentenceList[i-1])
print(summaryText)

