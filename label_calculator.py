# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 00:12:28 2019

@author: Bijoy
"""

import json
import sys

def assignClasses():
    '''
    :global: allArticlesWithFeatures
    :local: file, summaryStringLines
    :return none
    :Used to assign classes 1 or 0 depending upon the presence of the sentence in the summary.
    '''
    global allArticlesWithFeatures
    file = open('C:/Users/Bijoy/Python Code/My Thesis/allArticlesWithFeatures.json', 'r', encoding = 'utf-8')
    allArticlesWithFeatures = json.load(file)
    for i in range(1,191):
        file = open("C:/Users/Bijoy/Python Code/My Thesis/Dataset1/Summaries/Document_" + str(i) + "_Summary_1.txt", "r", encoding = "utf-8")
        summaryStringLines = file.read()
        file.close()
        summaryStringLines = summaryStringLines.replace(".", "")
        summaryStringLines = summaryStringLines.replace("?", " . ")
        summaryStringLines = summaryStringLines.replace("!", " . ")
        summaryStringLines = summaryStringLines.replace("\"", "")
        summaryStringLines = summaryStringLines.replace("।", " . ")
        summaryStringLines = summaryStringLines.replace("?", " . ")
        summaryStringLines = summaryStringLines.replace("\'", "")
        summaryStringLines = summaryStringLines.replace("\n", "")
        summaryStringLines = summaryStringLines.replace("\ufeff", "")
        summaryStringLines = summaryStringLines.replace(",", "")
        summaryStringLines = summaryStringLines.replace("’", "")
        summaryStringLines = summaryStringLines.replace("\\", "")
        summaryLines = [sentenceList.strip() for sentenceList in summaryStringLines.split(' . ')]
        del summaryLines[-1]
        print("\nFile Name - " + str(i))
        print("Length of Summary File - " + str(len(summaryLines)))
        print("Sentences in Summary File\n")
        for keys in summaryLines:
            print(keys)
        print("\n\n")
        count = 0
        print("Sentences in Text File")
        for keys in allArticlesWithFeatures[i-1].keys():
            print (keys)
        print ("\n\n")
        for sent in summaryLines:
            if sent in allArticlesWithFeatures[i-1]:
                allArticlesWithFeatures[i-1][sent]["class"] = 1
                count += 1
            else:
                print("Line Not matched - ")
                print("\t" + sent)
                sys.exit()
        print("Number of lines in Summary file that matched - " + str(count))
        

def saveFinalDataset():
    '''
    :global: allArticlesWithFeatures
    :local: file, articleCount
    :return: none
    :Used to the final dataset after the labels are assigned
    '''
    global allArticlesWithFeatures
    file = open('C:/Users/Bijoy/Python Code/My Thesis/finalDataset.csv', 'w+', encoding = 'utf-8')
    file.write("Article Number, Topic Feature, Tf-Idf, Sentence Position Feature, Class\n")
    articleCount = 0
    for articles in allArticlesWithFeatures:
        articleCount +=  1
        print(articleCount)
        for sentenceKey in articles:
            file.write(str(articleCount) + ", " + str(articles[sentenceKey]["topicFeature"]) + ", " + str(articles[sentenceKey]["tfIdf"]) + ", " + str(articles[sentenceKey]["sentencePositionFeature"]) + ", " + str(articles[sentenceKey]["class"]) + "\n")
    file.close()


        
        
allArticlesWithFeatures = list()
assignClasses()
saveFinalDataset()