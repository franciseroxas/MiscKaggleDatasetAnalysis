from datasets import load_dataset
import nltk
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np

def helperCleanText(text):
    text = text.lower()
    
    replaceChrList = ["'", '"', "(", ")", "!", "?", "...", "{", "}", "~", ';', ':', '--', " . ", " - ", '|']
    for _substring in replaceChrList:
        text = text.replace(_substring, "")
    return text

PATH = "dataset/cnn_dailymail"
data_files = {"train": "smallTrain.csv", "test": "smallTest.csv", "val": "smallVal.csv"}
dataset = load_dataset(PATH, data_files=data_files)

wordDict = {}
averageLenList = []
for key in data_files.keys():
    articlesList = dataset[key]['article']
    averageLen = 0
    
    for i in range(len(articlesList)):
        currArticle = helperCleanText(articlesList[i])
        splitArr = currArticle.split(" ")
        averageLen = averageLen + len(splitArr)
        
        for j in range(len(splitArr)):
            word = splitArr[j]
            if(wordDict.get(word) is None):
                wordDict[word] = 1
            else:
                wordDict[word] = wordDict[word] + 1
        print(key, i)
    print("---------------")
    
    averageLenList.append(averageLen / len(articlesList))

wordDict = dict(sorted(wordDict.items(), key=lambda item: item[1]))
print(wordDict, averageLenList)

import pdb
pdb.set_trace()
