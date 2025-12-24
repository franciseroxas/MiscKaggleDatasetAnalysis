from datasets import load_dataset
import nltk
import pandas as pd
from bs4 import BeautifulSoup
import unicodedata

#local imports
from contraction import contraction_mapping

def removeSentenceStartingWith(text, removeText = 'CLICK HERE'):
    startIdx = text.find(removeText)
    if(startIdx == -1):
        return text
    
    endIdx = startIdx + text[startIdx: len(text)].find(".") #search for the end of the sentence
    if(endIdx < startIdx): #No period found
        return text[0:startIdx]
        
    return text[0:startIdx] + " " + text[endIdx+1:len(text)] #Plus 1 to remove the period

#function to remove special characters or gramatical errors from the word
def manualPreprocessText(text):
    text = text.replace('...', '... ')
    text = text.replace(' ... ', '... ')
    text = text.replace('…', '... ')
    text = text.replace('@', " ")
    text = text.replace('(CNN)', "(CNN) ")
    text = text.replace(" . ", " ")
    text = text.replace(" bt ", " ") #bold text 
    return text

def removeContractions(text):
    returnText = ""
    for word in text.split(" "):
        replacedWord = contraction_mapping.get(word)
        if(replacedWord is not None):
            word = replacedWord
        returnText = returnText + word + " " 
    return returnText[0:len(returnText)-1] #remove the extra space        

def preprocess_text(data, colNameList = ['article', 'highlights'], strip = True):
    for colName in colNameList:
        for i in range(data.shape[0]):
            soup = BeautifulSoup(data[colName][i], 'lxml')
            #Retrieve raw text from html data
            data.loc[i, colName] = soup.get_text(strip=strip)
            
            #Remove contractions
            data.loc[i, colName] = removeContractions(data.loc[i, colName])
            
            #Remove unicode characters
            data.loc[i, colName] = unicodedata.normalize("NFKD", data.loc[i, colName])
            
            #Manual preprocessing of special characters 
            data.loc[i, colName] = manualPreprocessText(data.loc[i, colName])
            
            #Remove the click here links at the end of the text of highlights
            data.loc[i, colName] = removeSentenceStartingWith(data.loc[i, colName], removeText = 'CLICK HERE')
            
            #Remove the 'scroll down to watch' or 'scroll down for video' sentences
            data.loc[i, colName] = removeSentenceStartingWith(data.loc[i, colName], removeText = "Scroll down ")
            
            print('Progress:', colName, round(i / data.shape[0], 5))
    
    return data

PATH = "dataset/cnn_dailymail"
train_data=pd.read_csv(PATH + "/train.csv",nrows=10000)
train_data = preprocess_text(train_data)
train_data.to_csv(PATH + "/smallTrain.csv", index=False)
del train_data

test_data=pd.read_csv(PATH + "/test.csv",nrows=1000)
test_data = preprocess_text(test_data)
test_data.to_csv(PATH + "/smallTest.csv", index=False)
del test_data

val_data=pd.read_csv(PATH + "/validation.csv",nrows=1000)
val_data = preprocess_text(val_data)
val_data.to_csv(PATH + "/smallVal.csv", index=False)
del val_data

