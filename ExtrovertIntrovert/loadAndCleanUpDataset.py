import pandas as pd
import numpy as np
import argparse
import torch

def loadAndCleanUpDataset(fn = 'personality_dataset.csv'):
    df = pd.read_csv(fn)
    df = df[~df.isnull().any(axis=1)] #remove all rows with null entries. 
    return df

#Function for preprocessing the data
def turnDatasetIntoTorchTensor(df):
    #Convert the stage fear column to an integer 
    df.loc[df['Stage_fear'] == 'No', 'Stage_fear'] = 0
    df.loc[df['Stage_fear'] == 'Yes', 'Stage_fear'] = 1

    #Drained_after_socializing
    df.loc[df['Drained_after_socializing'] == 'No', 'Drained_after_socializing'] = 0
    df.loc[df['Drained_after_socializing'] == 'Yes', 'Drained_after_socializing'] = 1
    
    #Personality
    df.loc[df['Personality'] == 'Introvert', 'Personality'] = 0
    df.loc[df['Personality'] == 'Extrovert', 'Personality'] = 1
    
    df = normalizeData(df)
    processedData = np.astype(df.values, float)
    return torch.Tensor(processedData)

#Do 0-1 normalization rather than normalize the columns since some of the columns are binary
def normalizeData(df):
    for key in df.keys():
        df.loc[:, key] = (df[key] - np.min(df[key])) / (np.max(df[key]) - np.min(df[key]))
    return df

def main(args = None):
    parser = argparse.ArgumentParser(description='Simple function to remove the rows with nan from csv and place the reqult into a dataframe.')
    parser.add_argument('--dataset', type = str, default = 'personality_dataset.csv')
    args = parser.parse_args()

    df = loadAndCleanUpDataset(fn = args.dataset)
    turnDatasetIntoTorchTensor(df)
    print(df)
    return

if __name__ == "__main__":
    main() 
