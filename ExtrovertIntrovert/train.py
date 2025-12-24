#Basic imports
import numpy as np
import torch
import torch.nn as nn
import argparse
import time

#Nested Imports
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

#Functions / Classes that I made
from simpleMLP import simpleMLP
from loadAndCleanUpDataset import loadAndCleanUpDataset, turnDatasetIntoTorchTensor
from extroIntroDataset import extroIntroDataset

def train(dataset, randState = 42, test_size = 0.3, nEpochs = 100, batch_size = 32, learningRate = 0.0001):
    #Get a device to train the model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Load a dataframe and tensor that will represent the dataset.
    df = loadAndCleanUpDataset(fn = dataset)
    dataTensor = turnDatasetIntoTorchTensor(df)

    #Split into inputs and outputs
    personalityFeatures = dataTensor[:, 0:7]
    personalityType = dataTensor[:, 7]
    del df, dataTensor #not needed anymore

    #Get indices to access the data
    indices = np.arange(personalityFeatures.shape[0])

    #Train Test Split. Can be manually set for repeatability and easier result analysis / debugging
    X_train, _, y_train, _ = train_test_split(personalityFeatures, personalityType, test_size=test_size, random_state=randState)
    del personalityFeatures, personalityType, _ #Remove unused variables

    #Get the indices of the data so that accessing the corresponding features from the csv file is easier.
    X_train_dataset_indices, _, y_train_dataset_indices, _ = train_test_split(indices, indices, test_size=test_size, random_state=randState)
    del _ #remove unused variables from memory

    assert(np.sum(X_train_dataset_indices - y_train_dataset_indices) <= 1e-15) #Make sure that these are the same

    #Put the dataset into the torch dataset for use in a dataloader
    trainDataset = extroIntroDataset(X_train, y_train)
    del X_train, y_train

    train_dataloader = DataLoader(trainDataset, batch_size = batch_size, shuffle = True)
    
    #Get the model, criterion
    model = simpleMLP(7)
    if(torch.cuda.is_available()):
        model.cuda()

    criterion = nn.BCEWithLogitsLoss() #Sigmoid with binary cross entropy so that sigmoid does not need to be done in the model 
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, betas=(0.9, 0.999), eps=1e-08)

    #Save a list for the loss after time for a plt plot
    lossHistory = []

    #1-0 to Extrovert Introvert
    numericToExIntroTable = {0: "Introvert", 1: "Extrovert"}
    
    #Place Train Loop Here
    for epoch in range(nEpochs):
        startBatchTime = time.time()
        currentLoss = 0

        for batch_ndx, batch_data in enumerate(train_dataloader):
            csvFeaturesToTensorData, csvLabelsAsTensor = batch_data #List of 2. First index is the torch tensor for features. Second is the torch tensor for the labels (extrovert, introvert)

            #Send both to the gpu
            csvFeaturesToTensorData = csvFeaturesToTensorData.to(device)
            csvLabelsAsTensor = csvLabelsAsTensor.to(device)

            #Get the output predictions
            predictedLabel = model(csvFeaturesToTensorData)
            loss = criterion(predictedLabel, csvLabelsAsTensor)


            batchLoss = loss.cpu().detach().numpy() / batch_size
            currentLoss = currentLoss + loss.cpu().detach().numpy()

            print("Epoch", epoch, "| Batch", batch_ndx, "| Batch Loss", batchLoss)

            #Debug: Get a sample to check if it is working
            sampleTrueLabel = int(csvLabelsAsTensor[0].cpu().detach().numpy())

            #Need to convert to a value close to 0-1 with sigmoid
            sampleModelPred = predictedLabel.cpu().detach()
            sampleModelPred = nn.functional.sigmoid(sampleModelPred).numpy()[0]

            print("True Label:", numericToExIntroTable[sampleTrueLabel], " | True Value: ", sampleTrueLabel, " | Pred Label: ", numericToExIntroTable[np.round(sampleModelPred)], " | Pred Label Value: ", sampleModelPred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lossHistory.append(currentLoss)

    #Plot a figure for the loss over epochs
    plt.figure()
    plt.plot(np.arange(len(lossHistory)), lossHistory)
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss per epoch")
    plt.savefig("Extrovert vs Introvert Dataset Loss.png")



    return

def main(args = None):
    parser = argparse.ArgumentParser(description = "Training loop for the kaggle dataset: https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data")
    parser.add_argument('--dataset', type = str, default = "personality_dataset.csv") 
    args = parser.parse_args()
    train(dataset = args.dataset)
    return

if __name__ == "__main__":
    main()

