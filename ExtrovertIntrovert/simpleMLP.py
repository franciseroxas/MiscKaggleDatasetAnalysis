import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#For model layers only
class simpleMLP(nn.Module):
    #Leaving the feature size variable so that I can test ablation of features if need be
    def __init__(self, inputFeatureSize):
        super(simpleMLP, self).__init__()
        #Inplace to save on resources
        self.mlpForModel = nn.Sequential(nn.Linear(inputFeatureSize, 32),
                                         nn.ReLU(inplace = True),
                                         nn.LayerNorm(32),
                                         nn.Linear(32, 32),
                                         nn.ReLU(inplace = True),
                                         nn.LayerNorm(32),
                                         nn.Linear(32, 32),
                                         nn.ReLU(inplace = True),
                                         nn.LayerNorm(32))

        self.linear1 = nn.Linear(32, 32)
        self.relu1 = nn.ReLU(inplace = True)
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.linear2 = nn.Linear(32, 32)
        self.relu2 = nn.ReLU(inplace = True)
        self.batchnorm2 = nn.BatchNorm1d(32)

        self.linearFinal = nn.Linear(32, 1)
        #Layer for probabilty
        self.softmaxFinal = nn.Softmax(dim=0)

    def forward(self, inputFeatures):

        x = self.mlpForModel(inputFeatures)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.batchnorm1(x)

        x = self.linear2(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)

        x = self.linearFinal(x)
        yPred = x.squeeze()#self.softmaxFinal(x.squeeze())
        return yPred

#Test to see if the code works
if __name__ == "__main__":
    simpleMlp = simpleMLP(3)
    print(simpleMlp)
