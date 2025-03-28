import torch.nn as nn
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import os
# import Pillow as PIL
from PIL import Image
import torchvision.transforms as transforms
import random
from torchvision import datasets
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available

#Defining desired transform, possible image augmentation
totrans = transforms.Compose([
    #Was having an issue with grayscale channels being 3 for some reason. 
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
#Converts each image to a grayscale tensor contining normalized pixel data and its associated label 0 - 6
#Change the bit in quotation marks to wherever the files live on your machine
train_data = datasets.ImageFolder(root = r"C:\Users\Christian\Desktop\Academics\Neural Networks\archive\train", transform = totrans)

test_data = datasets.ImageFolder(root = r"C:\Users\Christian\Desktop\Academics\Neural Networks\archive\test", transform = totrans)

#Defining the mode. Conv layers for defining and locating features, relus to introduce non-linearity, flatten layers to fix dimesional
#mismatch in preparation for being fed itno fully connected layers. Outputs a non-normalized 7 dimensional vector
model = nn.Sequential(
    nn.Conv2d(1, 3, 3),
    nn.ReLU(),       
    nn.MaxPool2d(2, 2),       
    nn.Conv2d(3, 6, 3),
    nn.ReLU(),       
    nn.MaxPool2d(2, 2),      
    nn.ReLU(),
    #Change start_dim = 1 for passing batches, right now this is for a single image
    nn.Flatten(start_dim = 0),           
    nn.Linear(600, 16),       
    nn.Linear(16, 16),       
    nn.Linear(16, 7)          

)

#Test code meant to feed a single image through the model
# pic = totrans(Image.open(r"C:\Users\Christian\Desktop\Academics\Neural Networks\archive\train\angry\Training_3908.jpg"))
# test = model(pic)
# print(pic.shape)
# print(train_data[0])
# print(test)

train_loader = DataLoader(train_data,1000)
test_loader = DataLoader(test_data,1000)

#Defining the loss function. Cross entropy loss most appropriate for image classification using an imbalanced set
loss_function = nn.CrossEntropyLoss()
def validate():
    loss = 0
    accuracy = 0
    #Evaluation mode, disables certain operatrions for stability and consistency
    model.eval()
    #torch no grad disables gradient calculation to minimize resource consumption
    with torch.no_grad():
        for x,y in train_loader:
            output = model(x)

            loss = loss_function(output, y)
