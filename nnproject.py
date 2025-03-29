import torch.nn as nn
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms
import random as random
from torchvision import datasets
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available

#Defining desired transform, possible image augmentation
totrans = transforms.Compose([
    #Was having an issue with grayscale channels being 3 for some reason.
    #Reducing channel number should reduce computational load  
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
#Converts each image to a grayscale tensor contining normalized pixel data and its associated label 0 - 6
#Change the bit in quotation marks to wherever the files live on your machine
train_data = datasets.ImageFolder(root = r"C:\Users\Christian\Desktop\Academics\Neural Networks\archive\train", transform = totrans)

test_data = datasets.ImageFolder(root = r"C:\Users\Christian\Desktop\Academics\Neural Networks\archive\test", transform = totrans)

#Lets define a function to further separate our train data into a training set and a validation set
#Let's call the ratio... 80-20



directory_train=  r"C:\Users\Christian\Desktop\Academics\Neural Networks\archive\train"

def datasamples():
    #function made to create a tuple containing a list of tensor data for each image
    #and a list containing all of the label values
  
    def split():
        i = 0
        x_train = []
        y_train = []
        x_valid = []
        y_valid = []
        for emotion in os.listdir(directory_train):
            for pic in os.listdir(os.path.join(directory_train,emotion)):
                im = Image.open(os.path.join(directory_train,emotion,pic))
                im = totrans(im).to(device)
                # y= torch.tensor(i).to(device)
                rnum = random.random()
                #Hopefully, 80 percent of images are sent to the training set and 20 percent are sent to the valid set
                if rnum <= .8:
                    x_train.append(im)
                    y_train.append(i)
                else:
                    x_valid.append(im)
                    y_valid.append(i)
            i+=1
        y_train = torch.tensor(y_train).to(device)
        y_valid = torch.tensor(y_valid).to(device)
        return x_train, y_train, x_valid,y_valid
    return split()
tensor_train,label_train,tensor_valid,label_valid = datasamples()
# print(tensor_train)
# print(label_train)
train_loader = DataLoader((tensor_train, label_train), batch_size = 1000, shuffle = True)
train_N = len(train_loader.dataset)
valid_loader = DataLoader((tensor_valid,label_valid),batch_size = 1000)
valid_N = len(valid_loader.dataset)
print("train_N", train_N)
print("valid_N", valid_N)
   

# batch = next(iter(train_loader))     

# print(train_loader)
# print(valid_loader)
# print(batch)


# # #Defining the model. Conv layers for defining and locating features, relus to introduce non-linearity, flatten layers to fix dimesional
# # #mismatch in preparation for being fed itno fully connected layers. Outputs a non-normalized 7 dimensional vector

# model = nn.Sequential(
#     nn.Conv2d(1, 3, 3),
#     nn.ReLU(),       
#     nn.MaxPool2d(2, 2),       
#     nn.Conv2d(3, 6, 3),
#     nn.ReLU(),       
#     nn.MaxPool2d(2, 2),      
#     nn.ReLU(),
#     #Change start_dim = 1 for passing batches, right now this is for a single image (start_dim = 0)
#     nn.Flatten(start_dim = 1),           
#     nn.Linear(600, 16),       
#     nn.Linear(16, 16),       
#     nn.Linear(16, 7)          

# )

# #Test code meant to feed a single image through the model
# # pic = totrans(Image.open(r"C:\Users\Christian\Desktop\Academics\Neural Networks\archive\train\angry\Training_3908.jpg"))
# # test = model(pic)
# # print(pic.shape)
# # print(train_data[0])
# # print(test)

# train_loader = DataLoader(train_data,1000)
# test_loader = DataLoader(test_data,1000)



# #Much of this is heavily inspired by the DLI assignment for the sake of getting a baseline
# def get_batch_accuracy(output, y, N):
#     #It's OK to not normalize because we are looking for the highest value in the probability
#     #vector y output by the model
#     pred = output.argmax(dim=1, keepdim=True)
#     correct = pred.eq(y.view_as(pred)).sum().item()
#     print(correct/N)
#     return correct / N


# #Defining the loss function. Cross entropy loss most appropriate for image classification using an imbalanced set
# loss_function = nn.CrossEntropyLoss()
# optimizer = Adam(model.parameters())
# #Used to tune the hyperparameters
# def validate():
#     loss = 0
#     accuracy = 0
#     #Evaluation mode, disables certain operatrions for stability and consistency
#     model.eval()
#     #torch no grad disables gradient calculation to minimize resource consumption
#     with torch.no_grad():
#         for x,y in train_loader:
#             output = model(x)

#             loss = loss_function(output, y)
#             #Valid_N is a ploace holder describing the size of our validation batch
#             accuracy += get_batch_accuracy(output, y, valid_N)
#     print("validate ", accuracy)
#     return accuracy

# def training():
#     loss = 0
#     accuracy = 0

#     model.train()
#     for x,y in train_loader:
#         output = model(x)
#         #Clears gradients from previous batches to prevent cross contamination
#         optimizer.zero_grad()
#         batch_loss = loss_function(output, y)
#         #calculates a gradient based on the computed loss. Perhaps more accurately cost?
#         batch_loss.backward()
#         #Adjusts parameters based on the gradient and the defined optimization function
#         optimizer.step()

#         loss += batch_loss.item()
#         accuracy += get_batch_accuracy(output, y, train_N)
#     print("train ", accuracy)


# epochs = 20

# for epoch in range(epochs):
#     print('Epoch: {}'.format(epoch))
#     training()
#     validate()