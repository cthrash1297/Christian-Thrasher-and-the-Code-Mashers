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
from torchvision.io import read_image
from torch.utils.data.sampler import SubsetRandomSampler

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

def labels(path):
    imlabel = []
    impaths = []
    i = 0 
    for emotion in os.listdir(path):
        for pic in os.listdir(os.path.join(path,emotion)):
            imlabel.append(i)
            impaths.append((os.path.join(path,emotion,pic)))
        i+=1
    return impaths,imlabel
trainpaths,trainlabel = labels(r"C:\Users\Christian\Desktop\Academics\Neural Networks\archive\train")
testpaths,testlabel = labels(r"C:\Users\Christian\Desktop\Academics\Neural Networks\archive\test")

# print(imagepaths)


#Lets define a function to further separate our train data into a training set and a validation set
#Let's call the ratio... 80-20



directory_train=  r"C:\Users\Christian\Desktop\Academics\Neural Networks\archive\train"
#Loads a customdata set from the image directories, ready to be passed into a dataloader
class CustomDataset(Dataset):
    def __init__(self, labels,impaths,transform):
        self.transform = transform
        self.impaths = impaths
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self,idx):
            image = Image.open(self.impaths[idx])
            image = self.transform(image)
            label= self.labels[idx]
            return image, label




def loaders():
    training = CustomDataset(trainpaths,trainlabel,totrans)
    testing = CustomDataset(testpaths,testlabel,totrans)
    rindex = list(range(len(trainlabel)))
    splitdex = np.floor(.8*len(trainlabel))
    splitdex = int(splitdex)
    print(splitdex,"LOOOOOOOOOOOOOK")
    np.random.shuffle(rindex)
    trainsplit, validsplit = rindex[:splitdex],rindex[splitdex:]
    train_sampler = SubsetRandomSampler(trainsplit)
    valid_sampler = SubsetRandomSampler(validsplit)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=500,
        sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=500,
        sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100)
    return train_loader, valid_loader, test_loader

train_loader, valid_loader, test_loader = loaders()


# #Defining the model. Conv layers for defining and locating features, relus to introduce non-linearity, flatten layers to fix dimesional
# #mismatch in preparation for being fed itno fully connected layers. Outputs a non-normalized 7 dimensional vector

model = nn.Sequential(
    nn.Conv2d(1, 3, 3),
    nn.ReLU(),       
    nn.MaxPool2d(2, 2),       
    nn.Conv2d(3, 6, 3),
    nn.ReLU(),       
    nn.MaxPool2d(2, 2),      
    nn.ReLU(),
    #Change start_dim = 1 for passing batches, right now this is for a single image (start_dim = 0)
    nn.Flatten(start_dim = 1),           
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



#Much of this is heavily inspired by the DLI assignment for the sake of getting a baseline
def get_batch_accuracy(output, y, N):
    #It's OK to not normalize because we are looking for the highest value in the probability
    #vector y output by the model
    pred = output.argmax(dim=1, keepdim=True)
    # print(pred)
    correct = pred.eq(y.view_as(pred)).sum().item()
    # ok = y.view_as_(pred)
    # print(pred.shape,"LOOK HERE TO SEE THE STRUCTURE OF OUR MODEL OUTPUT")
    # print(ok.shape, "LOOK HERE TO SEE THE SHAPE OF OUR COMPARISON")
    return correct / N


#Defining the loss function. Cross entropy loss most appropriate for image classification using an imbalanced set
loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())
#Used to tune the hyperparameters
def validate():
    valid_N = len(valid_loader.dataset)
    loss = 0
    accuracy = 0
    #Evaluation mode, disables certain operatrions for stability and consistency
    model.eval()
    #torch no grad disables gradient calculation to minimize resource consumption
    with torch.no_grad():
        for x,y in train_loader:
            output = model(x)
            loss = loss_function(output, y).item()
            #Valid_N is a ploace holder describing the size of our validation batch
            # print(y, "THIS IS Y")
            
            accuracy += get_batch_accuracy(output, y, valid_N)
    print("validate ", accuracy)
    return accuracy

def training():
    loss = 0
    accuracy = 0
    model.train()
    train_N = len(train_loader.dataset)
    for x,y in train_loader:
        output = model(x)
        #Clears gradients from previous batches to prevent cross contamination
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        # print(output,"OUTPUT")
        # print(y,"Y")
        #calculates a gradient based on the computed loss. Perhaps more accurately cost?
        batch_loss.backward()
        #Adjusts parameters based on the gradient and the defined optimization function
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    print("train ", accuracy)


epochs = 20

for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    training()
    validate()