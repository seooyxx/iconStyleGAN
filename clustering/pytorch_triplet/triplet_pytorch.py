import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.models import resnet50

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

import os
import json
import zipfile
import pickle
import random


def open_pickle():
    with open('../../rawdata/resnet.pickle', "rb") as f:
        data = pickle.load(f)
    return data

train_data= open_pickle()
train_data_path = '../../rawdata/logos'

def get_train_dataset(IMAGE_SIZE=96):
    train_dataset = Dataset_Triplet(train_data,path=train_data_path,train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((IMAGE_SIZE,IMAGE_SIZE))]))
    return train_dataset


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    

class Dataset_Triplet():
    def __init__(self, data, path, train=True, transform=None):
        self.data_dict = data
        self.is_train = train
        self.transform = transform
        self.path = path
        if self.is_train:
            self.images = self.data_dict['Filenames']
            self.labels = self.data_dict['Labels']
            self.index = list(range(len(self.data_dict['Filenames'])))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        anchor_image_name = self.images[item]
        anchor_image_path = '../' + anchor_image_name
        anchor_img = Image.open(anchor_image_path).convert('RGB')
        
        if self.is_train:
            anchor_label = self.labels[item]
            
            positive_list = [i for i in self.index if i != item and self.labels[i] == anchor_label]
            positive_item = random.choice(positive_list)
            positive_image_name = self.images[positive_item]
            positive_image_path = '../' + positive_image_name
            positive_img = Image.open(positive_image_path).convert('RGB')
            
            negative_list = [i for i in self.index if i != item and self.labels[i] != anchor_label]
            negative_item = random.choice(negative_list)
            negative_image_name = self.images[negative_item]
            negative_image_path = '../' + negative_image_name
            negative_img = Image.open(negative_image_path).convert('RGB')
            
            if self.transform is not None:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)
                
        return anchor_img, positive_img, negative_img, anchor_label

    
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1-x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
                            
IMAGE_SIZE = 96
BATCH_SIZE = 64
DEVICE = get_default_device()
LEARNING_RATE = 0.005
EPOCHS = 10

train_dataset = get_train_dataset(IMAGE_SIZE = IMAGE_SIZE)
train_dl = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4,pin_memory=True)

class ResNet_Triplet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Feature_Extractor = resnet50(pretrained=True)
        num_filters = self.Feature_Extractor.fc.in_features
        self.Feature_Extractor.fc = nn.Sequential(
                  nn.Linear(num_filters,512),
                  nn.LeakyReLU(),
                  nn.Linear(512,10))
        self.Triplet_Loss = nn.Sequential(
                  nn.Linear(10,2))
    def forward(self,x):
        features = self.Feature_Extractor(x)
        triplets = self.Triplet_Loss(features)
        return triplets

if __name__ == '__main__':
    model = ResNet_Triplet()
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),lr = LEARNING_RATE)
    criterion = TripletLoss()

    for epoch in tqdm(range(EPOCHS), desc='Epochs'):
        running_loss = []
        for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(tqdm(train_dl, desc='Training', leave=False)):
            
            anchor_img = anchor_img.to(DEVICE)
            positive_img = positive_img.to(DEVICE)
            negative_img = negative_img.to(DEVICE)
            optimizer.zero_grad()
            anchor_out = model(anchor_img)
            positive_out = model(positive_img)
            negative_out = model(negative_img)
            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.cpu().detach().numpy())
            print('Epoch: {}/{} — Loss: {:.4f}'.format(epoch+1, EPOCHS, np.mean(running_loss)))

    torch.save({"model_state_dict": model.state_dict(),
                "optimzier_state_dict": optimizer.state_dict()},
                "trained_model.pth"
                )
