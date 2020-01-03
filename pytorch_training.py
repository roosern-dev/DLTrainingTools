import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
import time
import os
import copy
from torchvision.models.mobilenet import MobileNetV2

torch.cuda.current_device()     #initialize the cuda device to prevent error during lazy init  

imagePathTrain = r'E:\Estek-AIProject\INTEL-SOOP\RaspberryPiFolder\Images\Training\combinedForTraining'
imagePathValid = r'E:\Estek-AIProject\INTEL-SOOP\RaspberryPiFolder\Images\Training\ValidationSet'
modelSavePath = r'E:\Estek-AIProject\INTEL-SOOP\PyTorch\Model1\model1.pt'


data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
epochs = 10
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batchSize = 14

if __name__ == '__main__':
    trainImages = datasets.ImageFolder(root = imagePathTrain, transform = data_transform)
    validImages = datasets.ImageFolder(root = imagePathValid, transform = data_transform)
    train_data =  data.DataLoader(trainImages, batch_size = batchSize, shuffle = False, num_workers = 1)
    valid_data = data.DataLoader(validImages, batch_size=batchSize, shuffle= False, num_workers=1)

    '''
    model = models.mobilenet_v2(pretrained = True)
    num_ftrs = model.classifier[1].in_features
    #for param in model.parameters():
    #    param.requires_grad = False
    
    model.classifier = nn.Sequential(nn.Linear(in_features= num_ftrs, out_features = 1024),nn.ReLU(),\
        nn.Linear(in_features= 1024, out_features = 2), nn.Sigmoid())
    '''
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 1024), nn.ReLU(), nn.Linear(in_features= 1024, out_features = 2))

    for param in model.parameters():
        param.requires_grad = True
    
    model = model.to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8, weight_decay=0.035)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.1, last_epoch=-1)

    

    for epoch in range(epochs):
        iter = 0
        scheduler.step()
        total = 0
        total_correct = 0
        total_loss = 0

        model.train()
        for data, label in train_data:
            data = data.to('cuda')
            label = label.to('cuda')
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            #accumulated loss
            total_loss += loss.item() * label.size(0)
    
            #Total number of labels in batch
            batch_total = label.size(0)
            
            #total images
            total += batch_total
            
            #Obtaining predictions from max value
            _, predicted = torch.max(out, 1)
            
            #Calculate the number of correct answers
            batch_correct = torch.sum((predicted == label))
            #print("batch_correct: ",batch_correct)
            total_correct += batch_correct
            #print("total_correct: ", total_correct)
            
            #Print loss and accuracy
            
        total_valid_loss = 0
        valid_accuracy = 0
        total_correct_val = 0
        total_val = 0
            
        model.eval()
        with torch.no_grad():
            for data_valid, label_valid in valid_data:
                data_valid = data.to('cuda')
                label_valid = label_valid.to('cuda')
                valid_out = model(data_valid)
                batch_loss = criterion(valid_out, label_valid)
                total_valid_loss += batch_loss.item() * label_valid.size(0)

                _, predicted_val = torch.max(valid_out, 1)
                correct_val = torch.sum((predicted_val == label_valid))
                print("correct_val: ",correct_val)
                total_correct_val += correct_val
                total_val += label_valid.size(0)


                
        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%, \
        LR: {}'.format(epoch + 1, epochs, total_loss/len(trainImages),(total_correct / len(trainImages)) * 100, \
        optimizer.param_groups[0]['lr']))
        print("Validation: Loss: {:.4f}, Accuracy: {:.2f}%".format(total_valid_loss/len(validImages), total_correct_val/len(validImages)*100))
        

        #torch.save(model.state_dict(), modelSavePath)