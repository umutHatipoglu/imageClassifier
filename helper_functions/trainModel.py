import torch
from torch import nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np

def trainModel(model, trainLoaders, validLoaders, criterion, optimizer, epochs, use_gpu):
    outputResultSteps = 30
    start_time = time.time()
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)
 
    training_train_loss = []
    training_valid_loss = []
    validation_accuracies = []
    steps = 0

    for epoch in range(epochs):
        model.train()
        training_loss = 0
        
        for images, labels in trainLoaders:   
            steps += 1         
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            
            if steps % outputResultSteps == 0:
                validation_loss = 0
                validation_accuracy = 0
                model.eval()

                with torch.no_grad():
                    for images, labels in validLoaders:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)                    
                        batch_loss = criterion(logps, labels)
                        ps = torch.exp(logps)
                        
                        top_p, top_class = ps.topk(1, dim=1)                        
                        validation_accuracy += torch.mean((top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)).item()
                        
                        validation_loss += batch_loss.item()
                        
                        training_train_loss.append(training_loss/len(trainLoaders))
                        training_valid_loss.append(validation_loss/len(validLoaders))
                        validation_accuracies.append(validation_accuracy/len(validLoaders) * 100)

                        print("Epoch: {}\n".format(epoch),
                                "Training Loss: {:.4f}\n".format(training_loss/len(trainLoaders)),
                                "Validation Loss: {:.4f}\n".format(validation_loss/len(validLoaders)),
                                "Validation Accuracy: {:.4f}\n".format(validation_accuracy/len(validLoaders) * 100))

    
    # Get the total time that has elapsed
    elapsed_time = time.time() - start_time  
    print("Total Time: {}\n".format(elapsed_time))

