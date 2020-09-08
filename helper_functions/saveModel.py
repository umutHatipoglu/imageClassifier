import torch
from torch import nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np

def saveModel(model, train_datasets, epochs, optimizer, arch):
    
    print("Saving the model...")

    model.class_to_idx = train_datasets.class_to_idx

    if arch.lower() == "vgg19":
        input_features = 25088
    elif arch.lower() == "densenet161":
        input_features = 2208

    checkpoint = {
              'arch': arch,
              'classifier': model.classifier,
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict(),
              'epochs' : epochs,
              'optimizer_models' : optimizer.state_dict()}


    torch.save(checkpoint, 'checkpoint.pth')
    print("Done saving the model")