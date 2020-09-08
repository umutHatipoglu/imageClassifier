import torch
from torch import nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

def buildModel(arch, hidden_units):
    if arch.lower() == "vgg19":
        model = models.vgg19(pretrained=True)
        input_features = 25088
    else:
        model = models.densenet161(pretrained=True)
        input_features = 2208
    
    # Freeze the parameters
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                ('inputs', nn.Linear(25088, 120)),
                ('relu1', nn.ReLU()),
                ('dropout',nn.Dropout(0.5)),
                ('hidden_layer1', nn.Linear(120, 90)),
                ('relu2',nn.ReLU()),
                ('hidden_layer2',nn.Linear(90,70)),
                ('relu3',nn.ReLU()),
                ('hidden_layer3',nn.Linear(70,102)),
                ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier

    print("Done creating the model\n")
    return model