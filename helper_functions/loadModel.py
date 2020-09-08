import torch
from torch import nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np

def loadModel(checkpoint_file):
    # Load the model and force the tensors to be on the CPU
    checkpoint = torch.load(checkpoint_file)
   
    if(checkpoint['arch'].lower() == 'vgg19' or checkpoint['arch'].lower() == 'densenet161'):
        model = getattr(torchvision.models, checkpoint['arch'])(pretrained = True)


    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    # Freeze the parameters so we dont backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    print("Done loading the model")
    return model    