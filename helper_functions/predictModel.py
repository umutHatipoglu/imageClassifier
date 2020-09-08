import torch
from torch import nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np

from helper_functions import processImage 

def predict(categories, image_path, model, use_gpu, topk):
    '''
        Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    matchingCategoryLabels = []
    
    with torch.no_grad():
        processedImage = processImage(image_path)
        image = torch.from_numpy(np.array([processedImage])).float()
        logps = model.forward(image)
        ps = torch.exp(logps)
        probs, classes = ps.topk(topk)

        top_p = probs.tolist()[0]
        top_classes = classes.tolist()[0]

        idx_to_class = {v:k for k, v in model.class_to_idx.items()}

        for matchClass in top_classes:
            matchCategories = categories[idx_to_class[matchClass]]
            matchingCategoryLabels.append(matchCategories)

        print("Matching probabilites - Categorie Classes:", list(zip(top_p, matchingCategoryLabels)))
        return top_p, matchingCategoryLabels