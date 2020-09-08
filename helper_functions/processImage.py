from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms

def processImage(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image_path)
    img = img.resize((256,256))
    img = img.crop((0,0,224,224))
    img = np.array(img)/255
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = (img - means) / std  
    img = img.transpose((2, 0, 1))
          
    return np.array(img)