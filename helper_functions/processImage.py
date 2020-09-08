from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms

def processImage(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    # Open the image
    img = Image.open(image_path)

    # Resize the image    
    img = img.resize((256,256))
    
    # Crop the image
    img = img.crop((0,0,224,224))
    
    # Get the color channels
    img = np.array(img)/255
    
    # Normalize the images
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = (img - means) / std
          
    # Transpose the colors
    img = img.transpose((2, 0, 1))
          
    return np.array(img)