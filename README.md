# Image Classifier

In this project, I build a Python application that can train an image classifier on a dataset, then predict new images using the trained model.

# Dependencies
    * Python 3.7

# How to Run the Application
1. Training dataset
    * To get detail command run information, you can run => `python train.py --help`
    * Example command for training dataset => `python train.py '/.flowers'`
    
2. Using training dataset
    * To get detail command run information, you can run => `python predict.py --help`
    * Example command for training dataset => `python predict.py './flowers/test/1/image_06743.jpg' './checkpoint.pth'`
