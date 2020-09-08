import torch
from torch import nn as nn
from torch import optim as optim
import argparse
from helper_functions import trainModel, buildModel, saveModel, loadData

parser = argparse.ArgumentParser(description="Train NN")
parser.add_argument('data_directory', default='./flowers', help="Path for traning images")
parser.add_argument('--save_dir', default='./', help="Path for checkpoint")             
parser.add_argument('--arch', default="vgg19", help="Architecture types: (vgg19 or densenet161)")
parser.add_argument('--learning_rate', type=float, default="0.001", help="Learning rate")
parser.add_argument('--hidden_units', type=int, default=512, help="Hidden units number")
parser.add_argument('--epochs', type=int, default=2, help="Number of training epochs")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
parser.add_argument('--gpu', default=False, action='store_true',help="GPU usage(boolean)")

options = parser.parse_args()

train_dataloaders, valid_dataloaders, test_dataloaders, train_datasets = loadData(options.data_directory, options.batch_size)
model = buildModel(options.arch, options.hidden_units)

opt = optim.Adam(model.classifier.parameters(), options.learning_rate) 
criterion = nn.NLLLoss()

trainModel(model, train_dataloaders, valid_dataloaders, criterion, opt, options.epochs, options.gpu)
saveModel(model, train_datasets, options.epochs, opt, options.arch)
