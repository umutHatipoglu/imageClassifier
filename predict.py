import torchvision
from torchvision import datasets, transforms, models
import argparse

from helper_functions import loadJson
from helper_functions import predict
from helper_functions import loadModel

parser = argparse.ArgumentParser(description="Load NN")
parser.add_argument('data_directory', help="Path for imege files")
parser.add_argument('--gpu', default=False, action='store_true', help="Use gpu boolean")
parser.add_argument('--category_names', default = './cat_to_name.json', help="Category file path")
parser.add_argument('--top_k', default=1, type=int, help="Number for likely classes")
parser.add_argument('checkpoint', help="Path for checkpoint")

arguments = parser.parse_args()
categories = loadJson(arguments.category_names)
model = loadModel(arguments.checkpoint)
predict(categories, arguments.data_directory, model, arguments.gpu, arguments.top_k)
