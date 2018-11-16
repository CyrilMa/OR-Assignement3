import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import model as md
import torchvision.models as models

parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--model', type=str, default='experiment/model_10', metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='experiment/kaggle.csv', metavar='D',
                    help="name of the output csv file")

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

n_folds = 10
model_folds = [md.DenseNet161()]*n_folds
for i in range(n_folds):
    state_dict = torch.load(args.model+"_fold_%d.pth"%i)

if use_cuda:
    print('Using GPU')
    for model in model_folds:
        model.cuda()
else:
    print('Using CPU')

for model in model_folds:
    model.load_state_dict(state_dict)
    model.eval()

from data import data_transforms

test_dir = args.data + '/test_images/mistery_category'

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


output_file = open(args.outfile, "w")
output_file.write("Id,Category\n")
for f in tqdm(os.listdir(test_dir)):
    if 'jpg' in f:
        data = data_transforms(pil_loader(test_dir + '/' + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        if use_cuda:
            data = data.cuda()
        output = max(F.softmax(model(data)) for model in model_folds)
        pred = output.data.max(1, keepdim=True)[1]
        output_file.write("%s,%d\n" % (f[:-4], pred))

output_file.close()

print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle competition website')
        


