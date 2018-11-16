import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

nclasses = 20 

class DenseNet161(nn.Module):
    def __init__(self):
        super(DenseNet161, self).__init__()

        self.model = models.densenet161(pretrained=True)
        for i, param in list(self.model.named_parameters())[0:336]:
            param.requires_grad = False
        self.model.classifier = nn.Linear(self.model.classifier.in_features, nclasses)   

    def forward(self, x):
        x = self.model(x)
        return x
    
class DenseNet201(nn.Module):
    def __init__(self):
        super(DenseNet201, self).__init__()

        self.model = models.densenet201(pretrained=True)
        for i, param in list(self.model.named_parameters())[0:498]:
            param.requires_grad = False
        self.model.classifier = nn.Linear(self.model.classifier.in_features, nclasses)   

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet152(nn.Module):
    def __init__(self):
        super(ResNet152, self).__init__()

        self.model = models.resnet152(pretrained=True)
        for i, param in list(self.model.named_parameters())[:435]:
            param.requires_grad = False
    
        self.model.fc = nn.Linear(self.model.fc.in_features, nclasses)   

    def forward(self, x):
        x  = self.model(x)
        return x

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

        self.model = models.vgg19(pretrained=True)
        for i, param in self.model.named_parameters():
            param.requires_grad = False
    
        self.model.fc = nn.Linear(self.model.fc.in_features, nclasses)   

    def forward(self, x):
        x = F.softmax(self.model(x))
        return x
