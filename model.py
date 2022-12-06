import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18


class Model(nn.Module):
    def __init__(self, feature_dim=512):
        super(Model, self).__init__()

        # encoder
        self.resnet  = resnet18(weights=None)
        features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # projection head
        self.embedding = nn.Sequential(nn.Linear(features, 512, bias=True),nn.BatchNorm1d(num_features=512),
                         nn.ReLU(), nn.Linear(512, 256, bias=True))

    def forward(self, x):
        x = self.resnet(x)
        out = self.embedding(x)
        
        return x, out
        

class ClassifierModel(nn.Module):
    def __init__(self, model):
        super(ClassifierModel, self).__init__()

        # encoder
        self.resnet = nn.Sequential(*(list(model.children())[:-1] ))
        
        # projection head
        self.embedding = nn.Sequential(nn.Linear(512, 4))

    def forward(self, x):
        x = self.resnet(x)
        out = self.embedding(x)
        
        return x, out

