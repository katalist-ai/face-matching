from itertools import chain

from torch import nn
from torchvision.models import ResNet34_Weights, resnet34


class AgeSexClassify(nn.Module):
    def __init__(self):
        super(AgeSexClassify, self).__init__()
        self.base_model = resnet34(weights=ResNet34_Weights.DEFAULT)

        # freeze the base model weights
        for param in self.base_model.parameters():
            param.requires_grad = False

        # unfreeze the last layer (layer4)
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True

        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        # Add new layers for age and sex classification
        self.age = nn.Linear(num_ftrs, 2)
        self.sex = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        x = self.base_model(x)
        x_age = self.age(x)
        x_sex = self.sex(x)
        return x_age, x_sex

    def get_trainable_parameters(self):
        return chain(self.base_model.layer4.parameters(), self.age.parameters(), self.sex.parameters())
