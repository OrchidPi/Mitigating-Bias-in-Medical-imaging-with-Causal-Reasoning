#https://github.com/pangwong/pytorch-multi-label-classifier/blob/master/models/build_model.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

class MultiLabelResNet(nn.Module):
    def __init__(self, original_resnet, num_classes):
        super().__init__()
        self.features = nn.Sequential(*list(original_resnet.children())[:-1])  # Remove the original fc layer
        self.num_classes = num_classes
        for index, num_class in enumerate(num_classes):
            setattr(self, "FullyConnectedLayer_" + str(index), nn.Linear(original_resnet.fc.in_features, num_class))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten the output features of the convolutional layers
        outs = []
        for index, num_class in enumerate(self.num_classes):
            fc_layer = getattr(self, "FullyConnectedLayer_" + str(index))
            out = fc_layer(x)
            outs.append(out)
        return outs

#def get_resnet_classifier(resnet_type='resnet18', num_classes=num_class, pretrained=True):
  #  original_resnet = resnet18(pretrained=pretrained) if resnet_type == 'resnet18' else resnet50(pretrained=pretrained)
  #  model = MultiLabelResNet(original_resnet, num_classes)
  #  return model
