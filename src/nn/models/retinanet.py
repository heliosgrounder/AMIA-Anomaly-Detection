import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


class RetinaNet(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(RetinaNet, self).__init__()
        
        self.num_classes = num_classes
        self.pretrained = pretrained


        _weights = torchvision.models.detection.retinanet.RetinaNet_ResNet50_FPN_Weights.DEFAULT if self.pretrained else None
        self.model = torchvision.models.detection.retinanet_resnet50_fpn(weights=_weights)

        # in_channels = self.model.head.classification_head.conv[0].in_channels
        in_channels = 256
        num_anchors = self.model.head.classification_head.num_anchors

        self.model.head.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, self.num_classes)

    def forward(self, images, targets=None):
        if targets:
            return self.model(images, targets)
        else:
            return self.model(images)
