import torch
import torch.nn as nn
import torchvision


class FasterRCNN(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(FasterRCNN, self).__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained

        self.backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=self.pretrained)
        in_features = self.backbone.roi_heads.box_predictor.cls_score.in_features
        self.backbone.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=self.num_classes)

    def forward(self, x):
        return self.backbone(x)
