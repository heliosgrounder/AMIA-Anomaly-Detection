import torch
import torch.nn as nn
import torchvision


class FasterRCNN(nn.Module):
    def __init__(self, num_classes=15, pretrained=True, v1=True):
        super(FasterRCNN, self).__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained

        if v1:
            _weights = torchvision.models.detection.faster_rcnn.FasterRCNN_ResNet50_FPN_Weights.DEFAULT if self.pretrained else None
            self.backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=_weights)
        else:
            _weights = torchvision.models.detection.faster_rcnn.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if self.pretrained else None
            self.backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                weights=_weights,
                weights_backbone=torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
                trainable_backbone_layers=5
            )
        
        in_features = self.backbone.roi_heads.box_predictor.cls_score.in_features
        self.backbone.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=self.num_classes)


    def forward(self, images, targets=None):
        return self.backbone(images, targets)
