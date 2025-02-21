import torch
import torch.nn as nn

class YOLOv1(nn.Module):
    def __init__(self, num_classes=15, grid_size=7, bb=2):
        super(YOLOv1, self).__init__()

        self.grid_size = grid_size
        self.bb = bb
        self.num_clusses = num_classes
