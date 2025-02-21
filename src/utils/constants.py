import torch

LABELS = {
    0: "No finging",
    1: "Aortic enlargement",
    2: "Atelectasis",
    3: "Calcification",
    4: "Cardiomegaly",
    5: "Consolidation",
    6: "ILD",
    7: "Infiltration",
    8: "Lung Opacity",
    9: "Nodule/Mass",
    10: "Other lesion",
    11: "Pleural effusion",
    12: "Pleural thickening",
    13: "Pneumothorax",
    14: "Pulmonary fibrosis",
}

MODEL_TYPES = {
    0: "FasterRCNN",
    1: "YOLOv1",
    2: "RetinaNet"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
