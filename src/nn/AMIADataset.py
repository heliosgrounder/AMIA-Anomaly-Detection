import os
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.tv_tensors import BoundingBoxes


class AMIADataset(Dataset):
    def __init__(self, data_folder="data", transform=None):
        self.label_map = {
            0: "Aortic enlargement",
            1: "Atelectasis",
            2: "Calcification",
            3: "Cardiomegaly",
            4: "Consolidation",
            5: "ILD",
            6: "Infiltration",
            7: "Lung Opacity",
            8: "Nodule/Mass",
            9: "Other lesion",
            10: "Pleural effusion",
            11: "Pleural thickening",
            12: "Pneumothorax",
            13: "Pulmonary fibrosis",
            14: "No finding"
        }
        __KEY = "image_id"
        self.data_folder = data_folder
        self.transform = transform

        # img_size hashtable
        self.df_img_size = pd.read_csv(os.path.join(data_folder, "img_size.csv"))
        self.df_img_size.set_index(__KEY, inplace=True)

        # annotations hashtable
        self.df_annotations = pd.read_csv(os.path.join(data_folder, "train.csv"))
        self.df_annotations.set_index(__KEY, inplace=True)

        self.images_path = []
        for image_filename in os.listdir(os.path.join(data_folder, "train", "train")):
            self.images_path.append(os.path.join(data_folder, "train", "train", image_filename))

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):

        image_path = self.images_path[idx]
        image_uuid = os.path.splitext(os.path.basename(image_path))[0]
        img_size = self.df_img_size.loc[image_uuid].to_list()
        annotations = self.df_annotations.loc[image_uuid]

        image = Image.open(image_path).convert("RGB")

        boxes = []
        labels = []
        if set(annotations["class_id"]) == {14}:
            labels.append(14)
        else:
            for idx in range(annotations.index.size):
                anno_temp = annotations.iloc[idx][["class_id", "x_min", "y_min", "x_max", "y_max"]].to_list()
                labels.append(anno_temp[0])
                boxes.append(anno_temp[1:])
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int8)

        bbox = BoundingBoxes(boxes, format="xyxy", canvas_size=torch.Size(img_size))

        # transform part (incomplete)


        target = {
            "label": labels,
            "bbox": bbox
        }

        return image, target
