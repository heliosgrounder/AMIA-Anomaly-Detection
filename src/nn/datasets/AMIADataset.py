import os
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.tv_tensors import BoundingBoxes
import torchvision.transforms as T


class AMIADataset(Dataset):
    def __init__(self, data_folder="data", transform=None, train=True):
        __KEY = "image_id"
        self.data_folder = data_folder
        self.transform = transform
        self.train = train
        if train:
            folder = "train"
        else:
            folder = "test"

        # img_size hashtable
        self.df_img_size = pd.read_csv(os.path.join(data_folder, "img_size.csv"))
        self.df_img_size.set_index(__KEY, inplace=True)

        # annotations hashtable
        self.df_annotations = pd.read_csv(os.path.join(data_folder, f"{folder}.csv"))
        self.df_annotations.set_index(__KEY, inplace=True)

        self.images_path = []
        for image_filename in os.listdir(os.path.join(data_folder, folder, folder)):
            self.images_path.append(os.path.join(data_folder, folder, folder, image_filename))

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        image_uuid = os.path.splitext(os.path.basename(image_path))[0]
        img_size = self.df_img_size.loc[image_uuid].to_list()

        image = Image.open(image_path).convert("RGB")

        if self.train:
            annotations = self.df_annotations.loc[image_uuid]

            if set(annotations["class_id"]) == {14}:
                labels = torch.tensor([14], dtype=torch.int64)
                boxes = torch.empty((0, 4), dtype=torch.float32)
            else:
                merged_annotations = (
                    annotations.groupby("class_id")
                    .agg({
                        "x_min": "mean",
                        "y_min": "mean",
                        "x_max": "mean",
                        "y_max": "mean"
                    })
                    .reset_index()
                )

                labels = []
                boxes = []
                for _, row in merged_annotations.iterrows():
                    labels.append(int(row["class_id"]))
                    boxes.append([row["x_min"], row["y_min"], row["x_max"], row["y_max"]])

                labels = torch.tensor(labels, dtype=torch.int64)
                boxes = torch.tensor(boxes, dtype=torch.float32)
                

            boxes = BoundingBoxes(boxes, format="xyxy", canvas_size=torch.Size(img_size))

            # transform part
            if self.transform:
                image = self.transform(image)

            target = {
                "boxes": boxes,
                "labels": labels
            }

            return image, target
        else:
            # transform part
            if self.transform:
                image = self.transform(image)
            
            return image, image_uuid


def get_transform():
    return T.Compose([
        T.ToTensor()
    ])