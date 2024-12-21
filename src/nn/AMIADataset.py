import os
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.tv_tensors import BoundingBoxes


class AMIADataset(Dataset):
    def __init__(self, data_folder="data", transform=None):
        __KEY = "image_id"
        self.data_folder = data_folder
        self.transform = transform

        # img_size hashtable
        self.df_img_size = pd.read_csv(os.path.join(data_folder, "img_size.csv"))
        self.df_img_size.set_index(__KEY, inplace=True)

        # annotations hashtable
        self.images_info = []
        self.df_annotations = pd.read_csv(os.path.join(data_folder, "train.csv"))
        # self.df_annotations.set_index(__KEY, inplace=True)
        for idx in range(self.df_annotations.index.size):
            self.images_info.append(self.df_annotations[["image_id", "rad_id", "class_id", "x_min", "y_min", "x_max", "y_max"]].iloc[idx].to_list())

        # self.images_path = []
        # for image_filename in os.listdir(os.path.join(data_folder, "train", "train")):
        #     self.images_path.append(os.path.join(data_folder, "train", "train", image_filename))

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        image_info = self.images_info[idx]
        image_uuid = image_info[0]
        image_path = os.path.join(self.data_folder, "train", "train", image_uuid + ".png")
        img_size = self.df_img_size.loc[image_uuid].to_list()

        image = Image.open(image_path).convert("RGB")
        label = torch.tensor(image_info[2], dtype=torch.int8)
        bbox = []
        if label != 14:
            bbox = image_info[3:]
        bbox = BoundingBoxes([bbox], format="xyxy", canvas_size=torch.Size(img_size))

        # image_path = self.images_path[idx]
        # image_uuid = os.path.splitext(os.path.basename(image_path))[0]
        # img_size = self.df_img_size.loc[image_uuid].to_list()
        # annotation = self.df_annotations.loc[image_uuid].to_list()

        # image = Image.open(image_path).convert("RGB")
        # label = torch.tensor(annotation[1], dtype=torch.int8)
        # bbox = []
        # if label != 14:
        #     bbox = annotation[3:]
        # bbox = BoundingBoxes([bbox], format="xyxy", canvas_size=torch.Size(img_size))

        # transform part (incomplete)


        target = {
            "label": label,
            "bbox": bbox
        }

        return image, target
