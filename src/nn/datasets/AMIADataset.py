import os
import numpy as np
import pandas as pd
from PIL import Image

import cv2

import torch
from torch.utils.data import Dataset
from torchvision.tv_tensors import BoundingBoxes

import albumentations as A
import torchvision.transforms as T

from src.utils.utils import merge_bbox


class AMIADataset(Dataset):
    def __init__(self, data_folder="data", transform=None, train=True, no_findings=True):
        __KEY = "image_id"
        self.__IMAGE_SIZE = 1024
        self.data_folder = data_folder
        self.transform = transform
        self.train = train
        self.no_findings = no_findings
        if self.train:
            folder = "train"
        else:
            folder = "test"

        # img_size hashtable
        self.df_img_size = pd.read_csv(os.path.join(data_folder, "img_size.csv"))
        self.df_img_size.set_index(__KEY, inplace=True)

        # annotations hashtable
        self.df_annotations = pd.read_csv(os.path.join(data_folder, f"{folder}.csv"))
        if self.train:
            self.df_annotations["class_id"] = self.df_annotations["class_id"].apply(lambda x: x + 1 if x != 14 else 0)
            results = []
            for (image_id, class_id), group in self.df_annotations.groupby(["image_id", "class_id"]):
                if group[["x_min", "y_min", "x_max", "y_max"]].isna().all().all():
                    results.append({
                        "image_id": image_id,
                        "class_id": class_id,
                        "x_min": np.nan,
                        "y_min": np.nan,
                        "x_max": np.nan,
                        "y_max": np.nan
                    })
                else:
                    merged = merge_bbox(group, iou_threshold=0.5)
                    for bbox in merged:
                        results.append({
                            "image_id": image_id,
                            "class_id": class_id,
                            "x_min": bbox[0],
                            "y_min": bbox[1],
                            "x_max": bbox[2],
                            "y_max": bbox[3]
                        })
            self.df_annotations = pd.DataFrame(results)
            self.df_annotations = self.df_annotations.astype({
                "image_id": "str",
                "class_id": "int64",
                "x_min": "float32",
                "y_min": "float32",
                "x_max": "float32",
                "y_max": "float32"
            })
        
            # resize bboxes
            for column in ("x_min", "y_min", "x_max", "y_max"):
                dim = "dim1" if column[0] == "x" else "dim0"
                self.df_annotations[column] = self.df_annotations.apply(lambda x: x[column] / self.df_img_size.loc[x["image_id"]][dim] * 1024, axis=1)

        self.df_annotations.set_index(__KEY, inplace=True)

        self.images_path = []
        for image_filename in os.listdir(os.path.join(data_folder, folder, folder)):
            image_path = os.path.join(data_folder, folder, folder, image_filename)
            self.images_path.append(image_path)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        image_uuid = os.path.splitext(os.path.basename(image_path))[0]
        # img_size = self.df_img_size.loc[image_uuid].to_list()

        # image = Image.open(image_path).convert("RGB")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.train:
            annotations = self.df_annotations.loc[[image_uuid]]

            if set(annotations["class_id"]) == {0}:
                labels = torch.tensor([0], dtype=torch.int64) if self.no_findings else torch.empty((0,), dtype=torch.int64)
                boxes = torch.empty((0, 4), dtype=torch.float32)
            else:
                # merged_annotations = (
                #     annotations.groupby("class_id")
                #     .agg({
                #         "x_min": "mean",
                #         "y_min": "mean",
                #         "x_max": "mean",
                #         "y_max": "mean"
                #     })
                #     .reset_index()
                # )

                labels = []
                boxes = []
                for _, row in annotations.iterrows():
                    labels.append(int(row["class_id"]) - int(not self.no_findings))
                    boxes.append([row["x_min"], row["y_min"], row["x_max"], row["y_max"]])

                labels = torch.tensor(labels, dtype=torch.int64)
                boxes = torch.tensor(boxes, dtype=torch.float32)

            # boxes = BoundingBoxes(boxes, format="xyxy", canvas_size=torch.Size((self.__IMAGE_SIZE, self.__IMAGE_SIZE)))
            # print(boxes.numpy())
            # print(labels.numpy())

            # print(boxes)

            # transform part
            if self.transform:
                # image = self.transform(image)
                transformed = self.transform(
                    image=image, 
                    bboxes=boxes.numpy(),
                    labels=labels.numpy()
                )

            # target = {
            #     "boxes": boxes,
            #     "labels": labels,
            #     # "image_uuid": torch.tensor([image_uuid])
            # }
            

            image = transformed["image"]
            boxes = torch.from_numpy(transformed["bboxes"])
            labels = torch.tensor(transformed["labels"], dtype=torch.int64)

            target = {
                "boxes": boxes,
                "labels": labels,
                # "image_uuid": torcsh.tensor([image_uuid])
            }
        
            return image, target
        else:
            if self.transform:
                image = self.transform(image=image)
            return image, image_uuid


def get_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_train_transform():
    return A.Compose(
        [
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9)
            ], p=0.9),
            A.ImageCompression(compression_type="jpeg", quality_range=(85, 95), p=0.2),
            A.OneOf([
                A.Blur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0)
            ], p=0.1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.Resize(width=1024, height=1024, p=1.0),
            A.CoarseDropout(num_holes_range=(8, 12), hole_height_range=(32, 64), hole_width_range=(32, 64), fill=0, p=0.5),
            A.Normalize(),
            A.ToTensorV2(p=1.0)
        ], 
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"]
        )
    )


def get_validation_transform():
    return A.Compose(
        [
            A.Resize(width=1024, height=1024, p=1.0),
            A.Normalize(),
            A.ToTensorV2(p=1.0) 
        ],
        p=1.0
    )