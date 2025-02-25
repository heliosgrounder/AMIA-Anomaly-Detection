import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset

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

    def get_images(self, idx):
        image_path = self.images_path[idx]
        image_uuid = os.path.splitext(os.path.basename(image_path))[0]

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        annotations = self.df_annotations.loc[[image_uuid]]

        if set(annotations["class_id"]) == {0}:
            labels = torch.tensor([0], dtype=torch.int64) if self.no_findings else torch.empty((0,), dtype=torch.int64)
            boxes = torch.empty((0, 4), dtype=torch.float32)
        else:
            labels = []
            boxes = []
            for _, row in annotations.iterrows():
                labels.append(int(row["class_id"]) - int(not self.no_findings))
                boxes.append([row["x_min"], row["y_min"], row["x_max"], row["y_max"]])

            labels = torch.tensor(labels, dtype=torch.int64)
            boxes = torch.tensor(boxes, dtype=torch.float32)

        return image, boxes, labels
        
    def get_mixup(self, idx):
        image, boxes, labels = self.get_images(idx)
        random_image, random_boxes, random_labels = self.get_images(random.randint(0, len(self.images_path) - 1))

        fused_boxes = torch.cat((boxes, random_boxes))
        fused_labels = torch.cat((labels, random_labels))

        return (image + random_image) / 2, fused_boxes, fused_labels
    
    def get_cutmix(self, idx):
        half_size = self.__IMAGE_SIZE // 2
        x_center, y_center = [int(random.uniform(self.__IMAGE_SIZE * 0.25, self.__IMAGE_SIZE * 0.75)) for _ in range(2)]
        indexes = [idx] + [random.randint(0, len(self.images_path) - 1) for _ in range(3)]

        result_image = np.full((self.__IMAGE_SIZE, self.__IMAGE_SIZE, 3), 1, dtype=np.float32)
        result_boxes = torch.empty((0, 4), dtype=torch.float32)
        result_labels = torch.empty((0,), dtype=torch.int64)

        for i, index in enumerate(indexes):
            image, boxes, labels = self.get_images(index)
            if i == 0:
                x_min_1, y_min_1, x_max_1, y_max_1 = max(x_center - self.__IMAGE_SIZE, 0), max(y_center - self.__IMAGE_SIZE, 0), x_center, y_center
                x_min_2, y_min_2, x_max_2, y_max_2 = self.__IMAGE_SIZE - (x_max_1 - x_min_1), self.__IMAGE_SIZE - (y_max_1 - y_min_1), self.__IMAGE_SIZE, self.__IMAGE_SIZE
            elif i == 1:
                x_min_1, y_min_1, x_max_1, y_max_1 = x_center, max(y_center - self.__IMAGE_SIZE, 0), min(x_center + self.__IMAGE_SIZE, half_size * 2), self.__IMAGE_SIZE
                x_min_2, y_min_2, x_max_2, y_max_2 = 0, self.__IMAGE_SIZE - (y_max_1 - y_min_1), min(self.__IMAGE_SIZE, x_max_1 - x_min_1), self.__IMAGE_SIZE
            elif i == 2:
                x_min_1, y_min_1, x_max_1, y_max_1 = max(x_center - self.__IMAGE_SIZE, 0), y_center, x_center, min(half_size * 2, y_center + self.__IMAGE_SIZE)
                x_min_2, y_min_2, x_max_2, y_max_2 = self.__IMAGE_SIZE - (x_max_1 - x_min_1), 0, max(x_center, self.__IMAGE_SIZE), min(y_max_1 - y_min_1, self.__IMAGE_SIZE)
            elif i == 3:
                x_min_1, y_min_1, x_max_1, y_max_1 = x_center, y_center, min(x_center + self.__IMAGE_SIZE, half_size * 2), min(half_size * 2, y_center + self.__IMAGE_SIZE)
                x_min_2, y_min_2, x_max_2, y_max_2 = 0, 0, min(self.__IMAGE_SIZE, x_max_1 - x_min_1), min(y_max_1 - y_min_1, self.__IMAGE_SIZE)
            
            result_image[y_min_1:y_max_1, x_min_1:x_max_1] = image[y_min_2:y_max_2, x_min_2:x_max_2]
            padding_w = x_min_1 - x_min_2
            padding_h = y_min_1 - y_min_2

            boxes[:, 0] += padding_w
            boxes[:, 1] += padding_h
            boxes[:, 2] += padding_w
            boxes[:, 3] += padding_h

            result_boxes = torch.cat((result_boxes, boxes))
            result_labels = torch.cat((result_labels, labels))

        result_boxes = result_boxes.numpy()
        np.clip(result_boxes[:, 0:], 0, 2 * half_size, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        index_to_use = np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)

        result_boxes = torch.from_numpy(result_boxes[index_to_use])
        result_labels = result_labels[index_to_use]
        
        return result_image, result_boxes, result_labels


    def __getitem__(self, idx):
        if self.train:
            # if random.random() > 0.33:
            #     image, boxes, labels = self.get_images(idx)
            # elif random.random() > 0.5:
            #     image, boxes, labels = self.get_mixup(idx)
            # else:
            #     image, boxes, labels = self.get_cutmix(idx)

            image, boxes, labels = self.get_images(idx)

            if self.transform:
                transformed = self.transform(
                    image=image, 
                    bboxes=boxes.numpy(),
                    labels=labels.numpy()
                )
                
                image = transformed["image"]
                boxes = torch.from_numpy(transformed["bboxes"])
                labels = torch.tensor(transformed["labels"], dtype=torch.int64)
            else:
                totensor = T.Compose([
                    T.ToTensor()
                ])
                image = totensor(image)

            target = {
                "boxes": boxes,
                "labels": labels,
            }

            return image, target
        else:
            image_path = self.images_path[idx]
            image_uuid = os.path.splitext(os.path.basename(image_path))[0]

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image /= 255.0
            if self.transform:
                image = self.transform(image=image)
            else:
                totensor = T.Compose([
                    T.ToTensor()
                ])
                image = totensor(image)
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