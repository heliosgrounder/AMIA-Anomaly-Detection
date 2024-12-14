import os
import pandas as pd

from PIL import Image

from torch.utils.data import Dataset


class AMIADataset(Dataset):
    def __init__(self, data_folder="data", transform=None):
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
        for image_path in os.listdir(os.path.join(data_folder, "train", "train")):
            self.images_path.append(os.path.join(data_folder, "train", "train", image_path))

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        pass
