from sklearn.model_selection import KFold

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from tqdm import tqdm

from src.nn.config import Config

from src.nn.models.FasterRCNN import FasterRCNN
from src.nn.models.yolo import YOLOv1
from src.nn.models.retinanet import RetinaNet

from src.nn.datasets.AMIADataset import AMIADataset, get_transform, get_train_transform
from src.utils.utils import collate_fn

CONFIG = Config()


class Trainer:
    def __init__(self, 
                 batch_size=CONFIG.batch_size, 
                 learning_rate=CONFIG.learning_rate,
                 device=CONFIG.device, 
                 model_type=CONFIG.model_type,
                 examination=False):

        self.device = device
        self.model_type = model_type

        if self.model_type == "FasterRCNN":
            self.model = FasterRCNN()
        elif self.model_type == "YOLOv1":
            self.model = YOLOv1()
        elif self.model_type == "RetinaNet":
            self.model = RetinaNet()
        else:
            raise Exception("Dont have model with this code. Try Again.")
        self.model = self.model.to(self.device)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.dataset = AMIADataset(
            transform=get_train_transform(),
            no_findings=True if self.model_type == "FasterRCNN" else False
        )

        if examination:
            self.dataset = Subset(self.dataset, range(100))
        
        # train_size = int(0.8 * len(self.dataset))
        # test_size = len(self.dataset) - train_size
        # self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])

        # if examination:
        #     self.train_dataset = Subset(self.train_dataset, range(50))
        #     self.test_dataset = Subset(self.test_dataset, range(50))
        self.kfolds = KFold(n_splits=CONFIG.folds, shuffle=True)

        # self.train_loader = DataLoader(
        #     self.train_dataset, 
        #     batch_size=self.batch_size,
        #     shuffle=True,
        #     collate_fn=collate_fn
        # )
        # self.test_loader = DataLoader(
        #     self.test_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     collate_fn=collate_fn
        # )

        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=CONFIG.weight_decay)

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 50, 1, 0.0001)

    def train_model(self):
        self.model.train()

        running_loss = 0.0
        for images, targets in tqdm(self.train_loader, desc="Training"):
            images = list(img.to(self.device) for img in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            self.scheduler.step()

            running_loss += losses.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def test_model(self):
        self.model.train()

        running_loss = 0.0
        for images, targets in tqdm(self.test_loader, desc="Testing"):
            images = list(img.to(self.device) for img in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            running_loss += losses.item()
        
        avg_loss = running_loss / len(self.test_loader)
        return avg_loss

    def fit(self, num_epochs=CONFIG.num_epochs):
        best_loss = float("inf")
        for fold, (train_ids, test_ids) in enumerate(self.kfolds.split(self.dataset)):
            print(f"Fold {fold+1}/{CONFIG.folds}")
            self.train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
            self.test_sampler = torch.utils.data.SubsetRandomSampler(test_ids)
            self.train_loader = DataLoader(
                self.dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                sampler=self.train_sampler
            )
            self.test_loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                sampler=self.test_sampler
            )

            for epoch in range(num_epochs):
                train_loss = self.train_model()
                test_loss = self.test_model()
                print(f"EPOCH: {epoch + 1}/{num_epochs} TRAIN LOSS: {train_loss} TEST LOSS: {test_loss}")
                if best_loss >= test_loss:
                    best_loss = test_loss
                    torch.save(self.model.state_dict(), f"pth_models/{CONFIG.model_name}.pth")
                    print(f"BEST MODEL IN {epoch+1} EPOCH and {fold+1} FOLD")
