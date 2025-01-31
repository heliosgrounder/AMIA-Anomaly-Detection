import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from tqdm import tqdm

from src.nn.models.FasterRCNN import FasterRCNN
from src.nn.datasets.AMIADataset import AMIADataset, get_transform
from src.utils.constants import DEVICE
from src.utils.utils import collate_fn

class Trainer:
    def __init__(self, 
                 batch_size=8, 
                 learning_rate=5e-3,
                 device=DEVICE, 
                 examination=False):
        self.device = device

        self.model = FasterRCNN()
        self.model = self.model.to(self.device)

        self.batch_size=batch_size
        self.learning_rate = learning_rate
        
        self.dataset = AMIADataset(
            transform=get_transform(),
        )
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])

        if examination:
            self.train_dataset = Subset(self.train_dataset, range(50))
            self.test_dataset = Subset(self.test_dataset, range(50))


        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)

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

    def fit(self, num_epochs=10):
        for epoch in range(num_epochs):
            train_loss = self.train_model()
            test_loss = self.test_model()
            print(f"EPOCH: {epoch + 1}/{num_epochs} TRAIN LOSS: {train_loss} TEST LOSS: {test_loss}")

