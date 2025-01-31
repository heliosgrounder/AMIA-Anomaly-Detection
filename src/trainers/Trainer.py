import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from tqdm import tqdm

from src.nn.models.FasterRCNN import FasterRCNN
from src.nn.datasets.AMIADataset import AMIADataset, get_transform
from src.utils.constants import DEVICE

class Trainer:
    def __init__(self, 
                 batch_size=16, 
                 learning_rate=5e-3,
                 device=DEVICE, 
                 examination=False):
        self.device = device

        self.model = FasterRCNN()
        self.model = self.model.to(self.device)

        self.batch_size=batch_size
        self.learning_rate = learning_rate
        
        self.train_dataset = AMIADataset(
            transform=get_transform(),
        )
        self.test_dataset = AMIADataset(
            transform=get_transform(),
            train=False
        )

        if examination:
            self.train_dataset = Subset(self.train_dataset, range(50))
            self.test_dataset = Subset(self.test_dataset, range(50))


        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_loader,
            batch_size=self.batch_size,
            shuffle=False
        )

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)




    def train_model(self):
        self.model.train()

    def test_model(self):
        self.model.eval()

    def fit(self, num_epochs=10):
        for epoch in range(num_epochs):
            train_loss = self.train_model()
            test_loss = self.test_model()

