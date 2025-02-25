from src.utils.constants import DEVICE, MODEL_TYPES


class Config:
    def __init__(self):
        self.model_name = "best_model_augmentation_5"

        self.folds = 5
        self.num_epochs = 50
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.device = DEVICE

        self.model_type = MODEL_TYPES[0]
