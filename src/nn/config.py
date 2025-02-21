from src.utils.constants import DEVICE, MODEL_TYPES


class Config:
    def __init__(self):
        self.batch_size = 8
        self.learning_rate = 5e-4
        self.device = DEVICE

        self.model_type = MODEL_TYPES[0]
