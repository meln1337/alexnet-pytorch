import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
NUM_EPOCHS = 90
WEIGHT_DECAY = 5e-4
LR = 1e-2
BATCH_SIZE = 128
MOMENTUM = 0.9
NUM_CLASSES = 1000
IMG_SIZE = 224
IMG_CHANNELS = 3