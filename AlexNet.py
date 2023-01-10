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
STD = 1e-2

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv_net = nn.Sequential(
            # Input (N, 3, 224, 244)
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            # (N, 96, 54, 54)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (N, 96, 26, 26)
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            # (N, 256, 26, 26)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (N, 256, 12, 12)
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            # (N, 384, 12, 12)
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            # (N, 384, 12, 12)
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            # (N, 256, 12, 12)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (N, 256, 5, 5)
            nn.Flatten()
            # (N, 6400),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=6400, out_features=4096),
            # (N, 4096)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=NUM_CLASSES)
            # (N, 1000)
        )
        self.init_weights()

    def forward(self, x):
        x = self.conv_net(x)
        x = self.classifier(x)
        return x

    def init_weights(self):
        pass

model = AlexNet()
optimizer = optim.SGD(momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, params=model.parameters(), lr=LR)

input_data = torch.randn(size=(BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE))
assert model(input_data).shape == torch.Size([BATCH_SIZE, NUM_CLASSES])