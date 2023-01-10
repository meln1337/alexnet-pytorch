import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

# Hyperparameters
NUM_EPOCHS = 90
WEIGHT_DECAY = 5e-4
LR = 1e-2
BATCH_SIZE = 128
MOMENTUM = 0.9
NUM_CLASSES = 1000
IMG_SIZE = 224
IMG_CHANNELS = 3
MEAN = 0
STD = 1e-2
TRAIN_IMG_DIR = 'files/imagenet'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv_net = nn.Sequential(
            # Input (N, 3, 224, 244)
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            # (N, 96, 54, 54)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (N, 96, 26, 26)
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            # (N, 256, 26, 26)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
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
            # (N, 6400)
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
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=MEAN, std=STD)
                nn.init.constant_(layer.bias, 0)

        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)

dataset = datasets.ImageFolder(TRAIN_IMG_DIR)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = AlexNet().to(device)
optimizer = optim.SGD(momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, params=model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

input_data = torch.randn(size=(BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE))
assert model(input_data).shape == torch.Size([BATCH_SIZE, NUM_CLASSES])

hist = {
    'loss': [0 * NUM_EPOCHS]
}

for i in range(NUM_EPOCHS):
    lr_scheduler.step()
    print(f'Epoch: {i+1}')
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            hist['loss'][i] += loss.item() / len(train_loader)

        print(f'Epoch: {i+1}, loss={loss.item()}')
    print(f'Epoch: {i + 1}, average loss={hist["loss"][i]}')

torch.save(model, './AlexNet.pth')
print('Models are saved')