import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, Architecture, dropout=0.0):
        super().__init__()
        layers = []
        for idx in range(len(Architecture) - 1):
            layers += [nn.Linear(Architecture[idx], Architecture[idx + 1]), nn.ReLU(), nn.Dropout(dropout)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class CNN1D(nn.Module):
    def __init__(self, input_channels=3, output_classes=2, dropout_chance = 0.4):
        super().__init__()
        self.p = dropout_chance
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            nn.ReLU(),

            nn.Dropout(self.p),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.ReLU(),

            nn.Dropout(self.p),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            nn.ReLU(),


            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )
        self.lin = nn.Linear(256, output_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.lin(x)

class CNN2D(nn.Module):
    def __init__(self, input_channels=3, output_classes=2, dropout_chance = 0.4):
        super().__init__()
        self.p = dropout_chance
        self.features = nn.Sequential(
            
            nn.Conv2d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout2d(self.p),


            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout2d(self.p),


            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.ReLU(),


            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )
        self.lin = nn.Linear(256, output_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.lin(x)

MODELS = {
    "mlp": MLP,
    "cnn1d": CNN1D,
    "cnn2d": CNN2D}


def build_model(name, config = []):
    if name not in MODELS:
        raise ValueError(f"Modello '{name}' sconosciuto. Disponibili: {list(MODELS)}")
    return MODELS[name] if name != "mlp" else MODELS[name](**config)

