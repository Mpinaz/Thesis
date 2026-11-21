import copy
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score


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


def build_model(name, config=None):
    if name not in MODELS:
        raise ValueError(f"Modello '{name}' sconosciuto. Disponibili: {list(MODELS)}")
    config = config or {}
    return MODELS[name](**config)





def Training (model, device, Epochs, opt, crit, Train_Loader, Val_Loader, scheduler=None):
    Epoch_Save = 0
    Metrics = {"train_loss": [], "val_loss": [],
                 "train_acc": [], "val_acc": [],
                 "val_bacc": [], "val_f1p": []}
    best_val_bacc = 0.0
    best_state = None
    
    
    pbar = tqdm(range(Epochs), desc="Training")
    for epoch in pbar:
        model.train()
        tot_loss, correct, total = 0.0, 0, 0
        for x, y in Train_Loader:
            x, y = x.to(device), y.to(device)
    
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
    
            tot_loss += loss.item() * x.size(0)
            correct  += (out.argmax(1) == y).sum().item()
            total    += x.size(0)
    
        train_loss = tot_loss / total
        train_acc  = correct / total
    
        # Validation
    
        model.eval()
        tot_loss, total = 0.0, 0
        val_true, val_pred = [], []
        with torch.no_grad():
            for x, y in Val_Loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = crit(out, y)
    
                tot_loss += loss.item() * x.size(0)
                total    += x.size(0)
                val_true.append(y.cpu())
                val_pred.append(out.argmax(1).cpu())
    
        val_loss = tot_loss / total
        val_true = torch.cat(val_true).numpy()
        val_pred = torch.cat(val_pred).numpy()
        val_acc  = (val_true == val_pred).mean()
        val_bacc = balanced_accuracy_score(val_true, val_pred)
        val_f1p  = f1_score(val_true, val_pred, pos_label=1)
    
        scheduler.step(val_loss)   # da attivare con lo scheduler
    
        Metrics["train_loss"].append(train_loss)
        Metrics["val_loss"].append(val_loss)
        Metrics["train_acc"].append(train_acc)
        Metrics["val_acc"].append(val_acc)
        Metrics["val_bacc"].append(val_bacc)
        Metrics["val_f1p"].append(val_f1p)
    
        if not (epoch + 1) % 25:
              tqdm.write(f"\nEpoch {epoch+1}\ntrain_acc={train_acc:.4f} | val_acc={val_acc:.3f} | val_bacc={val_bacc:.3f} | val_f1p={val_f1p:.3f}")
    
        if val_bacc > best_val_bacc:
            Epoch_Save = epoch
            best_val_bacc = val_bacc
            best_state = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_state)
    print(f"Best val balanced accuracy: {best_val_bacc:.4f} at {Epoch_Save} Epoch")

    return model, Metrics
