import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Load dataset
df = pd.read_csv("energy_maps_with_labels.csv")
X = df.iloc[:, :-2].values.reshape(-1, 1, 8, 8).astype(np.float32)
y = df[["x_true", "y_true"]].values.astype(np.float32)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# Define CNN
class CNNSubPixelRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNSubPixelRegressor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(30):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

# Inference
model.eval()
with torch.no_grad():
    xb, yb = next(iter(test_loader))
    xb = xb.to(device)
    preds = model(xb).cpu().numpy()
    truths = yb.numpy()

    for i in range(5):
        print(f"Pred: ({preds[i][0]:.2f}, {preds[i][1]:.2f}) | True: ({truths[i][0]:.2f}, {truths[i][1]:.2f})")
