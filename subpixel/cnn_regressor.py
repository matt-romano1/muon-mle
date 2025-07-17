import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
# Load dataset
df = pd.read_csv("energy_maps_with_labels.csv")
X = df.iloc[:, :-2].values.reshape(-1, 1, 8, 8).astype(np.float32)
y = df[["x_true", "y_true"]].values.astype(np.float32)


# Plot samples
num_samples = 5
fig, axs = plt.subplots(1, num_samples, figsize=(15, 3))
for i, (emap, (x_, y_)) in enumerate(zip(np.squeeze(X[:num_samples]), y[:num_samples])):
    axs[i].imshow(emap, cmap='hot', interpolation='nearest')
    axs[i].set_title(f"x={x_:.2f}, y={y_:.2f}")
    axs[i].axis('off')
plt.tight_layout()
plt.show()

# Print out the range of true positions
x_min, x_max = np.min(y[:, 0]), np.max(y[:, 0])
y_min, y_max = np.min(y[:, 1]), np.max(y[:, 1])
print(f"x_true ranges from {x_min:.2f} to {x_max:.2f}")
print(f"y_true ranges from {y_min:.2f} to {y_max:.2f}")

# # normalization
# # --- INPUTS -------------------------------------------------
X = X/X.sum(axis=(2, 3), keepdims=True)


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

############################################ centroid method #########################################
# Parameters of detector
n_pix = 8
min_edge = -25
max_edge = +25

pixel_width = (max_edge - min_edge) / n_pix    # = 15/8 = 1.875
sigma_pixels = 0.7
sigma_phys = sigma_pixels * pixel_width
# 2) start half a pixel in, end half a pixel before the max_edge
centers = np.linspace(
    min_edge + pixel_width/2,
    max_edge - pixel_width/2,
    n_pix,
    dtype=np.float32
)
xx, yy = np.meshgrid(centers, centers, indexing="xy")


energy_maps = X_test.copy()
energy_maps = energy_maps.squeeze(axis=1)


def centroid_positions(energy_maps):
    # energy_maps can be a single (8,8) array or batch (N,8,8)
    E = energy_maps.reshape(-1, 8, 8)    # ensure batch
    # numerator: sum(E * coord) over channels
    x_num = (E * xx).sum(axis=(1, 2))
    y_num = (E * yy).sum(axis=(1, 2))
    E_sum = E.sum(axis=(1, 2)) + 1e-8    # avoid /0
    return np.column_stack((x_num/E_sum, y_num/E_sum))   # shape (N,2)


y_c = centroid_positions(energy_maps)
# Calculate mean distance error
distances_c = np.linalg.norm(y_test - y_c, axis=1)
mean_distance_error_c = np.mean(distances_c)
print(f"centroid mean distance error: {mean_distance_error_c:.4f}")


############################################ mle method ############################################
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
            nn.Dropout(0.4),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),
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
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

train_losses, val_losses, val_mae, val_dist_errors = [], [], [], []

for epoch in range(30):
    model.train()
    total_train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * xb.size(0)
    avg_train_loss = total_train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    val_loss, mae, dist_err = 0.0, 0.0, 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_loss += criterion(pred, yb).item() * xb.size(0)
            mae += torch.mean(torch.abs(pred - yb), dim=1).sum().item()
            dist_err += torch.norm(pred - yb, dim=1).sum().item()
    val_loss /= len(test_loader.dataset)
    mae /= len(test_loader.dataset)
    dist_err /= len(test_loader.dataset)
    val_losses.append(val_loss)
    val_mae.append(mae)
    val_dist_errors.append(dist_err)
    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, MAE={mae:.3f}, Dist Err={dist_err:.3f}")
    

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.plot(val_mae, label="Val MAE")
plt.plot(val_dist_errors, label="Val Distance Error")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Metric"); plt.show()

