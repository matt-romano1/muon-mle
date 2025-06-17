import numpy as np
import matplotlib.pyplot as plt
import random

# Parameters
grid_size = 8
num_samples = 5  # display 5 examples for demonstration
sigma = 0.7  # standard deviation of the energy spread

def generate_energy_map(true_x, true_y, sigma=0.7):
    """Generate 8x8 energy map centered at subpixel (true_x, true_y)."""
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    xx, yy = np.meshgrid(x, y)
    # 2D Gaussian centered at (true_x, true_y)
    energy_map = np.exp(-((xx - true_x)**2 + (yy - true_y)**2) / (2 * sigma**2))
    return energy_map / energy_map.sum()  # normalize total energy

# Generate and visualize a few samples
samples = []
for _ in range(num_samples):
    x_true = random.uniform(0, grid_size)
    y_true = random.uniform(0, grid_size)
    energy_map = generate_energy_map(x_true, y_true, sigma)
    samples.append((energy_map, (x_true, y_true)))

# Plot results
fig, axs = plt.subplots(1, num_samples, figsize=(15, 3))
for i, (emap, (x, y)) in enumerate(samples):
    axs[i].imshow(emap, cmap='hot', interpolation='nearest')
    axs[i].set_title(f"x={x:.2f}, y={y:.2f}")
    axs[i].axis('off')
plt.tight_layout()
plt.show()




import pandas as pd

# Generate larger dataset for training
num_samples = 10000
energy_maps = []
x_labels = []
y_labels = []

for _ in range(num_samples):
    x_true = random.uniform(0, grid_size)
    y_true = random.uniform(0, grid_size)
    energy_map = generate_energy_map(x_true, y_true, sigma)
    energy_maps.append(energy_map.flatten())  # Flatten to store as CSV
    x_labels.append(x_true)
    y_labels.append(y_true)

# Convert to DataFrame and save
df = pd.DataFrame(energy_maps)
df['x_true'] = x_labels
df['y_true'] = y_labels

# Save to file
output_path = "energy_maps_with_labels.csv"
df.to_csv(output_path, index=False)