import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

# Parameters of detector
n_pix = 8
min_edge = -7.5
max_edge = +7.5

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


def generate_energy_map(true_x, true_y, sigma=3):
    """Generate 8x8 energy map centered at subpixel (true_x, true_y)."""
    xx, yy = np.meshgrid(centers, centers, indexing="xy")
    # 2D Gaussian centered at (true_x, true_y)
    energy_map = np.exp(-((xx - true_x)**2 + (yy - true_y)
                        ** 2) / (2 * sigma**2))
    return energy_map  # / energy_map.sum()  # normalize total energy


# Generate and visualize a few samples
num_samples = 10000
energy_maps = []
energy_maps_flatten = []
x_labels = []
y_labels = []
for _ in range(num_samples):
    x_true = random.uniform(min_edge, max_edge)
    y_true = random.uniform(min_edge, max_edge)
    energy_map = generate_energy_map(x_true, y_true, sigma_phys)
    energy_maps.append((energy_map, (x_true, y_true)))
    energy_maps_flatten.append(energy_map.flatten())  # Flatten to store as CSV
    x_labels.append(x_true)
    y_labels.append(y_true)

# Save to file
output_path = "energy_maps_with_labels.csv"
df = pd.DataFrame(energy_maps_flatten)
df['x_true'] = x_labels
df['y_true'] = y_labels
df.to_csv(output_path, index=False)

# Plot results in a 5x5 grid
fig, axs = plt.subplots(5, 5, figsize=(10, 10))
for i, (emap, (x, y)) in enumerate(energy_maps[:25]):
    ax = axs[i // 5, i % 5]
    ax.imshow(emap, cmap='hot', interpolation='nearest')
    ax.set_title(f"x={x:.2f}, y={y:.2f}")
    ax.axis('off')
plt.tight_layout()
plt.show()
