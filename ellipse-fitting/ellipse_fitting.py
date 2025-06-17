import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import EllipseModel, ransac
from matplotlib.patches import Ellipse

# Step 1: Generate synthetic ellipse data
np.random.seed(0)

# True ellipse parameters
xc_true, yc_true = 50, 75  # center
a_true, b_true = 40, 20    # semi-axes
theta_true = np.deg2rad(30)  # rotation angle in radians

# Generate points along the ellipse
t = np.linspace(0, 2 * np.pi, 100)
x = a_true * np.cos(t)
y = b_true * np.sin(t)

# Rotate and translate the ellipse
x_rot = xc_true + x * np.cos(theta_true) - y * np.sin(theta_true)
y_rot = yc_true + x * np.sin(theta_true) + y * np.cos(theta_true)

ellipse_points = np.column_stack((x_rot, y_rot))

# Step 2: Add Gaussian noise
noisy_points = ellipse_points + np.random.normal(scale=1.5, size=ellipse_points.shape)

# Step 3: Add outliers
n_outliers = 20
outliers = np.random.uniform(low=0, high=120, size=(n_outliers, 2))

# Combine noisy inliers with outliers
all_points = np.vstack((noisy_points, outliers))

# Step 4: Apply RANSAC for robust ellipse fitting
model_ellipse, inliers = ransac(all_points, EllipseModel, min_samples=5,
                                residual_threshold=2.5, max_trials=1000)

# Extract fitted ellipse parameters
xc, yc, a, b, theta = model_ellipse.params

# Prepare for visualization
inlier_points = all_points[inliers]
outlier_points = all_points[~inliers]

# Step 5: Plot the results
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(inlier_points[:, 0], inlier_points[:, 1], color='blue', label='Inliers')
ax.scatter(outlier_points[:, 0], outlier_points[:, 1], color='red', label='Outliers')

# Add fitted ellipse
ellipse_patch = Ellipse((xc, yc), 2 * a, 2 * b, angle=np.degrees(theta),
                        edgecolor='green', facecolor='none', linewidth=2, label='Fitted Ellipse')
ax.add_patch(ellipse_patch)

ax.set_aspect('equal')
ax.set_title("Ellipse Fit with RANSAC")
ax.legend()
plt.tight_layout()
plt.show()


# Given: inlier_points (Nx2 numpy array)
x = inlier_points[:, 0]
y = inlier_points[:, 1]

# Build design matrix for conic section
D = np.column_stack((x**2, x * y, y**2, x, y, np.ones_like(x)))

# Solve D @ coeffs = 0 using least squares
# Note: Using SVD to find solution that minimizes |D @ coeffs| subject to ||coeffs|| = 1
_, _, Vt = np.linalg.svd(D)
conic_coeffs = Vt[-1, :]  # Last row of V^T corresponds to smallest singular value

# The coefficients A, B, C, D, E, F of the ellipse
A, B, C, D, E, F = conic_coeffs
print(f"Fitted Ellipse Equation: {A:.5f} x² + {B:.5f} xy + {C:.5f} y² + {D:.5f} x + {E:.5f} y + {F:.5f} = 0")
