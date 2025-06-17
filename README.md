# Subpixel Energy Map Regression

This project generates synthetic energy maps with subpixel labels and provides tools for training and evaluating regression models (e.g., CNNs) to predict subpixel positions from energy maps.

## Project Structure

- `cnn_regressor.py` — CNN model for regression tasks.
- `data_generation.py` — Script to generate synthetic energy maps and save them as CSV.
- `energy_maps_with_labels.csv` — Dataset: energy maps (flattened) with true subpixel positions.
- `saved_results/` — Output figures and screenshots.
- `docker/` — Docker setup for reproducible environments.
  - `Dockerfile` — Build instructions for the container.
  - `build_image.sh` — Script to build the Docker image.
  - `create_container.sh` — Script to run the Docker container.

## Getting Started

### 1. Using Docker

Build the Docker image:

```sh
cd docker
./build_image.sh
```

Run the container:

```sh
./create_container.sh
```

### 2. Generate Data

To generate a new dataset:

```sh
python3 data_generation.py
```

This will create or overwrite `energy_maps_with_labels.csv`.

## Dataset Format

- Each row in `energy_maps_with_labels.csv` contains a flattened energy map followed by `x_true` and `y_true` columns (the subpixel positions).

### 3. Train or Evaluate Model

Edit and run `cnn_regressor.py` to train or test your regression model.



## Requirements
- See Dockerfile for all dependencies (numpy, torch, pandas, scikit-learn, matplotlib, opencv, etc.)

## License

N/A

## Acknowledgments

N/A