import pandas as pd
import numpy as np
import xarray as xr
import torch
from torch.utils.data import TensorDataset, DataLoader
from models import knn_interpolate
import time

from configs import KNN_BATCH_SIZE, KNN_K

device = "cuda" if torch.cuda.is_available() else "cpu"

##################
### Load data ###
##################

train_data = torch.load("data/train_flux_tensor.pt", weights_only = False)
test_data = torch.load("data/test_flux_tensor.pt", weights_only = False)
print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Make torch.dataset
# takes two separate tensors as input
X_train, Y_train = train_data[:, :2].to(device), train_data[:, 2:].to(device)
test_dataset = TensorDataset(test_data[:, :2], test_data[:, 2:])

# Create DataLoader (test only)
test_loader = DataLoader(test_dataset, batch_size = KNN_BATCH_SIZE, shuffle = False)

# Container for k metrics
k_metrics = []

# 1, 2 and 4, 10, 16, 22, 28, 32 (we use 8 numbers for consistency with other experiments that depend on initialisation)
for k in [1, 2] + list(range(4, 36, 6)):
    KNN_K = k

    print(f"Evaluating KNN with k = {KNN_K}...")

    total_test_se = 0.0 
    # absolute error
    total_test_ae = 0.0 
    # number of elements
    total_test_n  = 0 

    num_batches = len(test_loader)

    # ---- START TIMER ----
    if device == "cuda":
            torch.cuda.synchronize()
            t0 = time.perf_counter()

    #### BATCH LOOP Test####
    # takes around 2 min
    for i, (X_batch, Y_batch) in enumerate(test_loader):

            # print(f"Processing batch {i+1}/{num_batches}...")
            
            X_batch = X_batch.to(device)
            # ground truth
            Y_batch = Y_batch.to(device)
            # prediction
            Y_hat = knn_interpolate(X_train, Y_train, X_batch, k = KNN_K, mode = "idw")
            
            # 2D error tensor of shape (batch_size, 2)
            error = Y_hat - Y_batch

            total_test_se += error.pow(2).sum().item()
            total_test_ae += error.abs().sum().item()
            # component-wise error (for consistency): N x 2 columns
            total_test_n += error.numel()

    # final metrics: "per column"
    test_mse = total_test_se / total_test_n
    test_mae = total_test_ae / total_test_n
    test_rmse = test_mse ** 0.5

    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")

    # ---- END TIMER ----
    if device == "cuda":
        torch.cuda.synchronize()
    run_seconds = time.perf_counter() - t0

    # Note: Train metrics are not very meaningful for KNN since it just memorizes the training data, so we skip it

    k_metrics.append({
        "k": k,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_run_minutes": run_seconds / 60.0,
    })

pd.DataFrame(k_metrics).to_csv(f"results/knn_idw_k_metrics.csv", index = False)
