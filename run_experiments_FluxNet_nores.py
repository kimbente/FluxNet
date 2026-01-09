import pandas as pd
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time

from models import FluxNet_nores
from configs import N_RUNS, LR, WEIGHT_DECAY, BATCH_SIZE, EPOCHS

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
train_dataset = TensorDataset(train_data[:, :2], train_data[:, 2:])
test_dataset = TensorDataset(test_data[:, :2], test_data[:, 2:])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

# Container for run metrics
run_metrics = []

#####################
### Eval function ###
#####################

def eval_epoch(model, loader, device):
    model.eval()
    mse_sum = mae_sum = 0.0
    n = 0
    for Xb, Yb in loader:
        Xb = Xb.to(device)
        Yb = Yb.to(device)
        err = model(Xb) - Yb
        mse_sum += (err**2).sum().item()
        mae_sum += err.abs().sum().item()
        n += err.numel()
    mse = mse_sum / n
    mae = mae_sum / n
    rmse = mse ** 0.5
    return mse, mae, rmse

###########
### RUN ###
###########

#### RUN LOOP ####
for n_run in range(1, N_RUNS + 1): 

    print(f"Starting run {n_run}/{N_RUNS}...")

    # ---- START TIMER ----
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Initialize model, optimizer, loss function
    fluxnet_model = FluxNet_nores().to(device)
    optim = torch.optim.AdamW(fluxnet_model.parameters(), lr = LR, weight_decay = WEIGHT_DECAY)
    loss_function = nn.MSELoss()

    train_losses = []

    #### EPOCH LOOP ####
    for epoch in range(1, EPOCHS + 1):

        # print to track
        print(f"Starting epoch {epoch}/{EPOCHS}...")
        
        fluxnet_model.train()
        train_loss_sum = 0.0

        #### BATCH LOOP ####
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optim.zero_grad(set_to_none=True)

            Y_hat = fluxnet_model(X_batch)
            loss = loss_function(Y_hat, Y_batch)

            loss.backward()
            optim.step()

            train_loss_sum += loss.item() * X_batch.size(0)

        #### END BATCH LOOP ####
        epoch_train = train_loss_sum / len(train_loader.dataset)
        train_losses.append(epoch_train)

    #### END EPOCH LOOP ####

    # ---- END TIMER ----
    if device == "cuda":
        torch.cuda.synchronize()
    run_seconds = time.perf_counter() - t0

    # Save first run model
    if n_run == 1:
        # Save trained model
        torch.save(fluxnet_model.state_dict(), "trained_models/fluxnet-nores_trained_20epochs.pth")
        # Save training loss convergence
        pd.DataFrame({"train_loss": train_losses}).to_csv(
        "trained_models/fluxnet-nores_trained_20epochs_loss_convergence.csv", index = False)
    
    #### EVAL ####
    # Calculate training MAE and RMSE, test MAE and RMSE
    train_mse, train_mae, train_rmse = eval_epoch(fluxnet_model, train_loader, device)
    test_mse, test_mae, test_rmse = eval_epoch(fluxnet_model, test_loader, device)

    run_metrics.append({
        "run": n_run,
        "train_mse": train_mse,
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "run_minutes": run_seconds / 60.0,
    })

    # print(f"[Run {n_run:03d}] train_mae = {train_mae:.6f} | test_mae = {test_mae:.6f}")
    print(f"[Run {n_run:03d}] train_rmse = {train_rmse:.6f} | test_rmse = {test_rmse:.6f}")

#### END RUN LOOP ####
# Save run metrics
pd.DataFrame(run_metrics).to_csv("results/fluxnet-nores_runs_metrics.csv", index = False)