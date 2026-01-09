import pandas as pd
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time

from models import ResMLP
from configs import N_RUNS, LR, WEIGHT_DECAY, BATCH_SIZE, EPOCHS, W_DIRECTIONAL_GUIDANCE

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

############################
### Directional Guidance ###
############################

### UNIT VELOCITIES OVER DOMAIN ###
# Actuall learn across full domain since train observations are sparse too
velocity_unit_norm = torch.load("data/directional_guidance/velocity_unit_norm.pt", 
                                     weights_only = False).to(device)

print("velocity_unit_norm shape:", velocity_unit_norm.shape)
# Extract number of velocities as upper bound
N_dg = velocity_unit_norm.shape[0]

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
    resmlp_model = ResMLP().to(device)
    optim = torch.optim.AdamW(resmlp_model.parameters(), lr = LR, weight_decay = WEIGHT_DECAY)
    loss_function = nn.MSELoss()

    train_losses = []

    #### EPOCH LOOP ####
    for epoch in range(1, EPOCHS + 1):
        
        print(f"Starting epoch {epoch}/{EPOCHS}...")
        
        resmlp_model.train()
        train_loss_sum = 0.0

        #### BATCH LOOP ####
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optim.zero_grad(set_to_none = True)

            Y_hat = resmlp_model(X_batch)
            mse_loss = loss_function(Y_hat, Y_batch)

            ### DIRECTIONAL guidance ###

            ### DIRECTIONAL guidance ###
            # Select another batch of random indicies
            idx = torch.randint(0, N_dg, (BATCH_SIZE,), device = device)

            batch_dg = velocity_unit_norm[idx]
            # Input locations
            batch_dg_in = batch_dg[:, [0, 1]].requires_grad_().to(device)
            # "directions" (unit vectors) as outputs
            batch_dg_out = batch_dg[:, [2, 3]].to(device)

            dg_output = resmlp_model(batch_dg_in)
            # cosine similarity per row, then turn into a loss
            cos = F.cosine_similarity(dg_output, batch_dg_out, dim = 1, eps = 1e-8)  # in [-1, 1]
            dg_loss = 1.0 - cos.mean() # 0 = same direction, 2 = opposite

            ### COMBINE LOSSES ###
            # weight and combine
            loss = (1 - W_DIRECTIONAL_GUIDANCE) * mse_loss + W_DIRECTIONAL_GUIDANCE * dg_loss

            ### END DIRECTIONAL guidance ###

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
        torch.save(resmlp_model.state_dict(), "trained_models/resmlp-dir_trained_20epochs.pth")
        # Save training loss convergence
        pd.DataFrame({"train_loss": train_losses}).to_csv(
        "trained_models/resmlp-dir_trained_20epochs_loss_convergence.csv", index = False)
    
    #### EVAL ####
    # Calculate training MAE and RMSE, test MAE and RMSE
    train_mse, train_mae, train_rmse = eval_epoch(resmlp_model, train_loader, device)
    test_mse, test_mae, test_rmse = eval_epoch(resmlp_model, test_loader, device)

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
pd.DataFrame(run_metrics).to_csv("results/resmlp-dir_runs_metrics.csv", index = False)