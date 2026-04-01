N_RUNS = 3
LR = 1e-3
WEIGHT_DECAY = 1e-7
BATCH_SIZE = 1024
EPOCHS = 15

# For ablations: try half and double LR
LR_RANGE = [0.0005, 0.001, 0.002]

# batch size that works with GPU
KNN_BATCH_SIZE = 2048
KNN_K = 16
