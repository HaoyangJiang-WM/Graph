import copy
import numpy as np
import os
import random
import torch
from torch.nn.functional import mse_loss
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torchinfo import summary
from tqdm import tqdm
from models import MLP, MultiLayerModel  # Assuming you have these model classes
# from dataset import LamaHDataset
from dataset_initial_bidi import LamaHDataset
# from dataset_end_bi import LamaHDataset
import torch_optimizer as optim  # Using AdaHessian optimizer (Note: Adam is used in the active train function)

# Global hyperparameter settings
hparams = {
    "data": {
        "root_gauge_id": 399,
        "rewire_graph": True,
        "window_size": 24,
        "stride_length": 1,
        "lead_time": 6,
        "normalized": True,
    },
    "model": {
        "architecture": None,  # set below
        "num_layers": None,  # set below
        "hidden_channels": 128,
        "param_sharing": False,
        "edge_orientation": None,  # set below
        "adjacency_type": "all",
    },
    "training": {
        "num_epochs": 200,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "random_seed": 748,
        "train_years": None,
        "holdout_size": 1 / 5,
        "grad_clip": 1.0,  # Gradient clipping threshold
    }
}

DATASET_PATH = "./LamaH-CE"
CHECKPOINT_PATH = "./checkpoint"


# Ensure reproducibility of training
def ensure_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Get edge weights
def get_edge_weights(edge_attr):
    return edge_attr[:, :]


# Construct the model
def construct_model(hparams, dataset):
    edge_weights = get_edge_weights(dataset.edge_attr)
    model_arch = hparams["model"]["architecture"]

    if model_arch == "MLP":
        return MLP(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                   hidden_channels=hparams["model"]["hidden_channels"],
                   num_hidden=hparams["model"]["num_layers"],
                   param_sharing=hparams["model"]["param_sharing"])
    elif model_arch == "UpdateUWithAttention":
        return MultiLayerModel(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                               hidden_channels=hparams["model"]["hidden_channels"],
                               num_hidden=hparams["model"]["num_layers"],
                               param_sharing=hparams["model"]["param_sharing"],
                               edge_orientation=hparams["model"]["edge_orientation"])
    raise ValueError("unknown model architecture", model_arch)


# Load the dataset
def load_dataset(path, hparams, split):
    if split == "train":
        years = hparams["training"]["train_years"]
    elif split == "test":
        years = [2016, 2017]
    else:
        raise ValueError("unknown split", split)
    return LamaHDataset(path,
                        years=years,
                        root_gauge_id=hparams["data"]["root_gauge_id"],
                        rewire_graph=hparams["data"]["rewire_graph"],
                        window_size=hparams["data"]["window_size"],
                        stride_length=hparams["data"]["stride_length"],
                        lead_time=hparams["data"]["lead_time"],
                        normalized=hparams["data"]["normalized"])


# Single training step
def train_step(model, train_loader, criterion, optimizer, device, reset_running_loss_after=10, accumulation_steps=4,
               grad_clip=None):
    model.train()
    train_loss = 0.0
    running_loss = 0.0
    running_counter = 1
    optimizer.zero_grad()  # Clear gradients at the beginning
    with tqdm(train_loader, desc="Training") as pbar:
        print("Delta t before training:", model.delta_t)
        for i, batch in enumerate(pbar):
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            # batch_edge_index = torch.flip(batch.edge_index, [0])  # [0] means swapping along the 0th dimension
            # pred = model(batch.x, batch_edge_index, batch.edge_attr)
            # # Reverse edges for bidirectional graph
            # reverse_edge_index = torch.flip(batch.edge_index, [0])
            # batch_edge_index = torch.cat([batch.edge_index, reverse_edge_index], dim=1)
            # batch_edge_attr = torch.cat([batch.edge_attr, batch.edge_attr], dim=0)
            # pred = model(batch.x, batch_edge_index, batch_edge_attr)

            loss = criterion(pred, batch.y)  # Directly use batch.y as the target value for loss calculation
            # loss = criterion(pred[batch.mask], batch.y)
            # Check if the loss is NaN or Inf

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("NaN or Inf detected in loss, skipping this batch.")
                optimizer.zero_grad()  # Skip the current batch, clear gradients
                continue
            # Backpropagation and gradient update
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()

            ##################################################
            # Gradient clipping to prevent exploding gradients
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Gradient accumulation, update parameters every accumulation_steps
            if (i + 1) % accumulation_steps == 0 or i == len(train_loader) - 1:
                optimizer.step()  # Update model parameters
                optimizer.zero_grad()  # Clear accumulated gradients

            train_loss += loss.item() * batch.num_graphs / len(train_loader.dataset)
            running_loss += loss.item() / reset_running_loss_after
            running_counter += 1
            if running_counter >= reset_running_loss_after:
                pbar.set_postfix({"loss": running_loss})
                running_counter = 1
                running_loss = 0.0
    return train_loss


def val_step(model, val_loader, criterion, dataset, device, reset_running_loss_after=10,
             save_path="sample_losses.npy"):
    model.eval()  # Switch model to evaluation mode
    val_loss = 0
    total_samples = 0  # Total number of samples
    sample_losses = []  # Used to store the loss for each sample

    # Get mean and standard deviation for denormalizing data
    mean = dataset.mean[:, [0]].to(device)  # Mean of the target variable
    std_squared = dataset.std[:, [0]].square().to(device)  # Square of the standard deviation of the target variable

    with torch.no_grad():  # Disable gradient calculation to speed up validation
        with tqdm(val_loader, desc="Validating") as pbar:
            for batch in pbar:
                batch = batch.to(device)  # Move batch to device
                pred = model(batch.x, batch.edge_index, batch.edge_attr)
                # batch_edge_index = torch.flip(batch.edge_index, [0])  # [0] means swapping along the 0th dimension
                # pred = model(batch.x, batch_edge_index, batch.edge_attr)

                # # Reverse edges for bidirectional graph
                # reverse_edge_index = torch.flip(batch.edge_index, [0])
                # batch_edge_index = torch.cat([batch.edge_index, reverse_edge_index], dim=1)
                # batch_edge_attr = torch.cat([batch.edge_attr, batch.edge_attr], dim=0)
                # pred = model(batch.x, batch_edge_index, batch_edge_attr)

                # Calculate loss per sample (no reduction)
                # loss_per_sample = criterion(pred[batch.mask], batch.y)
                loss_per_sample = criterion(pred, batch.y)
                # loss_per_sample = criterion(pred, batch.y, reduction='none')
                # Check for NaN or Inf loss
                if torch.isnan(loss_per_sample).any() or torch.isinf(loss_per_sample).any():
                    print("NaN or Inf detected in validation loss, skipping this batch.")
                    continue
                val_loss += loss_per_sample.item() * batch.num_graphs / len(val_loader.dataset)
    return val_loss, 1 # Returning 1 as a placeholder for nse_score, actual NSE calculation is separate


from torch_geometric.loader import DataLoader


# Define interestingness score function
def interestingness_score(batch, dataset, device):
    mean = dataset.mean[:, None, 0].repeat(batch.num_graphs, 1).to(device)
    std = dataset.std[:, None, 0].repeat(batch.num_graphs, 1).to(device)
    unnormalized_discharge = mean + std * batch.x[:, :, 0]

    # Check if unnormalized_discharge has outliers
    assert unnormalized_discharge.min() >= 0.0, "Negative discharge detected!"

    comparable_discharge = unnormalized_discharge / mean

    mean_central_diff = torch.gradient(comparable_discharge, dim=-1)[0].mean()
    trapezoid_integral = torch.trapezoid(comparable_discharge, dim=-1)

    score = 1e3 * (mean_central_diff ** 2) * trapezoid_integral

    # Ensure integral result has no NaN or Inf
    assert not trapezoid_integral.isinf().any(), "Infinite value in trapezoid integral!"
    assert not trapezoid_integral.isnan().any(), "NaN value in trapezoid integral!"

    # Print basic statistics of the interestingness score
    print(f"Score min: {score.min().item()}, max: {score.max().item()}, mean: {score.mean().item()}")

    return score.unsqueeze(-1)


# Define NSE calculation function
def evaluate_nse(model, dataset, val_dataset, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Mean and std squared for un-normalizing if necessary
    mean = dataset.mean[:, [0]].to(device)  # Shape: (358, 1)
    std_squared = dataset.std[:, [0]].square().to(device)

    weighted_model_error = 0.0
    weighted_mean_error = 0.0

    # Use DataLoader for batch processing
    data_loader = DataLoader(val_dataset[:-4], batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Testing"):
            batch = batch.to(device)

            # Forward pass
            print(batch.x.shape)
            pred = model(batch.x, batch.edge_index, batch.edge_attr)

            # Compute per-sample MSE: (y_true - y_pred)^2
            model_mse = mse_loss(pred, batch.y, reduction="none")

            # Calculate the number of samples in the current batch
            current_batch_size = batch.y.size(0) # Example: 5728

            # Expand the size of mean to match batch.y
            expanded_mean = mean.repeat(current_batch_size // mean.size(0), 1)  # Expand using repeat

            # Compute mean MSE: (y_true - y_mean)^2
            mean_mse = mse_loss(expanded_mean, batch.y, reduction="none")

            # If the dataset is normalized, mse calculation needs to be readjusted
            if dataset.normalized:
                # Expand std_squared to fit the shape of the batch
                std_squared_expanded = std_squared.repeat(batch.x.size(0) // std_squared.size(0), 1)

                # Check if std_squared_expanded matches correctly
                print(
                    f"std_squared_expanded shape: {std_squared_expanded.shape}, model_mse shape: {model_mse.shape}, mean_mse shape: {mean_mse.shape}")

                model_mse *= std_squared_expanded  # Remove the effect of normalization
                mean_mse *= std_squared_expanded  # Also remove normalization effect

            # Calculate interestingness_score and weight MSE
            score = interestingness_score(batch, dataset, device)  # Call the previous score calculation method

            # Accumulate weighted model and mean MSE
            weighted_model_error += (score * model_mse).sum().item()
            weighted_mean_error += (score * mean_mse).sum().item()

            # Print statistics of MSE values
            print(
                f"Model MSE min: {model_mse.min().item()}, max: {model_mse.max().item()}, mean: {model_mse.mean().item()}")
            print(
                f"Mean MSE min: {mean_mse.min().item()}, max: {mean_mse.max().item()}, mean: {mean_mse.mean().item()}")

    # Print accumulated weighted errors for debugging
    print(f"Weighted Model Error: {weighted_model_error}, Weighted Mean Error: {weighted_mean_error}")

    # Finally calculate weighted NSE
    weighted_nse = 1 - (weighted_model_error / weighted_mean_error)

    # Print the final NSE result
    print(f"Final NSE: {weighted_nse}")

    return weighted_nse


def train(model, dataset, hparams):
    print(summary(model, depth=2))

    holdout_size = hparams["training"]["holdout_size"]
    dataset_length = len(dataset)
    val_size = int(holdout_size * dataset_length)
    train_size = dataset_length - val_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=hparams["training"]["batch_size"], shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams["training"]["batch_size"], shuffle=False, num_workers=0,
                            pin_memory=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = mse_loss  # interestingness_score is no longer used in the loss directly

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hparams["training"]["learning_rate"],
                                 weight_decay=hparams["training"]["weight_decay"])

    # Use learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-6)

    model = model.to(device)
    print("Training on", device)

    history = {"train_loss": [], "val_loss": [], "nse_score": [], "best_model_params": None}
    min_val_loss = float("inf")

    for epoch in range(hparams["training"]["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{hparams['training']['num_epochs']}")
        train_loss = train_step(model, train_loader, criterion, optimizer, device, accumulation_steps=4,
                                grad_clip=hparams["training"]["grad_clip"])
        val_loss, nse_score_placeholder = val_step(model, val_loader, criterion, dataset, device) # nse_score_placeholder is not the actual NSE

        # Actual NSE calculation can be done here if needed per epoch, or at the end
        # For now, using the placeholder from val_step which is 1
        actual_nse_score = nse_score_placeholder # Replace with evaluate_nse if needed per epoch, be mindful of computational cost
        # If you want to calculate NSE each epoch (can be slow):
        # actual_nse_score = evaluate_nse(model, dataset, val_dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["nse_score"].append(actual_nse_score) # Storing the placeholder or actual calculated NSE

        print(f"[Epoch {epoch + 1}/{hparams['training']['num_epochs']}] Train: {train_loss:.4f} | Val: {val_loss:.4f} | NSE (placeholder): {actual_nse_score:.4f}")

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            history["best_model_params"] = copy.deepcopy(model.state_dict())

        # Adjust learning rate
        scheduler.step(val_loss)
        # test_nse = evaluate_nse(model, dataset, val_dataset) # Example of calling NSE evaluation
        # print(test_nse)

    return history


# Save training history and checkpoint
def save_checkpoint(history, hparams, filename, directory="./runs"):
    directory = directory.rstrip("/")
    os.makedirs(directory, exist_ok=True)
    out_path = f"{directory}/{filename}"
    torch.save({
        "history": history,
        "hparams": hparams
    }, out_path)
    print("Saved checkpoint", out_path)


# Main training program
if __name__ == '__main__':
    for fold_id, (train_years, test_years) in enumerate([(list(range(2000, 2016, 2)), [2016, 2017])]):

        for architecture in ["UpdateUWithAttention", "MLP"]:
            for edge_orientation in ["downstream"]:
                for num_layers in range(15, 30, 4):
                    hparams["training"]["train_years"] = train_years
                    dataset = load_dataset(DATASET_PATH, hparams, split="train")

                    hparams["model"]["architecture"] = architecture
                    hparams["model"]["edge_orientation"] = edge_orientation
                    hparams["model"]["num_layers"] = num_layers
                    ensure_reproducibility(hparams["training"]["random_seed"])

                    print(hparams["model"]["num_layers"], "layers used")
                    model = construct_model(hparams, dataset)
                    history = train(model, dataset, hparams)

                    chkpt_name = f"{architecture}_{edge_orientation}_{fold_id}.run"
                    save_checkpoint(history, hparams, chkpt_name, directory=CHECKPOINT_PATH)