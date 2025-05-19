import copy
import numpy as np
import os
import random
import torch
import torch.nn as nn  # For nn.Module, Linear, etc.
from math import floor, ceil  # For ceil function

from torch.nn.functional import mse_loss
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torchinfo import summary
from tqdm import tqdm
from models import MLP, MultiLayerModel  # Assuming you have these model classes
from dataset_mapping import LamaHDataset
# from dataset_complex import LamaHDataset # Example of another dataset
import torch_optimizer as optim  # Using AdaHessian optimizer (Note: Adam is actively used)

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
        "architecture": None,  # To be set below
        "num_layers": None,  # To be set below
        "hidden_channels": 128,
        "param_sharing": False,
        "edge_orientation": None,  # To be set below
        "adjacency_type": "all",  # Default adjacency type
    },
    "training": {
        "num_epochs": 100,  # Number of epochs updated
        "batch_size": 16,  # Batch size updated
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "random_seed": 784,
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


# Get edge weights (Note: this function is defined but not used by construct_model in this script version)
def get_edge_weights(edge_attr):
    return edge_attr[:, :]


# Construct the model
def construct_model(hparams_config, dataset_obj):  # Renamed for clarity
    # edge_weights = get_edge_weights(dataset_obj.edge_attr) # This line was present but edge_weights not used below
    model_arch = hparams_config["model"]["architecture"]

    common_args = {
        "in_channels": hparams_config["data"]["window_size"] * (1 + len(dataset_obj.MET_COLS)),  # type: ignore
        "hidden_channels": hparams_config["model"]["hidden_channels"],
        "num_hidden": hparams_config["model"]["num_layers"],
        "param_sharing": hparams_config["model"]["param_sharing"]
    }

    if model_arch == "MLP":
        return MLP(**common_args)
    elif model_arch == "UpdateUWithAttention":  # Assuming MultiLayerModel is UpdateUWithAttention
        return MultiLayerModel(
            **common_args,
            edge_orientation=hparams_config["model"]["edge_orientation"]
        )
    raise ValueError(f"Unknown model architecture: {model_arch}")


# Load the dataset
def load_dataset(path, hparams_config, split):  # Renamed for clarity
    if split == "train":
        years_data = hparams_config["training"]["train_years"]
    elif split == "test":
        years_data = [2016, 2017]  # Example test years
    else:
        raise ValueError(f"Unknown dataset split: {split}")
    return LamaHDataset(path,  # type: ignore
                        years=years_data,  # type: ignore
                        root_gauge_id=hparams_config["data"]["root_gauge_id"],  # type: ignore
                        rewire_graph=hparams_config["data"]["rewire_graph"],  # type: ignore
                        window_size=hparams_config["data"]["window_size"],  # type: ignore
                        stride_length=hparams_config["data"]["stride_length"],  # type: ignore
                        lead_time=hparams_config["data"]["lead_time"],  # type: ignore
                        normalized=hparams_config["data"]["normalized"])  # type: ignore


# Training step
def train_step(model, train_loader, criterion, optimizer, device, reset_running_loss_after=10):
    model.train()
    epoch_train_loss = 0.0
    running_loss_display = 0.0
    processed_batches_count = 0

    with tqdm(train_loader, desc="Training") as pbar:
        for original_data, _ in pbar:  # Train using only original data component
            original_data = original_data.to(device)

            optimizer.zero_grad()
            # Flipping edge_index: [0] swaps rows, effectively reversing edge direction for directed graphs.
            original_edge_index_flipped = torch.flip(original_data.edge_index, [0])

            predictions = model(original_data.x, original_edge_index_flipped, original_data.edge_attr)
            loss = criterion(predictions, original_data.y)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * original_data.num_graphs

            running_loss_display += loss.item()
            processed_batches_count += 1
            if processed_batches_count % reset_running_loss_after == 0:
                avg_running_loss = running_loss_display / reset_running_loss_after
                pbar.set_postfix({"loss": f"{avg_running_loss:.4f}"})  # type: ignore
                running_loss_display = 0.0

        avg_epoch_loss = epoch_train_loss / len(train_loader.dataset)  # type: ignore
        pbar.set_postfix({"epoch_avg_loss": f"{avg_epoch_loss:.4f}"})  # type: ignore

    return avg_epoch_loss


# Validation step
def val_step(model, val_loader, criterion, device, gauge_mapping, save_identifier, perturbed=False):
    """
    Validation step: calculates error for specified gauge_mapping nodes in each subgraph
    and saves predictions, targets, and input data.
    Args:
        model: The neural network model.
        val_loader: DataLoader for the validation data.
        criterion: Loss function.
        device: Computation device ('cuda' or 'cpu').
        gauge_mapping: Dictionary mapping original gauge IDs to new node indices.
        save_identifier: String identifier for naming saved files.
        perturbed: Boolean, True if using perturbed data, False for original.
    """
    model.eval()
    total_loss_sum = 0.0
    processed_samples_count = 0
    all_predictions_filtered_list = []
    all_targets_filtered_list = []
    all_inputs_list = []
    all_y_full_list = []

    gauge_node_indices = list(gauge_mapping.values())

    desc_str = f"Validating ({'Perturbed' if perturbed else 'Original'}) on {save_identifier}"
    with torch.no_grad():
        with tqdm(val_loader, desc=desc_str) as pbar:
            for original_data, perturbed_data_batch in pbar:
                current_data = perturbed_data_batch if perturbed else original_data
                current_data = current_data.to(device)

                # Flipping edge_index for validation as well
                current_edge_index_flipped = torch.flip(current_data.edge_index, [0])
                predictions = model(current_data.x, current_edge_index_flipped, current_data.edge_attr)

                batch_preds_filt_list = []
                batch_targets_filt_list = []
                batch_inputs_subgraph_list = []
                batch_y_subgraph_list = []

                for i in range(current_data.num_graphs):
                    node_start_idx = current_data.ptr[i]  # type: ignore
                    node_end_idx = current_data.ptr[i + 1]  # type: ignore

                    subgraph_predictions = predictions[node_start_idx:node_end_idx]
                    subgraph_targets_full = current_data.y[node_start_idx:node_end_idx]  # Full targets for the subgraph
                    subgraph_inputs_all = current_data.x[node_start_idx:node_end_idx]  # All inputs for the subgraph

                    subgraph_predictions_filtered = subgraph_predictions[gauge_node_indices]
                    subgraph_targets_filtered = subgraph_targets_full[gauge_node_indices]

                    batch_preds_filt_list.append(subgraph_predictions_filtered)
                    batch_targets_filt_list.append(subgraph_targets_filtered)
                    batch_inputs_subgraph_list.append(subgraph_inputs_all)
                    batch_y_subgraph_list.append(subgraph_targets_full)

                batch_predictions_cat = torch.cat(batch_preds_filt_list, dim=0)
                batch_targets_cat = torch.cat(batch_targets_filt_list, dim=0)
                batch_inputs_cat = torch.cat(batch_inputs_subgraph_list, dim=0)  # type: ignore
                batch_y_cat = torch.cat(batch_y_subgraph_list, dim=0)  # type: ignore

                loss = criterion(batch_predictions_cat, batch_targets_cat)
                total_loss_sum += loss.item() * batch_predictions_cat.size(0)
                processed_samples_count += batch_predictions_cat.size(0)

                all_predictions_filtered_list.append(batch_predictions_cat.cpu())
                all_targets_filtered_list.append(batch_targets_cat.cpu())
                all_inputs_list.append(batch_inputs_cat.cpu())
                all_y_full_list.append(batch_y_cat.cpu())

                if processed_samples_count > 0:
                    pbar.set_postfix({"val_loss": f"{(total_loss_sum / processed_samples_count):.4f}"})  # type: ignore

    avg_val_loss = total_loss_sum / processed_samples_count if processed_samples_count > 0 else 0.0

    final_predictions_filtered = torch.cat(all_predictions_filtered_list, dim=0)
    final_targets_filtered = torch.cat(all_targets_filtered_list, dim=0)
    final_inputs_all_nodes = torch.cat(all_inputs_list, dim=0)
    final_y_all_nodes = torch.cat(all_y_full_list, dim=0)

    data_type_suffix = "perturbed" if perturbed else "original"
    # Updated filename
    filename = f"my_last227_model_validation_data_{save_identifier}_{data_type_suffix}.pt"
    torch.save({
        "pred_filtered": final_predictions_filtered,
        "target_filtered": final_targets_filtered,
        "input_all_nodes": final_inputs_all_nodes,
        "y_all_nodes": final_y_all_nodes
    }, filename)
    print(f"Saved validation data to {filename}")

    return avg_val_loss


# Define interestingness score function (if used for analysis)
def interestingness_score(batch, dataset, device):
    mean_val = dataset.mean[:, None, 0].repeat(batch.num_graphs, 1).to(device)
    std_val = dataset.std[:, None, 0].repeat(batch.num_graphs, 1).to(device)
    unnormalized_discharge = mean_val + std_val * batch.x[:, :, 0]

    assert unnormalized_discharge.min() >= 0.0, "Negative discharge detected!"
    comparable_discharge = unnormalized_discharge / mean_val

    gradient_first_element = torch.gradient(comparable_discharge, dim=-1)[0]  # type: ignore
    mean_central_diff = gradient_first_element.mean()
    trapezoid_integral = torch.trapezoid(comparable_discharge, dim=-1)

    score = 1e3 * (mean_central_diff ** 2) * trapezoid_integral
    assert not trapezoid_integral.isinf().any(), "Infinite value in trapezoid integral!"
    assert not trapezoid_integral.isnan().any(), "NaN value in trapezoid integral!"
    # print(f"Score min: {score.min().item()}, max: {score.max().item()}, mean: {score.mean().item()}")
    return score.unsqueeze(-1)


# Define NSE calculation function (if used for analysis)
def evaluate_nse(model, dataset, val_dataset_subset, batch_size_arg=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    mean_val = dataset.mean[:, [0]].to(device)
    std_squared_val = dataset.std[:, [0]].square().to(device)

    total_weighted_model_error = 0.0
    total_weighted_mean_error = 0.0

    data_loader = DataLoader(val_dataset_subset, batch_size=batch_size_arg, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Calculating NSE"):
            batch = batch.to(device)
            # Assuming edge_index should also be flipped here if consistent with train/val steps
            edge_index_eval = torch.flip(batch.edge_index, [0])
            predictions = model(batch.x, edge_index_eval, batch.edge_attr)

            model_mse_per_sample = mse_loss(predictions, batch.y, reduction="none")

            num_nodes_in_batch = batch.y.size(0)
            num_graphs = num_nodes_in_batch // mean_val.size(0)  # Assuming mean_val.size(0) is nodes_per_graph
            expanded_mean = mean_val.repeat(num_graphs, 1)
            mean_mse_per_sample = mse_loss(expanded_mean, batch.y, reduction="none")

            if dataset.normalized:
                std_sq_expanded = std_squared_val.repeat(num_graphs, 1)
                model_mse_per_sample *= std_sq_expanded
                mean_mse_per_sample *= std_sq_expanded

            current_score = interestingness_score(batch, dataset, device)
            total_weighted_model_error += (current_score * model_mse_per_sample).sum().item()
            total_weighted_mean_error += (current_score * mean_mse_per_sample).sum().item()

    if total_weighted_mean_error == 0:
        print("Warning: Total weighted mean error is zero, NSE is undefined.")
        return -float('inf')
    weighted_nse = 1 - (total_weighted_model_error / total_weighted_mean_error)
    print(f"Final Weighted NSE: {weighted_nse}")
    return weighted_nse


# Main training loop
def train(model, dataset_obj, hparams_config):  # Renamed for clarity
    # Gauge mapping: original_gauge_id to new_node_index
    gauge_mapping_config = {
        214: 148, 215: 149, 216: 150, 218: 152,
        225: 158, 227: 160, 228: 161, 234: 166,
        323: 242, 351: 262, 372: 274, 373: 275,
        399: 288,
    }
    # Provide example input structure for torchinfo.summary
    # Adjust dimensions based on actual dataset and model input expectations
    example_batch_size = hparams_config["training"]["batch_size"]
    # Assuming a fixed number of nodes per graph for summary, e.g., 358 as seen in NSE comments
    # This might need adjustment if graphs are of variable size.
    # For now, we might have to skip summary if dynamic shapes are too complex for a static example.
    # print(summary(model, depth=2)) # Potentially complex to give generic input_data here

    holdout_size = hparams_config["training"]["holdout_size"]
    dataset_length = len(dataset_obj)
    holdout_length = ceil(holdout_size * dataset_length)
    train_length = dataset_length - holdout_length

    train_dataset, val_dataset = random_split(dataset_obj,
                                              [train_length, holdout_length],
                                              generator=torch.Generator().manual_seed(
                                                  hparams_config["training"]["random_seed"]))  # type: ignore

    train_loader = DataLoader(train_dataset, batch_size=hparams_config["training"]["batch_size"], shuffle=True,
                              drop_last=True, num_workers=2, pin_memory=True)  # type: ignore
    val_loader = DataLoader(val_dataset, batch_size=hparams_config["training"]["batch_size"], shuffle=False,
                            drop_last=True, num_workers=2, pin_memory=True)  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    criterion = mse_loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hparams_config["training"]["learning_rate"],
                                 weight_decay=hparams_config["training"]["weight_decay"])
    model = model.to(device)

    training_history = {"train_loss": [], "val_loss_original": [], "val_loss_perturbed": [], "best_model_params": None}
    min_validation_loss = float("inf")

    for epoch in range(hparams_config["training"]["num_epochs"]):
        current_train_loss = train_step(model, train_loader, criterion, optimizer, device)

        # Identifier for saving validation files, can include epoch or other details
        save_id = f"{hparams_config['model']['architecture']}_{hparams_config['model']['edge_orientation']}_e{epoch + 1}"

        val_loss_orig = val_step(model, val_loader, criterion, device,
                                 gauge_mapping_config,
                                 save_identifier=save_id,
                                 perturbed=False)
        val_loss_pert = val_step(model, val_loader, criterion, device,
                                 gauge_mapping_config,
                                 save_identifier=save_id,
                                 perturbed=True)

        training_history["train_loss"].append(current_train_loss)
        training_history["val_loss_original"].append(val_loss_orig)
        training_history["val_loss_perturbed"].append(val_loss_pert)

        print(f"[Epoch {epoch + 1}/{hparams_config['training']['num_epochs']}] "
              f"Train Loss: {current_train_loss:.4f} | "
              f"Val Original Loss: {val_loss_orig:.4f} | "
              f"Val Perturbed Loss: {val_loss_pert:.4f}")

        if val_loss_orig < min_validation_loss:
            min_validation_loss = val_loss_orig
            training_history["best_model_params"] = copy.deepcopy(model.state_dict())
            print(f"New best model saved at epoch {epoch + 1} with validation loss: {min_validation_loss:.4f}")

    return training_history


# Save training history and model checkpoint
def save_checkpoint(history_data, hparams_data, checkpoint_filename, checkpoint_dir="./runs"):
    safe_checkpoint_dir = checkpoint_dir.rstrip("/")
    os.makedirs(safe_checkpoint_dir, exist_ok=True)
    output_path = f"{safe_checkpoint_dir}/{checkpoint_filename}"
    torch.save({
        "history": history_data,
        "hparams": hparams_data
    }, output_path)
    print(f"Saved checkpoint to {output_path}")


# Main execution block
if __name__ == '__main__':
    # Using the hparams defined at the top of the file for this main execution.
    # If different configurations are needed per run, hparams can be modified within the loops.

    cv_folds_config = [(list(range(2000, 2016, 2)), [2016, 2017])]

    for fold_idx, (train_years_fold, test_years_fold) in enumerate(cv_folds_config):
        for architecture_config in ["UpdateUWithAttention", "MLP"]:
            for edge_orientation_config in ["upstream", "downstream", "bidirectional"]:
                print(
                    f"\n--- Training Fold {fold_idx + 1} | Arch: {architecture_config} | Edge Orient: {edge_orientation_config} ---")

                current_run_hparams = copy.deepcopy(hparams)  # Work with a copy for this run
                current_run_hparams["training"]["train_years"] = train_years_fold
                current_run_hparams["model"]["architecture"] = architecture_config
                current_run_hparams["model"]["edge_orientation"] = edge_orientation_config

                dataset_instance = load_dataset(DATASET_PATH, current_run_hparams, split="train")

                # Set num_layers, e.g., from dataset or fixed
                if hasattr(dataset_instance, 'longest_path') and callable(dataset_instance.longest_path):
                    current_run_hparams["model"]["num_layers"] = dataset_instance.longest_path()
                else:
                    # Fallback if longest_path is not available or if a fixed number is preferred
                    # current_run_hparams["model"]["num_layers"] = 10 # Example fixed value
                    if current_run_hparams["model"]["num_layers"] is None:
                        raise ValueError("num_layers is not set and cannot be derived from dataset.")

                ensure_reproducibility(current_run_hparams["training"]["random_seed"])

                print(f"{current_run_hparams['model']['num_layers']} layers used for model.")

                model_instance = construct_model(current_run_hparams, dataset_instance)
                training_run_history = train(model_instance, dataset_instance, current_run_hparams)

                checkpoint_filename = f"{architecture_config}_{edge_orientation_config}_fold{fold_idx}.run"
                save_checkpoint(training_run_history, current_run_hparams, checkpoint_filename,
                                directory=CHECKPOINT_PATH)