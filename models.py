import torch
from torch.nn import Module, ModuleList, Linear
import torch.nn as nn
import torch_scatter
import torch.nn.functional as F  # Import F.silu

# Base model class
class BaseModel(Module):
    def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, layerfun, edge_orientation):
        super().__init__()
        self.encoder = Linear(in_channels, hidden_channels)
        self.decoder = Linear(hidden_channels, 1)

        if param_sharing:
            self.layers = ModuleList(num_hidden * [layerfun()])
        else:
            self.layers = ModuleList([layerfun() for _ in range(num_hidden)])

        self.edge_orientation = edge_orientation

    def forward(self, x, edge_index, edge_attr=None):
        x = x.flatten(1)
        x_0 = self.encoder(x)
        x = x_0

        for layer in self.layers:
            x = self.apply_layer(layer, x, x_0, edge_index, edge_attr)
        x = self.decoder(x)
        return x

    def apply_layer(self, layer, x, x_0, edge_index, edge_attr):
        return layer(x, edge_index, edge_attr)


# MLP class for node-level operations
class MLP(BaseModel):
    def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing):
        layer_gen = lambda: Linear(hidden_channels, hidden_channels)
        super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, None)

    def apply_layer(self, layer, x, x_0, edge_index, edge_attr):
        return F.leaky_relu(layer(x), negative_slope=0.01)

# MLP for edge attributes
class MLPForEdgeAttributes(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPForEdgeAttributes, self).__init__()
        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, 1)

    def forward(self, edge_attr):
        # x = F.leaky_relu(self.fc1(edge_attr), negative_slope=0.01)
        x = torch.tanh(self.fc1(edge_attr))
        x = self.fc2(x)
        return x

# Update U using MLP and sparse difference matrices
class UpdateUWithMLP(nn.Module):
    def __init__(self, delta_t, input_dim, edge_attr_dim, output_dim, edge_orientation="upstream"):
        super(UpdateUWithMLP, self).__init__()
        self.delta_t = delta_t
        self.delta_v = nn.Parameter(torch.tensor(0.001))
        self.delta_g = nn.Parameter(torch.tensor(0.5))
        self.num_nodes = None
        self.input_dim = input_dim
        self.hidden_channels = output_dim
        self.edge_attr_dim = edge_attr_dim
        self.edge_orientation = edge_orientation

        self.edge_mlp_x = MLPForEdgeAttributes(edge_attr_dim, hidden_dim=16)
        self.edge_mlp_x_2 = MLPForEdgeAttributes(edge_attr_dim, hidden_dim=16)
        self.edge_mlp_z = MLPForEdgeAttributes(edge_attr_dim, hidden_dim=16)  # Separate MLP for delta_z
        self.weight = nn.Parameter(torch.Tensor(self.hidden_channels, self.hidden_channels))
        self.D2_weight = nn.Parameter(torch.Tensor(self.hidden_channels, self.hidden_channels))
        self.Az_weight = nn.Parameter(torch.Tensor(self.hidden_channels, self.hidden_channels))
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.Az_weight)
        self.norm1 = nn.LayerNorm(self.hidden_channels)  # Normalize first_term
        self.norm2 = nn.LayerNorm(self.hidden_channels)  # Normalize Az_transformed

    def forward(self, u, edge_index, edge_attr):
        # Check if u is a tuple and unpack if necessary
        if isinstance(u, tuple):
            u = u[0]

        if self.num_nodes is None:
            self.num_nodes = edge_index.max().item() + 1

        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, self.edge_attr_dim)
        elif edge_attr.size(1) != self.edge_attr_dim:
            raise ValueError(f"Expected edge_attr to have {self.edge_attr_dim} features, but got {edge_attr.size(1)}")

        # Compute delta_x and delta_z separately
        delta_x = self.edge_mlp_x(edge_attr) + 1e-6
        delta_z = self.edge_mlp_z(edge_attr) + 1e-6
        delta_x_2 = self.edge_mlp_x_2(edge_attr) + 1e-6
        if delta_x.size(0) != edge_index.size(1) or delta_z.size(0) != edge_index.size(1):
            raise RuntimeError(f"delta_x or delta_z size does not match edge_index size")

        # Select appropriate sparse difference matrix based on edge_orientation
        if self.edge_orientation == "downstream":
            D_adjusted, Az = create_forward_difference_matrix(self.num_nodes, edge_index, delta_x, delta_z)
        elif self.edge_orientation == "upstream":
            D_adjusted, Az = create_backward_difference_matrix(self.num_nodes, edge_index, delta_x, delta_z)
        elif self.edge_orientation == "bidirectional":
            D_adjusted, Az = create_central_difference_matrix(self.num_nodes, edge_index, delta_x, delta_z)
        else:
            raise ValueError("Unknown edge_orientation type.")

        # Step 1: Compute first_term
        uw = torch.matmul(u, self.weight)  # Shape: (n, m)
        uw_squared = uw ** 2
        first_term = torch.sparse.mm(D_adjusted, uw_squared)
        first_term = self.norm1(first_term)  # Normalize
        first_term = F.silu(first_term)  # Replace with F.silu

        # Step 2: Compute Az_transformed
        Az_transformed = torch.sparse.mm(Az, u)
        Az_transformed = torch.matmul(Az_transformed, self.Az_weight)
        Az_transformed = self.norm2(Az_transformed)  # Normalize
        Az_transformed = F.silu(Az_transformed) + 1e-6  # Replace with SiLU

        # Step 3: Compute u_next
        u_next = u - self.delta_t * (first_term + self.delta_g * Az_transformed)
        u_next = torch.clamp(u_next, min=-10, max=10)  # Clamp values to prevent gradient explosion
        return u_next

        # (ablation)
        # # Term1: u_diff = D_adjusted * u using sparse matrix multiplication
        # u = torch.tanh(torch.matmul(u, self.weight)) # Shape: (n, m)
        # # u = torch.matmul(u, self.weight)  # Shape: (n, m)
        # u_diff = torch.sparse.mm(D_adjusted, u)
        # # combine first_term and Az_transformed for u_next
        # u_next = u + torch.tanh(self.delta_t * u_diff)
        # return u_next


# MultiLayerModel for applying multiple layers of UpdateUWithMLP
class MultiLayerModel(BaseModel):
    def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation):
        super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, lambda: None, edge_orientation)

        self.delta_t = nn.Parameter(torch.tensor(0.7)) # type: ignore
        self.num_nodes = hidden_channels
        self.input_dim = in_channels
        self.hidden_channels = hidden_channels
        self.edge_attr_dim = 3 # Number of edge attributes

        if param_sharing:
            layer = UpdateUWithMLP(self.delta_t, self.input_dim, self.edge_attr_dim, output_dim=hidden_channels,
                                   edge_orientation=edge_orientation)
            self.layers = ModuleList(num_hidden * [layer])
        else:
            self.layers = ModuleList([UpdateUWithMLP(self.delta_t, self.input_dim, self.edge_attr_dim,
                                                     output_dim=hidden_channels, edge_orientation=edge_orientation) for
                                      _ in range(num_hidden)])

        self.decoder = Linear(hidden_channels, 1)

    def apply_layer(self, layer, x, x_0, edge_index, edge_attr):
        return layer(x, edge_index, edge_attr)


def create_forward_difference_matrix(num_nodes, edge_index, delta_x, delta_z):
    device = delta_x.device # type: ignore
    row = edge_index[0]  # Upstream node
    col = edge_index[1]  # Current node

    delta_x = delta_x.view(-1)
    delta_z = delta_z.view(-1)

    if delta_x.size(0) != edge_index.size(1) or delta_z.size(0) != edge_index.size(1):
        raise RuntimeError("delta_x or delta_z size does not match edge_index size")

    # Construct off-diagonal elements of sparse tensors D and Az
    values_D = 1 / delta_x
    values_Az = delta_z

    # Create the off-diagonal part of the sparse matrices D and Az
    D_off_diag = torch.sparse_coo_tensor(torch.stack([col, row]), values_D, (num_nodes, num_nodes), device=device)
    Az_off_diag = torch.sparse_coo_tensor(torch.stack([col, row]), values_Az, (num_nodes, num_nodes), device=device)

    # Calculate the diagonal elements of D and Az
    diag_values_D = torch_scatter.scatter_add(1 / delta_x, col, dim=0, dim_size=num_nodes)
    diag_values_Az = torch_scatter.scatter_add(delta_z, col, dim=0, dim_size=num_nodes)

    # Assign a default value of 1 to un-updated diagonal elements
    diag_values_D = diag_values_D + (diag_values_D == 0).float()  # If the diagonal element is 0, add 1
    diag_values_Az = diag_values_Az + (diag_values_Az == 0).float()  # If the diagonal element is 0, add 1

    # Create diagonal indices
    diag_indices = torch.arange(num_nodes, device=device)
    D_diag = torch.sparse_coo_tensor(torch.stack([diag_indices, diag_indices]), diag_values_D, (num_nodes, num_nodes), device=device)
    Az_diag = torch.sparse_coo_tensor(torch.stack([diag_indices, diag_indices]), diag_values_Az, (num_nodes, num_nodes), device=device)

    # Merge diagonal and off-diagonal parts
    D_sparse = D_diag + D_off_diag
    Az_sparse = Az_diag + Az_off_diag

    return D_sparse, Az_sparse


# Create a sparse tensor version of the second-order difference matrix
def create_2order_difference_matrix(num_nodes, edge_index, delta_x, delta_z):
    device = delta_x.device # type: ignore
    row = edge_index[0]
    col = edge_index[1]

    delta_x = delta_x.view(-1)
    delta_z = delta_z.view(-1)

    if delta_x.size(0) != edge_index.size(1) or delta_z.size(0) != edge_index.size(1):
        raise RuntimeError("delta_x or delta_z size does not match edge_index size")

    # Construct off-diagonal elements of D
    values_D = 1 / delta_x
    D_off_diag_temp1 = torch.sparse_coo_tensor(torch.stack([row, col]), values_D, (num_nodes, num_nodes), device=device)
    D_off_diag_temp2 = torch.sparse_coo_tensor(torch.stack([col, row]), values_D, (num_nodes, num_nodes), device=device)
    D_off_diag = D_off_diag_temp1 + D_off_diag_temp2


    # Calculate diagonal elements
    # For a 1D Laplacian (-1, 2, -1)/dx^2, the diagonal is -2/dx if dx is uniform.
    # Or sum of 1/dx for each neighbor.
    # The original code has diag_values = torch_scatter.scatter_add(2 / delta_x, row, ...)
    # This sums 2/delta_x for edges where 'row' is the source.
    # A more standard discrete Laplacian for node i is sum_j (u_j - u_i)/dx_ij^2 or (1/dx_ij)
    # Let's assume the formula intended: sum_j(1/dx_ij) for off-diagonal D[i,j] and -sum_j(1/dx_ij) for D[i,i]
    # The current code sums 2/delta_x for the diagonal based on 'row' as index.
    # If delta_x corresponds to the edge length, D[i,i] should be - sum_{j linked to i} (1/delta_x_ij)
    # And D[i,j] = 1/delta_x_ij
    # The code uses 2/delta_x for diagonal sum, which might be specific.
    diag_values = torch_scatter.scatter_add(1 / delta_x, row, dim=0, dim_size=num_nodes)
    diag_values += torch_scatter.scatter_add(1 / delta_x, col, dim=0, dim_size=num_nodes)


    diag_indices = torch.arange(num_nodes, device=device)
    D_diag = torch.sparse_coo_tensor(torch.stack([diag_indices, diag_indices]), -diag_values, (num_nodes, num_nodes), device=device)

    # Merge diagonal and off-diagonal parts
    D_sparse = D_diag + D_off_diag

    # Construct off-diagonal elements of Az matrix
    values_Az = delta_z
    Az_off_diag_temp1 = torch.sparse_coo_tensor(torch.stack([row, col]), values_Az, (num_nodes, num_nodes), device=device)
    Az_off_diag_temp2 = torch.sparse_coo_tensor(torch.stack([col, row]), values_Az, (num_nodes, num_nodes), device=device)
    Az_off_diag = Az_off_diag_temp1 + Az_off_diag_temp2


    # Az matrix typically doesn't have diagonal elements from this construction,
    # unless specified otherwise for source terms.
    Az_sparse = Az_off_diag

    return D_sparse, Az_sparse


# Testing code
if __name__ == "__main__":
    delta_t_val = 0.01
    num_nodes_val = 6
    input_dim_val = 3
    edge_attr_dim_val = 3
    num_layers_val = 3

    edge_orientation_val = "upstream" # "downstream", "bidirectional"

    edge_index_val = torch.tensor([
        [0, 1, 1, 2, 3, 4, 0],
        [1, 2, 3, 3, 4, 5, 3]
    ], dtype=torch.long)

    # Ensure edge attributes match edge_attr_dim_val
    edge_attr_val = torch.rand((edge_index_val.size(1), edge_attr_dim_val), dtype=torch.float)

    u_val = torch.rand((num_nodes_val, input_dim_val), dtype=torch.float)


    models_dict = {
        "MLP": MLP(in_channels=input_dim_val, hidden_channels=6, num_hidden=num_layers_val, param_sharing=False),
        "MultiLayerModel": MultiLayerModel(in_channels=input_dim_val, hidden_channels=6, num_hidden=num_layers_val,
                                           param_sharing=False, edge_orientation=edge_orientation_val)
    }

    for model_name_val, model_val in models_dict.items():
        print(f"Testing model: {model_name_val}")
        # For MLP, edge_attr might not be used by its apply_layer, but forward expects it
        if model_name_val == "MLP":
             # MLP base class forward expects edge_attr, even if apply_layer doesn't use it
            output_val = model_val(u_val, edge_index_val, edge_attr_val)
        else:
            output_val = model_val(u_val, edge_index_val, edge_attr_val)
        print(f"Model output:\n{output_val}\n")