import os
import pandas as pd
import tarfile
import torch
import urllib.request

from torch_geometric.data import Data, Dataset
from tqdm import tqdm


class LamaHDataset(Dataset):
    DATA_URL = "https://zenodo.org/record/5153305/files/1_LamaH-CE_daily_hourly.tar.gz"
    Q_COL = "qobs"
    MET_COLS = [
        "prec",  # precipitation
        "volsw_123",  # topsoil moisture
        "2m_temp",  # air temperature
        "surf_press",  # surface pressure
    ]

    def __init__(self, root_dir, years=range(2000, 2018), root_gauge_id=399, rewire_graph=True,
                 window_size=24, stride_length=1, lead_time=6, normalized=False):
        if not set(years).issubset(range(2000, 2018)):
            raise ValueError("Only years between 2000 and 2017 are supported")

        self.years = years
        self.root_gauge_id = root_gauge_id
        self.rewire_graph = rewire_graph
        self.window_size = window_size
        self.stride_length = stride_length
        self.lead_time = lead_time
        self.normalized = normalized

        super().__init__(root_dir)  # calls download() and process() if necessary

        adj_df = pd.read_csv(self.processed_paths[0])
        self.gauges = list(sorted(set(adj_df["ID"]).union(adj_df["NEXTDOWNID"])))
        rev_index = {gauge_id: i for i, gauge_id in enumerate(self.gauges)}
        edge_cols = adj_df[["ID", "NEXTDOWNID"]].applymap(lambda x: rev_index[x])
        self.edge_index = torch.tensor(edge_cols.values.transpose(), dtype=torch.long)
        weight_cols = adj_df[["dist_hdn", "elev_diff", "strm_slope"]]
        self.edge_attr = torch.tensor(weight_cols.values, dtype=torch.float)

        stats_df = pd.read_csv(self.processed_paths[1], index_col="ID")
        self.mean = torch.tensor(stats_df[[f"{col}_mean" for col in [self.Q_COL] + self.MET_COLS]].values,
                                 dtype=torch.float)
        self.std = torch.tensor(stats_df[[f"{col}_std" for col in [self.Q_COL] + self.MET_COLS]].values,
                                dtype=torch.float)

        self.year_sizes = [(24 * (365 + int(year % 4 == 0)) - (window_size + lead_time)) // stride_length + 1
                           for year in years]
        self.year_tensors = [[] for _ in years]
        print("Loading dataset into memory...")
        for gauge_id in tqdm(self.gauges):
            q_df = pd.read_csv(f"{self.raw_dir}/{self.raw_file_names[2]}/hourly/ID_{gauge_id}.csv",
                               sep=";", usecols=["YYYY"] + [self.Q_COL])
            met_df = pd.read_csv(f"{self.raw_dir}/{self.raw_file_names[1]}/hourly/ID_{gauge_id}.csv",
                                 sep=";", usecols=["YYYY"] + self.MET_COLS)
            if normalized:
                q_df[self.Q_COL] = (q_df[self.Q_COL] - stats_df.loc[gauge_id, f"{self.Q_COL}_mean"]) / stats_df.loc[
                    gauge_id, f"{self.Q_COL}_std"]
                for col in self.MET_COLS:
                    met_df[col] = (met_df[col] - stats_df.loc[gauge_id, f"{col}_mean"]) / stats_df.loc[
                        gauge_id, f"{col}_std"]
            for i, year in enumerate(years):
                q_tensor = torch.tensor(q_df[q_df["YYYY"] == year][self.Q_COL].values, dtype=torch.float).unsqueeze(-1)
                met_tensor = torch.tensor(met_df[met_df["YYYY"] == year][self.MET_COLS].values, dtype=torch.float)
                self.year_tensors[i].append(torch.cat([q_tensor, met_tensor], dim=1))
        self.year_tensors[:] = map(torch.stack, self.year_tensors)

    @property
    def raw_file_names(self):
        return ["B_basins_intermediate_all/1_attributes",
                "B_basins_intermediate_all/2_timeseries",
                "D_gauges/2_timeseries"]

    @property
    def processed_file_names(self):
        return [f"adjacency_{self.root_gauge_id}_{self.rewire_graph}.csv",
                f"statistics_{self.root_gauge_id}_{self.rewire_graph}.csv"]

    def download(self):
        print("Downloading LamaH-CE from Zenodo to", self.raw_dir)
        total_size = int(urllib.request.urlopen(self.DATA_URL).info().get("Content-Length"))
        with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc="Downloading") as pbar:
            filename, _ = urllib.request.urlretrieve(self.DATA_URL,
                                                     filename="./archive.tar",
                                                     reporthook=lambda _, n, __: pbar.update(n))
        archive = tarfile.open(filename)
        for member in tqdm(archive.getmembers(), desc="Extracting"):
            if member.name.startswith(tuple(self.raw_file_names)):
                archive.extract(member, self.raw_dir)
        os.remove(filename)

    def process(self):
        adj_df = pd.read_csv(f"{self.raw_dir}/{self.raw_file_names[0]}/Stream_dist.csv", sep=";")
        adj_df.drop(columns="strm_slope", inplace=True)  # will re-calculate from dist_hdn and elev_diff

        stats_df = pd.DataFrame(
            columns=sum([[f"{col}_mean", f"{col}_std"] for col in [self.Q_COL] + self.MET_COLS], []),
            index=pd.Index([], name="ID")
        )

        connected_gauges = set(adj_df["ID"]).union(adj_df["NEXTDOWNID"])
        print(f"Discovering feasible gauges...")
        feasible_gauges = set(self._collect_upstream(self.root_gauge_id, adj_df, stats_df))
        print()
        assert feasible_gauges.issubset(connected_gauges)
        print(f"Discovered {len(feasible_gauges)} feasible gauges starting at ID {self.root_gauge_id} "
              + ("with graph rewiring" if self.rewire_graph else "without graph rewiring"))

        for gauge_id in tqdm(connected_gauges - feasible_gauges, desc="Bad gauge removal"):
            adj_df = self._remove_gauge_edges(gauge_id, adj_df)

        print("Saving final adjacency list to", self.processed_paths[0])
        adj_df["strm_slope"] = adj_df["elev_diff"] / adj_df["dist_hdn"]  # re-calculate
        adj_df.sort_values(by="ID", inplace=True)
        adj_df.to_csv(self.processed_paths[0], index=False)

        print("Saving feature summary statistics to", self.processed_paths[1], end="\n\n")
        stats_df.sort_values(by="ID", inplace=True)
        stats_df.to_csv(self.processed_paths[1], index=True)

    def _collect_upstream(self, gauge_id, adj_df, stats_df):
        print(f"Processing gauge #{gauge_id}", end="\r", flush=True)
        collected_ids = set()
        is_valid, gauge_stats = self._has_valid_data(gauge_id)
        if is_valid:
            collected_ids.add(gauge_id)
            stats_df.loc[gauge_id] = gauge_stats
        if is_valid or self.rewire_graph:
            predecessor_ids = set(adj_df[adj_df["NEXTDOWNID"] == gauge_id]["ID"])
            collected_ids.update(*[self._collect_upstream(pred_id, adj_df, stats_df) for pred_id in predecessor_ids])
        return collected_ids

    def _has_valid_data(self, gauge_id):
        q_df = pd.read_csv(f"{self.raw_dir}/{self.raw_file_names[2]}/hourly/ID_{gauge_id}.csv",
                           sep=";", usecols=["YYYY", self.Q_COL])
        met_df = pd.read_csv(f"{self.raw_dir}/{self.raw_file_names[1]}/hourly/ID_{gauge_id}.csv",
                             sep=";", usecols=["YYYY"] + self.MET_COLS)
        if (q_df[self.Q_COL] > 0).all() and (q_df[self.Q_COL] <= 1e30).all():
            q_df = q_df[(q_df["YYYY"] >= 2000) & (q_df["YYYY"] <= 2017)]
            met_df = met_df[(met_df["YYYY"] >= 2000) & (met_df["YYYY"] <= 2017)]
            if len(q_df) == (18 * 365 + 5) * 24 and len(met_df) == (18 * 365 + 5) * 24:  # number of hours in 2000-2017
                q_df_train = q_df[q_df["YYYY"] <= 2015]
                met_df_train = met_df[met_df["YYYY"] <= 2015]
                return True, [q_df_train[self.Q_COL].mean(), q_df_train[self.Q_COL].std()] \
                             + sum([[met_df_train[col].mean(), met_df_train[col].std()] for col in self.MET_COLS], [])
        return False, None

    def _remove_gauge_edges(self, gauge_id, adj_df):
        incoming_edges = adj_df.loc[adj_df["NEXTDOWNID"] == gauge_id]
        outgoing_edges = adj_df.loc[adj_df["ID"] == gauge_id]

        adj_df.drop(labels=incoming_edges.index, inplace=True)
        adj_df.drop(labels=outgoing_edges.index, inplace=True)

        if self.rewire_graph:  # need to rewire nodes that are adjacent to a deleted node
            bypass = incoming_edges.merge(outgoing_edges, how="cross", suffixes=["", "_"])
            bypass["NEXTDOWNID"] = bypass["NEXTDOWNID_"]
            bypass["dist_hdn"] += bypass["dist_hdn_"]
            bypass["elev_diff"] += bypass["elev_diff_"]
            adj_df = pd.concat([adj_df, bypass[["ID", "NEXTDOWNID", "dist_hdn", "elev_diff"]]],
                               ignore_index=True, copy=False)

        return adj_df.reset_index(drop=True)

    def len(self):
        return sum(self.year_sizes)

#     def get(self, idx):
#         year_tensor, offset = self._decode_index(idx)
#         x = year_tensor[:, offset:(offset + self.window_size)]
#         y = year_tensor[:, offset + self.window_size + (self.lead_time - 1), 0]
#         return Data(x=x, y=y.unsqueeze(-1), edge_index=self.edge_index, edge_attr=self.edge_attr)

    # def get(self, idx):
    #     year_tensor, offset = self._decode_index(idx)
    #
    #     # 获取原始数据（未扰动）
    #     x_original = year_tensor[:, offset:(offset + self.window_size)].clone()  # 复制，防止原始数据被修改
    #     y = year_tensor[:, offset + self.window_size + (self.lead_time - 1), 0]
    #
    #     x_perturbed = x_original.clone()  # 克隆原始数据
    #
    #     # 定义需要加扰动的 gauge 及对应扰动值 (gauge_id: perturbation)
    #     gauges_to_perturb = {
    #         225: 0.1,
    #         227: -0.2,
    #         228: 0.1
    #     }
    #
    #     for g_id, perturbation in gauges_to_perturb.items():
    #         gauge_idx = self.gauges.index(g_id)
    #         # 只在最后一个时间点添加扰动 (对于特征维度0)
    #         x_perturbed[gauge_idx, :, 0] += perturbation
    #
    #     # 构造原始数据和扰动数据的 Data 对象
    #     original_data = Data(
    #         x=x_original,
    #         y=y.unsqueeze(-1),
    #         edge_index=self.edge_index,
    #         edge_attr=self.edge_attr,
    #     )
    #     perturbed_data = Data(
    #         x=x_perturbed,
    #         y=y.unsqueeze(-1),
    #         edge_index=self.edge_index,
    #         edge_attr=self.edge_attr,
    #     )
    #
    #     return original_data, perturbed_data


    def get(self, idx):
        year_tensor, offset = self._decode_index(idx)

        # 获取原始数据（未扰动）
        x_original = year_tensor[:, offset:(offset + self.window_size)].clone()  # 复制，防止原始数据被修改
        y = year_tensor[:, offset + self.window_size + (self.lead_time - 1), 0]

        # 添加扰动到指定 gauge（gauge 214）
        gauge_214_idx = self.gauges.index(227)  # 获取 gauge 214 的索引
        x_perturbed = x_original.clone()  # 克隆原始数据，防止直接修改
        perturbation = 0.5  # 定义扰动大小
        # x_perturbed[gauge_214_idx, :, 0] += perturbation
        x_perturbed[gauge_214_idx, -3, 0] += 0.2 * perturbation
        x_perturbed[gauge_214_idx, -2, 0] += 0.5*perturbation
        x_perturbed[gauge_214_idx, -1, 0] += perturbation
#         x_perturbed[gauge_214_idx] += perturbation  # 在指定节点添加扰动

        # 构造原始数据和扰动数据的 Data 对象
        original_data = Data(
            x=x_original,
            y=y.unsqueeze(-1),
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
        )
        perturbed_data = Data(
            x=x_perturbed,
            y=y.unsqueeze(-1),
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
        )

        return original_data, perturbed_data

    def _decode_index(self, idx):
        for i, size in enumerate(self.year_sizes):
            idx -= size
            if idx < 0:
                return self.year_tensors[i], self.stride_length * (idx + size)
        raise AssertionError("Corrupt internal state. This should never happen!")

    def normalize(self, x):
        return (x - self.mean[:, None, :]) / self.std[:, None, :]

    def denormalize(self, x):
        return self.std[:, None, :] * x + self.mean[:, None, :]

    def longest_path(self):
        def longest_upstream_path(gauge_idx):
            predecessor_ids = self.edge_index[0, self.edge_index[1] == gauge_idx].tolist()
            if not predecessor_ids:
                return 0
            else:
                return 1 + max(longest_upstream_path(pred_id) for pred_id in predecessor_ids)

        return max(longest_upstream_path(start_idx) for start_idx in self.edge_index[1].unique())

    def get_gauge_mapping(self):
        """
        返回一个字典，映射每个节点的索引到对应的 gauge_id。
        """
        return {i: gauge_id for i, gauge_id in enumerate(self.gauges)}
    
    def extract_path_from_data(self, data, start_gauge_id, end_gauge_id):
        """
        从完整图中提取路径子图，包含节点特征、边信息和目标值。
        同时返回节点对应关系（Gauge ID -> Index）。
        """
        try:
            start_idx = self.gauges.index(start_gauge_id)  # 获取 start_gauge_id 的索引
            end_idx = self.gauges.index(end_gauge_id)  # 获取 end_gauge_id 的索引
        except ValueError as e:
            raise ValueError(f"Gauge ID not found in gauges: {e}")

        # 路径提取逻辑
        branch_nodes = [start_idx]  # 路径上的节点索引
        branch_edges = []  # 路径上的边
        current_idx = start_idx  # 当前节点索引

        def get_successor(node_idx):
            """
            获取当前节点的所有出边
            """
            successors = data.edge_index[1, data.edge_index[0] == node_idx].tolist()
            return successors

        # 找到 start 到 end 的路径
        while current_idx != end_idx:
            successors = get_successor(current_idx)
            if not successors:
                raise ValueError(f"No path found from gauge ID {start_gauge_id} to {end_gauge_id}")
            current_idx = successors[0]  # 假设路径是唯一的，选第一个 successor
            branch_nodes.append(current_idx)
            branch_edges.append((branch_nodes[-2], branch_nodes[-1]))

        # 构建节点索引映射
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(branch_nodes)}  # 构造新子图的映射
        new_edge_index = torch.tensor(
            [[node_map[edge[0]], node_map[edge[1]]] for edge in branch_edges],
            dtype=torch.long
        ).t()

        # 提取节点特征
        node_features = data.x[branch_nodes]

        # 提取边属性
        edge_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
        for edge in branch_edges:
            edge_mask |= (data.edge_index[0] == edge[0]) & (data.edge_index[1] == edge[1])
        edge_attributes = data.edge_attr[edge_mask]

        # 提取目标值（y）
        if hasattr(data, 'y') and data.y is not None:
            subgraph_y = data.y[branch_nodes]  # 根据子图节点索引提取 y
        else:
            subgraph_y = torch.empty(0)  # 如果 y 不存在则返回空张量

        # 构造 Gauge ID -> Index 映射
        gauge_index_mapping = {self.gauges[idx]: idx for idx in branch_nodes}

        # 返回子图和映射关系
        return Data(
            x=node_features,
            edge_index=new_edge_index,
            edge_attr=edge_attributes,
            y=subgraph_y  # 将目标值添加到子图
        ), gauge_index_mapping


DATASET_PATH = "./LamaH-CE"
dataset = LamaHDataset(DATASET_PATH, years=[2000])
print(dataset[0])
print(dataset.edge_attr.shape)

