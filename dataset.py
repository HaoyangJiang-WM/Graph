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

    def get(self, idx):
        year_tensor, offset = self._decode_index(idx)
        x = year_tensor[:, offset:(offset + self.window_size)]
        y = year_tensor[:, offset + self.window_size + (self.lead_time - 1), 0]
        return Data(x=x, y=y.unsqueeze(-1), edge_index=self.edge_index, edge_attr=self.edge_attr)

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
