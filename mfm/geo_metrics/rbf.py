# mfm/geo_metrics/rbf.py

import pytorch_lightning as pl
import torch
from sklearn.cluster import KMeans
import numpy as np


class RBFNetwork(pl.LightningModule):
    def __init__(
        self,
        current_timestep,
        next_timestep,
        n_centers: int = 100,
        kappa: float = 1.0,
        lr=1e-2,
        datamodule=None,
        image_data=False,
    ):
        super().__init__()
        self.K = n_centers
        self.current_timestep = current_timestep
        self.next_timestep = next_timestep
        self.clustering_model = KMeans(n_clusters=self.K)
        self.kappa = kappa
        self.last_val_loss = 1
        self.lr = lr
        self.W = torch.nn.Parameter(torch.rand(self.K, 1))
        self.datamodule = datamodule
        self.image_data = image_data

        # バッファ（学習後もデバイス追従）
        self.register_buffer("C", torch.empty(0))       # [K,D]
        self.register_buffer("lamda", torch.empty(0))   # [K,1]
        self.register_buffer("sigmas", torch.empty(0))  # [K,1]

    def on_before_zero_grad(self, *args, **kwargs):
        self.W.data = torch.clamp(self.W.data, min=0.0001)

    @torch.no_grad()
    def _compute_centroids_torch(self, all_data: torch.Tensor, labels: np.ndarray, K: int) -> torch.Tensor:
        """labels(np.ndarray) に基づいて all_data(torch [N,D]) のクラスタ中心を torch で計算"""
        N, D = all_data.shape
        C = torch.zeros(K, D, dtype=all_data.dtype, device=all_data.device)
        labels_t = torch.from_numpy(labels).to(all_data.device)
        for k in range(K):
            idx = (labels_t == k)
            if idx.any():
                pts = all_data[idx, :]
                C[k] = pts.mean(dim=0)
            else:
                C[k] = 0.0
        return C

    def _concat_flat(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        """2つのテンソルを結合してフラット化"""
        x = torch.cat([t0, t1])
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        return x

    def _dedup_with_indices(self, X_np: np.ndarray):
        """
        np.unique で完全一致行を一意化。保持する行の昇順インデックスを返す。
        
        注意: この処理は学習結果に影響を与えますが、既存のロジックをそのまま維持します。
        """
        _, keep_idx = np.unique(X_np, axis=0, return_index=True)
        keep_idx.sort()  # 元の順序を保つ
        return keep_idx

    def on_train_start(self):
        with torch.no_grad():
            # ===== データ準備（結合 → フラット化）=====
            if self.image_data:
                # all_data: 高解像ピクセル側（クラスタ中心/σ用）
                all_data = self._concat_flat(self.datamodule.train_x0, self.datamodule.train_x1)

                # data_to_fit: 低解像（ambient）側（KMeans用）
                ambient = self._concat_flat(self.datamodule.ambient_x0, self.datamodule.ambient_x1)
                data_to_fit = ambient.cpu().numpy().astype(np.float32, copy=False)
                
                print(f"[RBF-Image] Using ambient data for KMeans: {data_to_fit.shape}")
                print(f"[RBF-Image] Using full resolution for centroids: {all_data.shape}")
            else:
                batch = next(iter(self.trainer.datamodule.train_dataloader()))
                metric_samples_batch_filtered = [
                    x for i, x in enumerate(batch[0]["metric_samples"][0])
                    if i in [self.current_timestep, self.next_timestep]
                ]
                all_data = torch.cat(metric_samples_batch_filtered)
                if all_data.ndim > 2:
                    all_data = all_data.reshape(all_data.shape[0], -1)
                data_to_fit = all_data.cpu().numpy().astype(np.float32, copy=False)

            # ===== 重複除去（x0∪x1 の完全一致サンプルを全て削除）=====
            # 注意: この処理は既存のロジックをそのまま維持
            keep_idx = self._dedup_with_indices(data_to_fit)
            n_before = data_to_fit.shape[0]
            if len(keep_idx) < n_before:
                n_removed = n_before - len(keep_idx)
                removal_rate = n_removed / n_before * 100
                print(f"[RBF] Deduplication: removed {n_removed}/{n_before} samples ({removal_rate:.1f}%)")
                print(f"[RBF] Data size: {n_before} -> {len(keep_idx)}")
                
                # data_to_fit (numpy) に適用
                data_to_fit = data_to_fit[keep_idx]
                # all_data (torch) にも同じインデックスを適用（順序対応を仮定）
                idx_t = torch.from_numpy(keep_idx).to(all_data.device)
                all_data = all_data.index_select(0, idx_t)
            else:
                print(f"[RBF] No duplicates found: {n_before} unique samples")

            # ===== KMeans（K > N の場合は自動縮小）=====
            n_unique = data_to_fit.shape[0]
            if self.K > n_unique:
                oldK = self.K
                self.K = int(n_unique)
                print(f"[RBF] WARNING: Reducing K due to insufficient data: {oldK} -> {self.K}")
                print(f"[RBF] Consider using fewer centers (n_centers < {n_unique})")
            
            # 再構築（K が変わった可能性に対応）
            self.clustering_model = KMeans(n_clusters=self.K, random_state=42)
            print(f"[RBF] Fitting KMeans: N={len(data_to_fit)}, D={data_to_fit.shape[1]}, K={self.K}")
            self.clustering_model.fit(data_to_fit)

            labels = self.clustering_model.labels_
            counts = np.bincount(labels, minlength=self.K)
            empty_clusters = int((counts == 0).sum())
            singletons = int((counts == 1).sum())
            
            print(f"[RBF] KMeans result:")
            print(f"  - Total clusters: {self.K}")
            print(f"  - Cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
            print(f"  - Empty clusters: {empty_clusters}")
            print(f"  - Singleton clusters: {singletons}")

            # ===== クラスタ中心（pixel は torch で再計算）=====
            all_data = all_data.to(self.device, non_blocking=True)
            if self.image_data:
                C = self._compute_centroids_torch(all_data, labels, self.K)
                print(f"[RBF] Computed centroids from full resolution data: {C.shape}")
            else:
                C = torch.tensor(self.clustering_model.cluster_centers_, dtype=torch.float32, device=self.device)

            # ===== σ_k 計算（分散0検出）=====
            # 注意: 既存のロジックをそのまま維持（学習結果を変えないため）
            sigmas = torch.zeros(self.K, 1, dtype=torch.float32, device=self.device)
            labels_t = torch.from_numpy(labels).to(all_data.device)
            zero_var_count = 0
            zero_var_points_total = 0
            
            for k in range(self.K):
                idx = (labels_t == k)
                if not idx.any():
                    sigmas[k, 0] = 0.0
                    continue
                pts = all_data[idx, :]  # [n_k, D]
                var_dim = ((pts - C[k]) ** 2).mean(dim=0)
                sigma_k = torch.sqrt(var_dim.sum() if self.image_data else var_dim.mean())
                sigmas[k, 0] = sigma_k
                
                if sigma_k.item() == 0.0:
                    zero_var_count += 1
                    zero_var_points_total += int(counts[k])

            # ===== λ_k = 0.5 / (κ σ_k)^2 =====
            # 注意: 既存のロジックをそのまま維持
            lamda = torch.empty_like(sigmas)
            nonzero_mask = (sigmas.squeeze(1) > 0)
            lamda[nonzero_mask, 0] = 0.5 / (self.kappa * sigmas[nonzero_mask, 0]) ** 2
            lamda[~nonzero_mask, 0] = torch.inf  # 後で除去

            nonfinite_lambda = int(torch.isfinite(lamda).logical_not().sum().item())
            lamda_min = float(torch.where(torch.isfinite(lamda), lamda, torch.tensor(float('inf'), device=lamda.device)).min().item())
            lamda_max_finite = float(torch.where(torch.isfinite(lamda), lamda, torch.tensor(float('-inf'), device=lamda.device)).max().item())
            
            print(f"[RBF] Sigma statistics:")
            print(f"  - Range: [{float(sigmas.min().item()):.6f}, {float(sigmas.max().item()):.6f}]")
            print(f"  - Zero-variance clusters: {zero_var_count}/{self.K} (containing {zero_var_points_total} points)")
            print(f"[RBF] Lambda statistics:")
            print(f"  - Finite range: [{lamda_min:.6f}, {lamda_max_finite:.6f}]")
            print(f"  - Non-finite (will be removed): {nonfinite_lambda}")

            # 分散ゼロクラスタの詳細表示
            if zero_var_count > 0:
                zero_idx = (sigmas.squeeze(1) == 0.0).nonzero(as_tuple=False).flatten().cpu().numpy()
                sizes_str = ", ".join(str(int(s)) for s in counts[zero_idx][: min(10, len(zero_idx))])
                print(f"[RBF] Zero-variance cluster details:")
                print(f"  - Count: {len(zero_idx)}")
                print(f"  - Total points affected: {zero_var_points_total}")
                print(f"  - Sizes (first {min(10, len(zero_idx))}): [{sizes_str}]")

            # ===== 分散ゼロクラスタの完全除去 =====
            # 注意: 既存のロジックをそのまま維持
            keep_mask = (sigmas.squeeze(1) > 0) & torch.isfinite(lamda.squeeze(1))
            n_keep = int(keep_mask.sum().item())
            n_drop = self.K - n_keep
            
            if n_keep == 0:
                raise RuntimeError(
                    "[RBF] FATAL: All clusters have zero variance. Cannot proceed.\n"
                    "Possible causes:\n"
                    "  1. Duplicate data in x0 and x1\n"
                    "  2. Too many centers (n_centers) for the data\n"
                    "  3. Data dimensionality issue\n"
                    f"Current settings: n_centers={self.K}, data_shape={data_to_fit.shape}"
                )
            
            if n_drop > 0:
                print(f"[RBF] Removing {n_drop} zero-variance clusters: K {self.K} -> {n_keep}")
                C = C[keep_mask]
                lamda = lamda[keep_mask]
                sigmas = sigmas[keep_mask]
                self.W = torch.nn.Parameter(torch.rand(n_keep, 1, device=self.device))
                self.K = n_keep
                print(f"[RBF] Final configuration: K={self.K}, centers_shape={C.shape}")

            # ===== バッファに格納 =====
            self.C = C
            self.lamda = lamda
            self.sigmas = sigmas
            print(f"[RBF] Initialization complete: K={self.K}, device={self.device}")

    def forward(self, x):
        dev = self.C.device
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        x = x.to(dev, non_blocking=True)
        dist2 = torch.cdist(x, self.C) ** 2  # [B,K]
        phi_x = torch.exp(-0.5 * self.lamda[None, :, :] * dist2[:, :, None])  # [B,K,1]
        h_x = (self.W.to(dev) * phi_x).sum(dim=1)  # [B,1]
        return h_x

    def training_step(self, batch, batch_idx):
        if self.image_data:
            inputs = batch
        else:
            inputs = torch.cat(
                [
                    x
                    for i, x in enumerate(batch["train_samples"][0])
                    if i in [self.current_timestep, self.next_timestep]
                ]
            )
        loss = ((1 - self.forward(inputs)) ** 2).mean()
        self.log(
            "MetricModel/train_loss_learn_metric",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        metric_samples_batch_filtered = [
            x
            for i, x in enumerate(batch["val_samples"][0])
            if i in [self.current_timestep, self.next_timestep]
        ]
        h = self.forward(torch.cat(metric_samples_batch_filtered))
        loss = ((1 - h) ** 2).mean()
        self.log(
            "MetricModel/val_loss_learn_metric",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.last_val_loss = loss.detach()
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def compute_metric(self, x, alpha=1, epsilon=1e-2, image_hx=False):
        """
        メトリックテンソルを計算
        
        注意: 既存のロジックをそのまま維持（学習結果を変えないため）
        """
        h_x = self.forward(x)
        if image_hx:
            h_x = 1 - torch.abs(1 - h_x)
            M_x = 1 / (h_x**alpha + epsilon)
        else:
            M_x = 1 / (h_x + epsilon) ** alpha
        # デバイスを揃える（既存の修正を維持）
        return M_x.to(x.device, non_blocking=True)