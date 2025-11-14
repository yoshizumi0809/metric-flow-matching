import os
import torch
import wandb
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics.functional import mean_squared_error
from torchdyn.core import NeuralODE

import lpips
from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from fld.metrics.FID import FID

from mfm.networks.utils import flow_model_torch_wrapper
from mfm.utils import wasserstein_distance, plot_arch, plot_lidar, plot_sphere
from mfm.flow_matchers.ema import EMA
from mfm.flow_matchers.eval_utils import FIDImageDataset


class FlowNetTrainBase(pl.LightningModule):
    def __init__(
        self,
        flow_matcher,
        flow_net,
        skipped_time_points=None,
        ot_sampler=None,
        args=None,
    ):
        super().__init__()
        self.flow_matcher = flow_matcher
        self.flow_net = flow_net
        self.ot_sampler = ot_sampler
        self.skipped_time_points = skipped_time_points

        self.optimizer_name = args.flow_optimizer
        self.lr = args.flow_lr
        self.weight_decay = args.flow_weight_decay
        self.whiten = args.whiten
        self.working_dir = args.working_dir
        self.args = args  # 実験名など

    def forward(self, t, xt):
        return self.flow_net(t, xt)

    def _compute_loss(self, main_batch):
        main_batch_filtered = [
            x for i, x in enumerate(main_batch) if i not in self.skipped_time_points
        ]
        x0s, x1s = main_batch_filtered[:-1], main_batch_filtered[1:]
        ts, xts, uts = self._process_flow(x0s, x1s)

        t = torch.cat(ts)
        xt = torch.cat(xts)
        ut = torch.cat(uts)
        vt = self(t[:, None], xt)

        loss = mean_squared_error(vt, ut)
        return loss

    def _process_flow(self, x0s, x1s):
        ts, xts, uts = [], [], []
        t_start = self.timesteps[0]

        for i, (x0, x1) in enumerate(zip(x0s, x1s)):
            x0, x1 = torch.squeeze(x0), torch.squeeze(x1)

            if self.ot_sampler is not None:
                x0, x1 = self.ot_sampler.sample_plan(
                    x0,
                    x1,
                    replace=True,
                )
            if self.skipped_time_points and i + 1 >= self.skipped_time_points[0]:
                t_start_next = self.timesteps[i + 2]
            else:
                t_start_next = self.timesteps[i + 1]

            t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(
                x0, x1, t_start, t_start_next
            )

            ts.append(t)
            xts.append(xt)
            uts.append(ut)
            t_start = t_start_next
        return ts, xts, uts

    def training_step(self, batch, batch_idx):
        main_batch = batch["train_samples"][0]
        self.timesteps = torch.linspace(0.0, 1.0, len(main_batch)).tolist()
        loss = self._compute_loss(main_batch)
        if self.flow_matcher.alpha != 0:
            self.log(
                "FlowNet/mean_geopath_cfm",
                (self.flow_matcher.geopath_net_output.abs().mean()),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        self.log(
            "FlowNet/train_loss_cfm",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        main_batch = batch["val_samples"][0]
        self.timesteps = torch.linspace(0.0, 1.0, len(main_batch)).tolist()
        val_loss = self._compute_loss(main_batch)
        self.log(
            "FlowNet/val_loss_cfm",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return val_loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if isinstance(self.flow_net, EMA):
            self.flow_net.update_ema()

    def configure_optimizers(self):
        if self.optimizer_name == "adamw":
            optimizer = AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
            )
        return optimizer


class FlowNetTrainTrajectory(FlowNetTrainBase):
    def test_step(self, batch, batch_idx):
        data_type = self.trainer.datamodule.data_type
        node = NeuralODE(
            flow_model_torch_wrapper(self.flow_net),
            solver="euler",
            sensitivity="adjoint",
            atol=1e-5,
            rtol=1e-5,
        )

        t_exclude = self.skipped_time_points[0] if self.skipped_time_points else None
        if t_exclude is not None:
            traj = node.trajectory(
                batch[t_exclude - 1],
                t_span=torch.linspace(
                    self.timesteps[t_exclude - 1], self.timesteps[t_exclude], 101
                ),
            )
            X_mid_pred = traj[-1]
            traj = node.trajectory(
                batch[t_exclude - 1],
                t_span=torch.linspace(
                    self.timesteps[t_exclude - 1],
                    self.timesteps[t_exclude + 1],
                    101,
                ),
            )
            if data_type == "arch":
                plot_arch(
                    batch,
                    traj,
                    time_steps=[t_exclude - 1, t_exclude + 1],
                    n_samples=400,
                    fname=os.path.join(os.getcwd(), f"arch_trajs.png"),
                )
            elif data_type == "sphere":
                mean_distance_from_sphere = torch.abs(
                    torch.sqrt((X_mid_pred**2).sum(dim=1)) - 1
                ).mean()
                self.log(
                    "mean_distance_from_sphere",
                    mean_distance_from_sphere,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )
                plot_sphere(
                    batch,
                    traj,
                    time_steps=[t_exclude - 1, t_exclude + 1],
                    n_samples=100,
                    fname=os.path.join(os.getcwd(), f"sphere_trajs.png"),
                )

            EMD = wasserstein_distance(X_mid_pred, batch[t_exclude], p=1)
            self.final_EMD = EMD
            self.log("test_EMD", EMD, on_step=False, on_epoch=True, prog_bar=True)


class FlowNetTrainLidar(FlowNetTrainBase):
    def test_step(self, batch, batch_idx):
        # 変更なし
        pass


class FlowNetTrainImage(FlowNetTrainBase):
    def on_test_start(self):
        # ODE 準備（ピクセル空間）
        self.node = NeuralODE(
            flow_model_torch_wrapper(self.flow_net).to(self.device),
            solver="tsit5",
            sensitivity="adjoint",
            atol=1e-5,
            rtol=1e-5,
        )
        self.all_outputs = []

        # 可視化/評価用（ピクセルは [0,1] 想定）
        self.image_size = self.trainer.datamodule.image_size
        self.ambient_x0 = self.trainer.datamodule.ambient_x0  # 64x64（ある場合）
        self.ambient_x1 = self.trainer.datamodule.ambient_x1

        # FID 用の postprocess（ピクセル版）
        def pixel_postprocess(batch: torch.Tensor):
            # 入力: [B,3,H,W] in [0,1]
            # 出力: List[np.uint8 HWC]
            imgs = []
            for i in range(batch.size(0)):
                arr = (batch[i].permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255.0).astype("uint8")
                imgs.append(arr)
            return imgs

        self.postprocess = pixel_postprocess

        # 保存ディレクトリ
        exp_name = getattr(self.args, "experiment_name", None) or "pixel_flow"
        base_dir = os.path.join(self.working_dir, "generated_samples")
        self.sample_dir = os.path.join(base_dir, exp_name)
        if self.global_rank == 0:
            os.makedirs(self.sample_dir, exist_ok=True)
            self.print(f"[FlowNetTest-Pixel] Saving samples to: {self.sample_dir}")

    def _to_01(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(0.0, 1.0)

    def test_step(self, batch, batch_idx):
        # ===== 生成（ピクセル空間）=====
        with torch.no_grad():
            t_dense = torch.linspace(0, 1, 100).to(self.device)
            traj = self.node.trajectory(batch.to(self.device), t_span=t_dense)
        traj = traj.transpose(0, 1)  # [B,T,C,H,W]

        # ===== 可視化設定 =====
        t_vals = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], device=self.device)
        col_indices = torch.round(t_vals * (t_dense.numel() - 1)).long().tolist()
        n_cols = len(col_indices)

        n_rows_images = 4  # 2〜5行目に4枚
        n_rows_total = 1 + n_rows_images
        n_show = min(n_rows_images, traj.size(0))

        # 可視化用ピクセル
        imgs_sel = torch.stack([traj[i, col_indices, ...] for i in range(n_show)], dim=0)
        imgs_01 = self._to_01(imgs_sel.detach().cpu())  # [n_show, n_cols, 3, H, W]

        # ===== パネル描画 =====
        if self.global_rank == 0:
            fig_w = n_cols * 2.2
            fig_h = n_rows_total * 2.2
            fig, axes = plt.subplots(n_rows_total, n_cols, figsize=(fig_w, fig_h))

            # 1行目: t の値
            for j in range(n_cols):
                ax = axes[0, j] if n_cols > 1 else axes[0]
                ax.axis("off")
                ax.set_facecolor("white")
                ax.text(
                    0.5, 0.5, f"t={t_vals[j].item():.2f}",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold",
                )

            # 2〜5行目: 遷移画像
            for i in range(n_show):
                for j in range(n_cols):
                    ax = axes[1 + i, j] if n_cols > 1 else axes[1 + i]
                    img = imgs_01[i, j]
                    ax.imshow(img.permute(1, 2, 0).numpy())
                    ax.axis("off")

            plt.tight_layout()
            out_path = os.path.join(self.sample_dir, f"transition_panel_b{batch_idx:04d}.png")
            plt.savefig(out_path, dpi=200)
            wandb.log({f"transition_panel/b{batch_idx:04d}": wandb.Image(plt)})
            plt.close(fig)

        # ===== 評価用（最終時刻のみ）=====
        final_imgs = self._to_01(traj[:, col_indices[-1], ...].detach().cpu())
        self.all_outputs.append(final_imgs)

    def on_test_epoch_end(self):
        # ===== 評価 =====
        all_outputs = torch.cat(self.all_outputs, dim=0).cpu()  # [N,3,H,W]
        # 目標側（x1）もピクセルテンソル
        x1_cpu = self.trainer.datamodule.val_x1[: all_outputs.size(0)].cpu()
        x0_cpu = self.trainer.datamodule.val_x0[: all_outputs.size(0)].cpu()

        # FID / LPIPS
        fid = self.compute_fid(
            FIDImageDataset(x1_cpu, self.postprocess),
            FIDImageDataset(all_outputs, self.postprocess),
        )
        lpips_val = self.compute_lpips(
            self._prep_for_lpips(x0_cpu), self._prep_for_lpips(all_outputs)
        )

        self.log("FID", fid, on_step=False, on_epoch=True, prog_bar=True)
        self.log("LPIPS", lpips_val, on_step=False, on_epoch=True, prog_bar=True)

    def _prep_for_lpips(self, x: torch.Tensor) -> torch.Tensor:
        # LPIPS は [-1,1] / 3ch を期待。今は [0,1] なので変換
        x = x.float().clamp(0, 1)
        x = x * 2.0 - 1.0
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            x = x[:, :3, ...]
        return x

    def compute_lpips(self, x0_data, gen_data):
        loss_fn = lpips.LPIPS(net="vgg").to(x0_data.device)
        return loss_fn(x0_data, gen_data).mean().item()

    def compute_fid(self, val_data, gen_data):
        feature_extractor = InceptionFeatureExtractor()
        val_feat = feature_extractor.get_features(val_data)
        gen_feat = feature_extractor.get_features(gen_data)
        return FID().compute_metric(val_feat, None, gen_feat)