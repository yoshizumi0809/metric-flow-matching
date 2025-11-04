# mfm/flow_matchers/flow_net_train.py

import os
import torch
import wandb
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics.functional import mean_squared_error
from torchdyn.core import NeuralODE
from torchvision import transforms

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

        # ★ 追加: 実験名タグ作成などで使う
        self.args = args

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
        x0, cloud_points = batch
        node = NeuralODE(
            flow_model_torch_wrapper(self.flow_net),
            solver="euler",
            sensitivity="adjoint",
        )
        with torch.no_grad():
            traj = node.trajectory(
                x0,
                t_span=torch.linspace(0, 1, 101),
            ).cpu()

        if self.whiten:
            traj_shape = traj.shape
            traj = traj.reshape(-1, 3)
            traj = self.trainer.datamodule.scaler.inverse_transform(
                traj.detach().numpy()
            ).reshape(traj_shape)
            cloud_points = torch.tensor(
                self.trainer.datamodule.scaler.inverse_transform(
                    cloud_points.detach().numpy()
                )
            )
        traj = torch.transpose(torch.tensor(traj), 0, 1)

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
        ax.view_init(elev=30, azim=-115, roll=0)

        plot_lidar(ax, cloud_points, xs=traj)
        plt.savefig(os.path.join(os.getcwd(), f"lidar.png"), dpi=300)
        wandb.log({"lidar": wandb.Image(plt)})


class FlowNetTrainImage(FlowNetTrainBase):
    def on_test_start(self):
        # ODE 準備
        self.node = NeuralODE(
            flow_model_torch_wrapper(self.flow_net).to(self.device),
            solver="tsit5",
            sensitivity="adjoint",
            atol=1e-5,
            rtol=1e-5,
        )
        self.all_outputs = []

        # VAE / 後処理
        self.vae = self.trainer.datamodule.vae.to(self.device)
        self.postprocess = self.trainer.datamodule.process.postprocess
        image_size = self.trainer.datamodule.image_size
        self.ambient_x0 = transforms.Resize((image_size, image_size))(
            self.trainer.datamodule.ambient_x0
        )
        self.ambient_x1 = self.trainer.datamodule.ambient_x1

        # ===== 保存ディレクトリ: generated_samples/<experiment_name> に固定 =====
        exp_name = getattr(self.args, "experiment_name", None)
        if not exp_name or len(str(exp_name)) == 0:
            # experiment_name が無いときだけフォールバックで旧タグを使う
            seed = getattr(self.args, "seed_current", None)
            if seed is None and getattr(self.args, "seeds", None):
                seed = self.args.seeds[0]
            seed_str = f"seed{seed}" if seed is not None else "seedNA"
            data_type = getattr(self.trainer.datamodule, "data_type", "image")
            exp_name = f"{data_type}_{self.args.x0_label}_to_{self.args.x1_label}_{seed_str}"

        base_dir = os.path.join(self.working_dir, "generated_samples")
        self.sample_dir = os.path.join(base_dir, exp_name)
        if self.global_rank == 0:
            os.makedirs(self.sample_dir, exist_ok=True)
            self.print(f"[FlowNetTest] Saving samples to: {self.sample_dir}")

    def _to_01(self, x_minus1_to_1: torch.Tensor) -> torch.Tensor:
        """[-1,1] -> [0,1]"""
        return (x_minus1_to_1.clamp(-1, 1) + 1.0) * 0.5

    def test_step(self, batch, batch_idx):
        # ===== 生成 =====
        with torch.no_grad():
            # 高分解能で軌跡を取り、その中から等間隔サンプルを可視化
            t_dense = torch.linspace(0, 1, 100).to(self.device)
            traj = self.node.trajectory(batch.to(self.device), t_span=t_dense)
        # [T, B, C, H, W] -> [B, T, C, H, W]
        traj = traj.transpose(0, 1)

        # ===== 可視化設定 =====
        # 列: t = 0, 0.2, 0.4, 0.6, 0.8, 1.0
        t_vals = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], device=self.device)
        # 近いインデックスに丸め
        col_indices = torch.round(t_vals * (t_dense.numel() - 1)).long().tolist()
        n_cols = len(col_indices)

        # 行: 1行目= t 表示、2〜5行目= 4枚のサンプルの遷移
        n_rows_images = 4
        n_rows_total = 1 + n_rows_images
        n_show = min(n_rows_images, traj.size(0))  # バッチが小さいときも安全

        # 選択する latent をまとめてデコード（高速化）
        # 形: [n_show, n_cols, C, H, W] を一括 decode
        latents_sel = torch.stack(
            [traj[i, col_indices, ...] for i in range(n_show)], dim=0
        )  # [n_show, n_cols, C, H, W]
        flat_latents = latents_sel.reshape(n_show * n_cols, *latents_sel.shape[2:])
        decoded = self.vae.decode(flat_latents).sample.cpu().detach()  # [-1,1]
        imgs_01 = self._to_01(decoded)  # [0,1]
        # 形を [n_show, n_cols, 3, H, W] に戻す
        imgs_01 = imgs_01.reshape(n_show, n_cols, *imgs_01.shape[1:])

        # ===== パネル描画（1行目にt、2〜5行目に各サンプルの遷移）=====
        if self.global_rank == 0:
            fig_w = n_cols * 2.2
            fig_h = n_rows_total * 2.2
            fig, axes = plt.subplots(n_rows_total, n_cols, figsize=(fig_w, fig_h))

            # 1行目: t の値だけを中央に表示（白背景）
            for j in range(n_cols):
                ax = axes[0, j] if n_cols > 1 else axes[0]
                ax.axis("off")
                ax.set_facecolor("white")
                ax.text(
                    0.5,
                    0.5,
                    f"t={t_vals[j].item():.2f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )

            # 2〜5行目: 画像を配置
            for i in range(n_show):
                for j in range(n_cols):
                    ax = axes[1 + i, j] if n_cols > 1 else axes[1 + i]
                    img = imgs_01[i, j]
                    # [C,H,W] -> [H,W,C]
                    ax.imshow(img.permute(1, 2, 0).numpy())
                    ax.axis("off")

            plt.tight_layout()
            # 保存
            out_path = os.path.join(
                self.sample_dir, f"transition_panel_b{batch_idx:04d}.png"
            )
            plt.savefig(out_path, dpi=200)
            wandb.log({f"transition_panel/b{batch_idx:04d}": wandb.Image(plt)})
            plt.close(fig)

        # 生成物は評価用に保持（FID/LPIPS用）
        # ※ t=1 のみを all_outputs に追加（評価は最終出力で行う）
        final_latent = traj[:, col_indices[-1], ...]
        output = self.vae.decode(final_latent).sample.cpu().detach()
        self.all_outputs.append(output)

    def on_test_epoch_end(self):
        # ===== 評価 =====
        all_outputs = torch.cat(self.all_outputs, dim=0).cpu()
        x1_cpu = self.ambient_x1[: all_outputs.size(0)].cpu()
        x0_cpu = self.ambient_x0[: all_outputs.size(0)].cpu()

        fid = self.compute_fid(
            FIDImageDataset(x1_cpu, self.postprocess),
            FIDImageDataset(all_outputs, self.postprocess),
        )
        lpips_val = self.compute_lpips(
            self._prep_for_lpips(x0_cpu), self._prep_for_lpips(all_outputs)
        )

        self.log("FID", fid, on_step=False, on_epoch=True, prog_bar=True)
        self.log("LPIPS", lpips_val, on_step=False, on_epoch=True, prog_bar=True)

        # ★ 要望に合わせて、ここでは追加の画像保存（グリッド/個別）は行わない

    def _prep_for_lpips(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x_min, x_max = float(x.min()), float(x.max())
        if x_max > 1.0 or x_min < 0.0:
            x = x / 255.0
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