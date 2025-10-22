import torch
import pytorch_lightning as pl

from mfm.flow_matchers.ema import EMA


class GeoPathNetTrain(pl.LightningModule):
    def __init__(
        self,
        flow_matcher,
        args,
        skipped_time_points: list = None,
        ot_sampler=None,
        data_manifold_metric=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.flow_matcher = flow_matcher
        self.geopath_net = flow_matcher.geopath_net
        self.ot_sampler = ot_sampler
        self.skipped_time_points = skipped_time_points if skipped_time_points else []
        self.optimizer_name = args.geopath_optimizer
        self.lr = args.geopath_lr
        self.weight_decay = args.geopath_weight_decay
        self.args = args
        self.data_manifold_metric = data_manifold_metric
        self.multiply_validation = 4

        self.first_loss = None
        self.timesteps = None
        self.computing_reference_loss = False

    def forward(self, x0, x1, t):
        return self.geopath_net(x0, x1, t)

    def on_train_start(self):
        self.first_loss = self.compute_initial_loss()

    def compute_initial_loss(self):
        self.geopath_net.train(mode=False)
        total_loss = 0
        total_count = 0
        with torch.enable_grad():
            self.t_val = []
            for i in range(
                self.trainer.datamodule.num_timesteps - len(self.skipped_time_points)
            ):
                self.t_val.append(
                    torch.rand(
                        self.trainer.datamodule.batch_size * self.multiply_validation,
                        requires_grad=True,
                    )
                )
        self.computing_reference_loss = True
        with torch.no_grad():
            old_alpha = self.flow_matcher.alpha
            self.flow_matcher.alpha = 0
            for batch in self.trainer.datamodule.train_dataloader():
                self.timesteps = torch.linspace(
                    0.0, 1.0, len(batch[0]["train_samples"][0])
                )
                loss = self._compute_loss(
                    batch[0]["train_samples"][0],
                    batch[0]["metric_samples"][0],
                )
                total_loss += float(loss)
                total_count += 1
            self.flow_matcher.alpha = old_alpha
        self.computing_reference_loss = False
        self.geopath_net.train(mode=True)
        return total_loss / total_count if total_count > 0 else 1.0

    def _compute_loss(self, main_batch, metric_samples_batch):
        main_batch_filtered = [
            x.to(self.device)
            for i, x in enumerate(main_batch)
            if i not in self.skipped_time_points
        ]
        metric_samples_batch_filtered = [
            x.to(self.device)
            for i, x in enumerate(metric_samples_batch)
            if i not in self.skipped_time_points
        ]

        x0s, x1s = main_batch_filtered[:-1], main_batch_filtered[1:]
        samples0, samples1 = (
            metric_samples_batch_filtered[:-1],
            metric_samples_batch_filtered[1:],
        )

        ts, xts, uts = self._process_flow(x0s, x1s)

        velocities = []
        for i in range(len(ts)):
            samples = torch.cat([samples0[i], samples1[i]], dim=0)
            vel = self.data_manifold_metric.calculate_velocity(
                xts[i], uts[i], samples, i
            )
            velocities.append(vel)

        if len(velocities) == 0:
            # すべて重複で全スキップになったケースでも安全に動作
            self.log(
                "GeoPathNet/all_pairs_skipped",
                1.0,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        loss = torch.mean(torch.cat(velocities) ** 2)
        self.log(
            "GeoPathNet/mean_velocity_geopath",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    @staticmethod
    def _equal_mask(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """サンプル次元以外が完全一致しているかのマスク"""
        # 形状: (B, ...)
        reduce_dims = tuple(range(1, x0.ndim))
        if x0.dtype.is_floating_point:
            # 完全一致のみを重複と見なす（数値誤差での誤検出を避ける）
            diff = (x0 - x1).abs()
            return (diff.amax(dim=reduce_dims) == 0)
        else:
            return (x0 == x1).all(dim=reduce_dims)

    def _process_flow(self, x0s, x1s):
        ts, xts, uts = [], [], []
        t_start = self.timesteps[0]

        for i, (x0, x1) in enumerate(zip(x0s, x1s)):
            x0, x1 = torch.squeeze(x0), torch.squeeze(x1)

            if self.trainer.validating or self.computing_reference_loss:
                repeat_tuple = (self.multiply_validation, 1) + (1,) * (
                    len(x0.shape) - 2
                )
                x0 = x0.repeat(repeat_tuple)
                x1 = x1.repeat(repeat_tuple)

            if self.ot_sampler is not None:
                x0, x1 = self.ot_sampler.sample_plan(x0, x1, replace=True)

            # --- 重複除外 & フォールバック再ペアリング ---
            same_mask = self._equal_mask(x0, x1)
            keep_mask = ~same_mask
            dup_ratio = same_mask.float().mean().item()
            self.log(
                "GeoPathNet/dup_ratio_in_pair",
                dup_ratio,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

            if keep_mask.sum().item() == 0:
                # すべて重複 → x1 をランダムにシャッフルして再ペア
                if x1.shape[0] > 1:
                    perm = torch.randperm(x1.shape[0], device=x1.device)
                    x1_shuf = x1[perm]
                    same_mask = self._equal_mask(x0, x1_shuf)
                    keep_mask = ~same_mask
                    if keep_mask.sum().item() > 0:
                        x1 = x1_shuf
                    else:
                        # それでも全重複ならこのタイムスライスはスキップ
                        if self.skipped_time_points and i + 1 >= self.skipped_time_points[0]:
                            t_start = self.timesteps[i + 2]
                        else:
                            t_start = self.timesteps[i + 1]
                        continue
                else:
                    # バッチサイズ1でどうにもならない
                    if self.skipped_time_points and i + 1 >= self.skipped_time_points[0]:
                        t_start = self.timesteps[i + 2]
                    else:
                        t_start = self.timesteps[i + 1]
                    continue

            x0 = x0[keep_mask]
            x1 = x1[keep_mask]

            if self.skipped_time_points and i + 1 >= self.skipped_time_points[0]:
                t_start_next = self.timesteps[i + 2]
            else:
                t_start_next = self.timesteps[i + 1]

            t = None
            if self.trainer.validating or self.computing_reference_loss:
                t = self.t_val[i]
                if t.shape[0] != x0.shape[0]:
                    t = t[: x0.shape[0]]

            t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(
                x0, x1, t_start, t_start_next, training_geopath_net=True, t=t
            )
            ts.append(t)
            xts.append(xt)
            uts.append(ut)
            t_start = t_start_next

        return ts, xts, uts

    def training_step(self, batch, batch_idx):
        main_batch = batch["train_samples"][0]
        metric_batch = batch["metric_samples"][0]
        tangential_velocity_loss = self._compute_loss(main_batch, metric_batch)
        if self.first_loss:
            tangential_velocity_loss = tangential_velocity_loss / self.first_loss
        self.log(
            "GeoPathNet/mean_geopath_geopath",
            (self.flow_matcher.geopath_net_output.abs().mean()),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "GeoPathNet/train_loss_geopath",
            tangential_velocity_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return tangential_velocity_loss

    def validation_step(self, batch, batch_idx):
        main_batch = batch["val_samples"][0]
        metric_batch = batch["metric_samples"][0]
        tangential_velocity_loss = self._compute_loss(main_batch, metric_batch)
        if self.first_loss:
            tangential_velocity_loss = tangential_velocity_loss / self.first_loss
        self.log(
            "GeoPathNet/val_loss_geopath",
            tangential_velocity_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return tangential_velocity_loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if isinstance(self.geopath_net, EMA):
            self.geopath_net.update_ema()

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.geopath_net.parameters(), lr=self.lr)
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.geopath_net.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        return optimizer