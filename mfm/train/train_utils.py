import yaml
import string
import secrets
import os
from datetime import datetime

import torch
import wandb
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from torchdyn.core import NeuralODE

from mfm.utils import plot_images_trajectory
from mfm.networks.utils import flow_model_torch_wrapper


def load_config(path):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def merge_config(args, config_updates):
    for key, value in config_updates.items():
        if not hasattr(args, key):
            raise ValueError(
                f"Unknown configuration parameter '{key}' found in the config file."
            )
        setattr(args, key, value)
    return args


def generate_group_string(length=16):
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def dataset_name2datapath(dataset_name, working_dir):
    if dataset_name == "eb":
        return os.path.join(working_dir, "data", "eb_velocity_v5.npz")
    elif dataset_name == "cite":
        return os.path.join(working_dir, "data", "op_cite_inputs_0.h5ad")
    elif dataset_name == "multi":
        return os.path.join(working_dir, "data", "op_train_multi_targets_0.h5ad")
    elif dataset_name == "lidar":
        return os.path.join(working_dir, "data", "rainier2-thin.las")
    elif dataset_name == "afhq":
        return os.path.join(working_dir, "data", "afhq")
    elif dataset_name == "celeba":
        return os.path.join(working_dir, "data", "celeba")
    else:
        raise ValueError("Dataset not recognized")


# ============== ここから新規: 保存名をわかりやすく揃えるユーティリティ ==============
def _exp_tag(args, phase, data_type):
    """ログ／保存名に使う実験タグ（x0->x1, seed含む）"""
    x0 = getattr(args, "x0_label", "x0")
    x1 = getattr(args, "x1_label", "x1")
    seed = getattr(args, "seed_current", getattr(args, "seed", None))
    seed_str = f"seed{seed}" if seed is not None else "seedNA"
    return f"{data_type}_{phase}_{x0}_to_{x1}_{seed_str}"


def _ckpt_dir(args, phase, data_type, run_id=None):
    """チェックポイントの保存ディレクトリを一意で分かりやすく"""
    tag = _exp_tag(args, phase, data_type)
    # run_id を付けたいときは付ける（W&Bとの対応に便利）
    rid = f"run-{run_id}" if run_id else f"ts-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    path = os.path.join(args.working_dir, "checkpoints", data_type, tag, rid)
    os.makedirs(path, exist_ok=True)
    return path


def _filename_pattern(monitor_name: str | None, phase: str, args) -> str:
    """保存ファイル名のパターン（monitor があれば値も付ける）"""
    tag = _exp_tag(args, phase, args.data_type)
    if monitor_name:
        # '/' はファイル名に使えないので '_' に
        mon = monitor_name.replace("/", "_")
        return f"{tag}-{{epoch:03d}}-{{{mon}:.5f}}"
    else:
        return f"{tag}-{{epoch:03d}}"
# ============================================================================


def create_callbacks(args, phase, data_type, run_id, datamodule=None):
    """
    ▷ 変更点
      - 保存先: <working_dir>/checkpoints/<data_type>/<data_type>_<phase>_<x0>_to_<x1>_seed*/run-<wandb_id>/
      - すべて save_last=True を有効化（途中中断しても last.ckpt が残る）
      - 監視指標をフェーズごとに明示（Flow: FlowNet/val_loss_cfm, GeoPath: GeoPathNet/val_loss_geopath）
      - ファイル名に metric 値 or epoch を含める
    """
    # 監視するメトリクス名をフェーズごとに定義
    if phase == "geopath":
        monitor = "GeoPathNet/val_loss_geopath"
        mode = "min"
    elif phase == "flow":
        monitor = "FlowNet/val_loss_cfm"
        mode = "min"
    else:
        raise ValueError("Unknown phase")

    dirpath = _ckpt_dir(args, phase, data_type, run_id=run_id)
    filename = _filename_pattern(monitor, phase, args)

    callbacks = []

    if phase == "geopath":
        early_stop_callback = EarlyStopping(
            monitor=monitor,
            patience=args.patience_geopath,
            mode=mode,
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            mode=mode,
            save_top_k=3,      # 上位3つ
            save_last=True,    # ★重要
            every_n_epochs=1,
            auto_insert_metric_name=False,
        )
        callbacks = [checkpoint_callback, early_stop_callback]

    elif phase == "flow":
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            mode=mode,
            save_top_k=3,
            save_last=True,      # ★重要
            every_n_epochs=max(1, int(getattr(args, "check_val_every_n_epoch", 1))),
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpoint_callback)

        # 可視化コールバック（画像タスクのみ）
        if args.data_type == "image":
            plotting_callback = PlottingCallback(
                plot_interval=max(1, int(getattr(args, "check_val_every_n_epoch", 1))),
                datamodule=datamodule,
            )
            callbacks.append(plotting_callback)

        # 早期終了（任意）
        if getattr(args, "patience", None):
            early_stop_callback = EarlyStopping(
                monitor=monitor,
                patience=args.patience,
                mode=mode,
            )
            callbacks.append(early_stop_callback)

    return callbacks


class PlottingCallback(Callback):
    def __init__(self, plot_interval, datamodule):
        self.plot_interval = plot_interval
        self.datamodule = datamodule

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        pl_module.flow_net.train(mode=False)
        if epoch % self.plot_interval == 0 and epoch != 0:
            node = NeuralODE(
                flow_model_torch_wrapper(pl_module.flow_net).to(self.datamodule.device),
                solver="tsit5",
                sensitivity="adjoint",
                atol=1e-5,
                rtol=1e-5,
            )

            for mode in ["train", "val"]:
                x0 = getattr(self.datamodule, f"{mode}_x0")
                x0 = x0[0:15]
                fig = self.trajectory_and_plot(x0, node, self.datamodule)
                wandb.log({f"Trajectories {mode.capitalize()}": wandb.Image(fig)})
        pl_module.flow_net.train(mode=True)

    def trajectory_and_plot(self, x0, node, datamodule):
        selected_images = x0[0:15]
        with torch.no_grad():
            traj = node.trajectory(
                selected_images.to(datamodule.device),
                t_span=torch.linspace(0, 1, 100).to(datamodule.device),
            )

        traj = traj.transpose(0, 1)
        traj = traj.reshape(*traj.shape[0:2], *datamodule.dim)

        fig = plot_images_trajectory(
            traj.to(datamodule.device),
            datamodule.vae.to(datamodule.device),
            datamodule.process,
            num_steps=5,
        )
        return fig