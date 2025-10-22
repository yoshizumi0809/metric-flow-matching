import argparse
import copy
import os

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb

from torchcfm.optimal_transport import OTPlanSampler

from mfm.flow_matchers.models.mfm import MetricFlowMatcher
from mfm.geo_metrics.metric_factory import DataManifoldMetric
from mfm.flow_matchers.flow_net_train import (
    FlowNetTrainTrajectory,
    FlowNetTrainLidar,
    FlowNetTrainImage,
)
from mfm.flow_matchers.geopath_net_train import GeoPathNetTrain
from mfm.dataloaders.trajectory_data import TemporalDataModule
from mfm.dataloaders.image_data import ImageDataModule
from mfm.dataloaders.lidar_data import LidarDataModule
from mfm.networks.flow_networks.mlp import VelocityNet
from mfm.networks.geopath_networks.mlp import GeoPathMLP
from mfm.networks.unet_base import UNetModelWrapper as UNetModel
from mfm.networks.geopath_networks.unet import GeoPathUNet
from mfm.utils import set_seed
from mfm.train.parsers import parse_args
from mfm.flow_matchers.ema import EMA
from mfm.train.train_utils import (
    load_config,
    merge_config,
    generate_group_string,
    dataset_name2datapath,
    create_callbacks,
)


def main(args: argparse.Namespace, seed: int, t_exclude: int) -> None:
    set_seed(seed)
    if args.data_type == "lidar":
        assert args.dim == 3 and args.data_name == "lidar"
    elif args.data_type == "arch":
        assert args.dim == 2
    elif args.data_type == "sphere":
        assert args.dim == 3
    elif args.data_type == "image":
        assert not args.whiten
        assert args.data_name == "afhq"

    skipped_time_points = [t_exclude] if t_exclude else []

    # ======================
    # DATAMODULES
    # ======================
    if args.data_type in ["arch", "scrna", "sphere"]:
        datamodule = TemporalDataModule(
            args=args,
            skipped_datapoint=t_exclude,
        )
    elif args.data_type == "lidar":
        datamodule = LidarDataModule(args=args)
    elif args.data_type == "image":
        datamodule = ImageDataModule(args=args)
    else:
        raise ValueError("Data type not recognized")

    # ======================
    # Networks
    # ======================
    if args.data_type in ["arch", "scrna", "lidar", "sphere"]:
        flow_net = VelocityNet(
            dim=args.dim,
            hidden_dims=args.hidden_dims_flow,
            activation=args.activation_flow,
            batch_norm=False,
        )
        geopath_net = GeoPathMLP(
            input_dim=args.dim,
            hidden_dims=args.hidden_dims_geopath,
            time_geopath=args.time_geopath,
            activation=args.activation_geopath,
            batch_norm=False,
        )
    elif args.data_type == "image":
        flow_net = UNetModel(
            geopath_model=False,
            dim=datamodule.dim,
            num_channels=args.unet_num_channels,
            num_res_blocks=args.unet_num_res_blocks,
            channel_mult=args.unet_channel_mult,
            dropout=args.unet_dropout,
            resblock_updown=args.unet_resblock_updown,
            use_new_attention_order=args.unet_use_new_attention_order,
            attention_resolutions=args.unet_attention_resolutions,
            num_heads=args.unet_num_heads,
        )
        geopath_net = GeoPathUNet(
            geopath_model=True,
            dim=datamodule.dim,
            num_channels=args.unet_num_channels_geopath,
            num_res_blocks=args.unet_num_res_blocks_geopath,
            channel_mult=args.unet_channel_mult_geopath,
            dropout=args.unet_dropout_geopath,
            use_checkpoint=False,
        )

    if args.ema_decay is not None:
        flow_net = EMA(model=flow_net, decay=args.ema_decay)
        geopath_net = EMA(model=geopath_net, decay=args.ema_decay)

    ot_sampler = (
        OTPlanSampler(method=args.optimal_transport_method)
        if args.optimal_transport_method != "None"
        else None
    )

    # ======================
    # WandB (run 名に experiment_name を反映)
    # ======================
    wandb.init(
        project=f"mfm-{args.data_type}-{args.data_name}",
        group=args.group_name,
        name=(args.experiment_name if getattr(args, "experiment_name", None) else None),
        config=vars(args),
        dir=args.working_dir,
    )
    # 実際にディレクトリ名として使うキー（experiment_name がなければ run.id）
    exp_key = args.experiment_name or wandb.run.id

    # ======================
    # Metric Flow Matching module
    # ======================
    flow_matcher_base = MetricFlowMatcher(
        geopath_net=geopath_net,
        sigma=args.sigma,
        alpha=int(args.mfm),
    )

    # ======================
    # ALGO 1: GeoPath Training
    # ======================
    if args.mfm:
        data_manifold_metric = DataManifoldMetric(
            args=args,
            skipped_time_points=skipped_time_points,
            datamodule=datamodule,
        )
        # 保存先は train_utils.create_callbacks 側で experiment_name を解決
        geopath_callbacks = create_callbacks(
            args, phase="geopath", data_type=args.data_type, run_id=exp_key
        )

        geopath_model = GeoPathNetTrain(
            flow_matcher=flow_matcher_base,
            skipped_time_points=skipped_time_points,
            ot_sampler=ot_sampler,
            data_manifold_metric=data_manifold_metric,
            args=args,
        )
        wandb_logger = WandbLogger()

        trainer = Trainer(
            max_epochs=args.epochs,
            callbacks=geopath_callbacks,
            accelerator=args.accelerator,
            logger=wandb_logger,
            num_sanity_val_steps=0,
            default_root_dir=args.working_dir,
            gradient_clip_val=(1.0 if args.data_type == "image" else None),
        )

        # 参照する（あるいは学習して得た）GeoPath ckpt
        if args.load_geopath_model_ckpt:
            best_model_path = args.load_geopath_model_ckpt
        else:
            trainer.fit(geopath_model, datamodule=datamodule)
            best_model_path = geopath_callbacks[0].best_model_path

        geopath_model = GeoPathNetTrain.load_from_checkpoint(best_model_path)

        # GeoPath の学習済みを Flow 側に反映
        flow_matcher_base.geopath_net = geopath_model.geopath_net

        # ログ：GeoPath ckpt の保存先を明示
        geopath_dir = os.path.join(
            args.working_dir, "checkpoints", args.data_type, exp_key, "geopath_model"
        )
        print(f"[INFO] GeoPath checkpoints → {geopath_dir}")

    # ======================
    # ALGO 2: Flow Matching Training / Testing
    # ======================
    if args.data_type in ["arch", "scrna", "sphere"]:
        datamodule = TemporalDataModule(
            args=args,
            skipped_datapoint=t_exclude,
        )

    flow_callbacks = create_callbacks(
        args,
        phase="flow",
        data_type=args.data_type,
        run_id=exp_key,
        datamodule=datamodule,
    )

    if args.data_type in ["arch", "scrna", "sphere"]:
        FlowNetTrain = FlowNetTrainTrajectory
    elif args.data_type == "lidar":
        FlowNetTrain = FlowNetTrainLidar
    elif args.data_type == "image":
        FlowNetTrain = FlowNetTrainImage
    else:
        raise ValueError("Data type not recognized")

    flow_train = FlowNetTrain(
        flow_matcher=flow_matcher_base,
        flow_net=flow_net,
        ot_sampler=ot_sampler,
        skipped_time_points=skipped_time_points,
        args=args,
    )

    wandb_logger = WandbLogger()

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=flow_callbacks,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        accelerator=args.accelerator,
        logger=wandb_logger,
        default_root_dir=args.working_dir,
        gradient_clip_val=(1.0 if args.data_type == "image" else None),
        num_sanity_val_steps=(0 if args.data_type == "image" else None),
    )

    # ======== 学習 or テストのみ ========
    if getattr(args, "only_test_flow", False):
        ckpt_for_test = args.resume_flow_model_ckpt or "last"
        print(f"[INFO] Test-only mode. Flow ckpt = {ckpt_for_test}")
        trainer.test(flow_train, datamodule=datamodule, ckpt_path=ckpt_for_test)
    else:
        trainer.fit(
            flow_train, datamodule=datamodule, ckpt_path=args.resume_flow_model_ckpt
        )
        trainer.test(flow_train, datamodule=datamodule, ckpt_path="last")

    # ログ：Flow ckpt の保存先を明示
    flow_dir = os.path.join(
        args.working_dir, "checkpoints", args.data_type, exp_key, "flow_model"
    )
    print(f"[INFO] Flow checkpoints → {flow_dir}")

    # 画像の保存先（FlowNetTrainImage 側で experiment_name を使っていればこのパスになる）
    samples_dir = os.path.join(
        args.working_dir,
        "generated_samples",
        (args.experiment_name if getattr(args, "experiment_name", None) else ""),
    )
    print(f"[INFO] Generated samples → {samples_dir}")

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    updated_args = copy.deepcopy(args)
    if args.config_path:
        config = load_config(args.config_path)
        updated_args = merge_config(updated_args, config)

    updated_args.group_name = generate_group_string()
    updated_args.data_path = dataset_name2datapath(
        updated_args.data_name, updated_args.working_dir
    )
    for seed in updated_args.seeds:
        if updated_args.t_exclude:
            for i, t_exclude in enumerate(updated_args.t_exclude):
                updated_args.t_exclude_current = t_exclude
                updated_args.seed_current = seed
                updated_args.gamma_current = updated_args.gammas[i]
                main(updated_args, seed=seed, t_exclude=t_exclude)
        else:
            updated_args.seed_current = seed
            updated_args.gamma_current = updated_args.gammas[0]
            main(updated_args, seed=seed, t_exclude=None)