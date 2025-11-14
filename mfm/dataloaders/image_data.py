# mfm/dataloaders/image_data.py

import os
from typing import Optional, List

from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torchvision.datasets import ImageFolder

LABELS_MAP = {
    "afhq": {
        "cat": 0,
        "dog": 1,
        "wild": 2,
    },
}


class ImageDataModule(pl.LightningDataModule):
    """
    Pixel-space DataModule:
      - 入力画像をそのまま RGB ピクセル空間 [B, 3, H, W] で扱う
      - VAE / diffusers 依存を完全に排除
      - x0_label == 'gaussian' の場合は RGB 標準正規ノイズを sigmoid で [0,1] に写像して使用
      - GeoPath/Metric 用に self.all_data（訓練全体のピクセルテンソル）を提供
    """

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.data_name = args.data_name

        self.image_size = args.image_size
        self.x0_label = args.x0_label
        self.x1_label = args.x1_label

        self.num_timesteps = 2  # 既存仕様を踏襲

        # device 情報（CPU/GPU/MPS）: ここでは主に torch.no_grad() の to() で使う想定
        self.device = args.accelerator
        if args.accelerator == "gpu":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "mps"

        # ピクセル空間の前処理（[0,1] Tensor 化）
        self.pixel_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),  # -> [0,1], CHW
            ]
        )

        # 低解像度（可視化・比較用）
        self.ambient_transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ]
        )

        # キャッシュ先（ピクセル空間 & 低解像度）
        self.train_pixel_save_path = os.path.join(
            self.data_path, f"{self.data_name}_train_pixel_{self.image_size}.pt"
        )
        self.val_pixel_save_path = os.path.join(
            self.data_path, f"{self.data_name}_val_pixel_{self.image_size}.pt"
        )
        self.train_ambient_save_path = os.path.join(
            self.data_path, f"{self.data_name}_train_ambient_64.pt"
        )
        self.val_ambient_save_path = os.path.join(
            self.data_path, f"{self.data_name}_val_ambient_64.pt"
        )

        required_files = [
            self.train_pixel_save_path,
            self.val_pixel_save_path,
            self.train_ambient_save_path,
            self.val_ambient_save_path,
        ]

        # キャッシュ構築 or 読み込み
        if all(os.path.exists(fp) for fp in required_files):
            self._load_pixel_space()
            self._load_ambient_space()
        else:
            self._build_and_save_pixel_space()
            self._build_and_save_ambient_space()

        # データ準備（x0=gaussian にも対応）
        self._prepare_data()

    # ------------------------------------------------------------
    # 基本データセット
    # ------------------------------------------------------------
    def image_base_dataset(
        self, split: str, transform: Optional[transforms.Compose] = None
    ):
        if split == "train":
            path = os.path.join(self.data_path, "train")
        elif split == "val":
            path = os.path.join(self.data_path, "val")
        else:
            raise NotImplementedError

        if transform is None:
            transform = transforms.ToTensor()
        dataset = ImageFolder(path, transform)
        return dataset

    # ------------------------------------------------------------
    # ピクセル空間キャッシュ
    # ------------------------------------------------------------
    def _build_and_save_pixel_space(self):
        base_train = self.image_base_dataset("train", self.pixel_transform)
        base_val = self.image_base_dataset("val", self.pixel_transform)

        def _to_tensor_dict(ds: ImageFolder, save_path: str):
            loader = DataLoader(ds, batch_size=64, shuffle=False, drop_last=False)
            images: List[torch.Tensor] = []
            labels: List[torch.Tensor] = []
            with torch.no_grad():
                for img, lab in tqdm(loader, desc=f"Building pixel cache ({os.path.basename(save_path)})"):
                    # img: [B,3,H,W] in [0,1]
                    images.append(img)
                    labels.append(lab)
            data = {
                "pixel": torch.cat(images, dim=0),   # [N,3,H,W]
                "label": torch.cat(labels, dim=0),   # [N]
            }
            torch.save(data, save_path)
            print(f"[Pixel cache] saved: {save_path}")
            return data

        self.data_train = _to_tensor_dict(base_train, self.train_pixel_save_path)
        self.data_val   = _to_tensor_dict(base_val,   self.val_pixel_save_path)

    def _load_pixel_space(self):
        self.data_train = torch.load(self.train_pixel_save_path)
        self.data_val   = torch.load(self.val_pixel_save_path)

    # ------------------------------------------------------------
    # 低解像度（可視化用）キャッシュ
    # ------------------------------------------------------------
    def _build_and_save_ambient_space(self):
        base_train = self.image_base_dataset("train", self.ambient_transform)
        base_val   = self.image_base_dataset("val",   self.ambient_transform)

        def _to_tensor_dict(ds: ImageFolder, save_path: str):
            loader = DataLoader(ds, batch_size=64, shuffle=False, drop_last=False)
            images: List[torch.Tensor] = []
            labels: List[torch.Tensor] = []
            with torch.no_grad():
                for img, lab in tqdm(loader, desc=f"Building ambient cache ({os.path.basename(save_path)})"):
                    images.append(img)  # [B,3,64,64]
                    labels.append(lab)
            data = {
                "mean":  torch.cat(images, dim=0),  # 命名は既存に合わせて mean を踏襲
                "label": torch.cat(labels, dim=0),
            }
            torch.save(data, save_path)
            print(f"[Ambient cache] saved: {save_path}")
            return data

        self.ambient_train = _to_tensor_dict(base_train, self.train_ambient_save_path)
        self.ambient_val   = _to_tensor_dict(base_val,   self.val_ambient_save_path)

    def _load_ambient_space(self):
        self.ambient_train = torch.load(self.train_ambient_save_path)
        self.ambient_val   = torch.load(self.val_ambient_save_path)

    # ------------------------------------------------------------
    # データ準備（train/val セット & all_data）
    # ------------------------------------------------------------
    def _prepare_data(self) -> None:
        """
        - x0_label == 'gaussian' のときは RGB ノイズを sigmoid で [0,1] に写像して使用
        - それ以外はデータセットの実画像を使用
        - self.all_data をセット（GeoPath/metric 用）
        """

        # x1 は常に実データから
        train_x1 = self.data_train["pixel"][
            self.data_train["label"] == LABELS_MAP[self.data_name][self.x1_label]
        ]
        val_x1 = self.data_val["pixel"][
            self.data_val["label"] == LABELS_MAP[self.data_name][self.x1_label]
        ]

        if self.x0_label == "gaussian":
            print("[INFO] x0_label='gaussian' (pixel) → sampling RGB Gaussian noise in [0,1].")
            N_train = train_x1.shape[0]
            N_val   = val_x1.shape[0]
            H, W = self.image_size, self.image_size

            # 標準正規→sigmoidで [0,1]
            train_x0 = torch.sigmoid(torch.randn(N_train, 3, H, W))
            val_x0   = torch.sigmoid(torch.randn(N_val,   3, H, W))
        else:
            # 実データから抽出
            train_x0 = self.data_train["pixel"][
                self.data_train["label"] == LABELS_MAP[self.data_name][self.x0_label]
            ]
            val_x0 = self.data_val["pixel"][
                self.data_val["label"] == LABELS_MAP[self.data_name][self.x0_label]
            ]

        # 形状情報（ピクセル空間）
        self.dim = train_x0.shape[1:]  # (3, H, W)

        # GeoPath / Metric が参照する全体データ（学習用の全ピクセル）
        # ※ 既存コードの期待に合わせて「train 全体」を突っ込みます
        #   （必要に応じて cat([train_x0, train_x1, train_wild]) でも可）
        self.all_data = self.data_train["pixel"]

        # 低解像度可視化用（存在しないラベルを参照しないよう guard）
        # x0
        if self.x0_label in LABELS_MAP[self.data_name]:
            self.ambient_x0 = self.ambient_train["mean"][
                self.ambient_train["label"] == LABELS_MAP[self.data_name][self.x0_label]
            ]
        else:
            # gaussian の場合など、可視化で使うならランダムに作ってもよいが、
            # ここでは None にして Flow 側で使わない前提にする
            self.ambient_x0 = None
        # x1
        self.ambient_x1 = self.ambient_train["mean"][
            self.ambient_train["label"] == LABELS_MAP[self.data_name][self.x1_label]
        ]

        # 保存（trainer 側で参照）
        self.train_x0 = train_x0
        self.train_x1 = train_x1
        self.val_x0 = val_x0
        self.val_x1 = val_x1

        # DataLoader 群（既存の CombinedLoader 構成を踏襲）
        self.train_dataloaders = [
            DataLoader(self.train_x0, batch_size=self.batch_size, shuffle=True,  drop_last=True),
            DataLoader(self.train_x1, batch_size=self.batch_size, shuffle=True,  drop_last=True),
        ]
        self.val_dataloaders = [
            DataLoader(self.val_x0,   batch_size=min(self.batch_size, self.val_x0.shape[0]), shuffle=False, drop_last=True),
            DataLoader(self.val_x1,   batch_size=min(self.batch_size, self.val_x0.shape[0]), shuffle=False, drop_last=True),
        ]
        self.metric_samples_dataloaders = [
            DataLoader(torch.Tensor([0]), batch_size=1, shuffle=False, drop_last=False),
            DataLoader(torch.Tensor([0]), batch_size=1, shuffle=False, drop_last=False),
        ]

    # ------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------
    def train_dataloader(self):
        combined_loaders = {
            "train_samples": CombinedLoader(self.train_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(self.metric_samples_dataloaders, mode="min_size"),
        }
        return CombinedLoader(combined_loaders, mode="max_size_cycle")

    def val_dataloader(self):
        combined_loaders = {
            "val_samples": CombinedLoader(self.val_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(self.metric_samples_dataloaders, mode="min_size"),
        }
        return CombinedLoader(combined_loaders, mode="max_size_cycle")

    def test_dataloader(self):
        return DataLoader(
            self.val_x0,
            batch_size=16,
            shuffle=False,
            drop_last=False,
        )