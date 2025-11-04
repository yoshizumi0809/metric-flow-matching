import os
from typing import Any, Optional

from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import Dataset
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL
from torchvision.datasets import ImageFolder

LABELS_MAP = {
    "afhq": {
        "cat": 0,
        "dog": 1,
        "wild": 2,
    },
}


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.data_name = args.data_name

        self.image_size = args.image_size
        self.x0_label = args.x0_label
        self.x1_label = args.x1_label

        self.num_timesteps = 2

        self.device = args.accelerator
        if args.accelerator == "gpu":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "mps"

        self.process = VaeImageProcessor(do_convert_rgb=True)
        self.ambient_transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ]
        )
        self.latent_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True
        )

        self.train_ambient_save_path = os.path.join(
            self.data_path,
            f"{self.data_name}_train_ambient_dataset_64.pt",
        )
        self.val_ambient_save_path = os.path.join(
            self.data_path,
            f"{self.data_name}_val_ambient_dataset_64.pt",
        )
        self.train_latent_save_path = os.path.join(
            self.data_path,
            f"{self.data_name}_train_latent_dataset_{self.image_size}.pt",
        )
        self.val_latent_save_path = os.path.join(
            self.data_path,
            f"{self.data_name}_val_latent_dataset_{self.image_size}.pt",
        )

        required_files = [
            self.train_ambient_save_path,
            self.val_ambient_save_path,
            self.train_latent_save_path,
            self.val_latent_save_path,
        ]

        # ============ Gaussian モード追加（修正版） ============
        if self.x0_label == "gaussian":
            print("[INFO] x0_label='gaussian' detected → sampling x0 from standard normal distribution.")

            # 1) 先に latent データをロードして latent 形状を把握（C,H,W を得る）
            if all(os.path.exists(file_path) for file_path in required_files):
                self._load_latent_space()
            else:
                self._get_latent_space()

            # latent 形状（例: [4, 64, 64]）
            latent_shape = tuple(self.data_train["mean"].shape[1:])  # (C,H,W)
            # 念のため C をチェック（SDのVAEなら 4 が想定）
            if len(latent_shape) != 3:
                raise RuntimeError(f"[gaussian] unexpected latent shape: {latent_shape}")

            # 2) x0 を標準正規で生成（[N, C, H, W]）
            num_samples = max(10000, self.batch_size * 100)  # バッファ多め
            self.ambient_x0 = torch.randn((num_samples, *latent_shape))

            # 3) ambient（実画像）もロードするが、x0 は上書きしないように注意
            if all(os.path.exists(file_path) for file_path in required_files):
                self._load_ambient_space(gaussian_mode=True)
            else:
                self._get_ambient_space()
                # _get_ambient_space は ambient_x0/1 を作るので x0 を上書き回避
                # → ambient_x1 だけここで再設定しておく
                ambient_data = torch.load(self.train_ambient_save_path)
                self.ambient_x1 = ambient_data["mean"][
                    ambient_data["label"] == LABELS_MAP[self.data_name][self.x1_label]
                ]

            # 4) 最後にデータ準備（gaussian_mode=True）
            self._prepare_data(gaussian_mode=True)
            return
        # =====================================================

        # 通常モード
        if all(os.path.exists(file_path) for file_path in required_files):
            self._load_ambient_space()
            self._load_latent_space()
        else:
            self._get_ambient_space()
            self._get_latent_space()

        self._prepare_data()

    def image_base_dataset(self, split, transform=None):
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

    def _load_ambient_space(self, gaussian_mode: bool = False) -> None:
        ambient_data = torch.load(self.train_ambient_save_path)

        # x1 は常にデータセットから取得
        self.ambient_x1 = ambient_data["mean"][
            ambient_data["label"] == LABELS_MAP[self.data_name][self.x1_label]
        ]

        if gaussian_mode:
            # x0 は既にガウシアンで生成済み。上書きしない！
            return

        # 通常モードのみ x0 をデータセットから取得
        self.ambient_x0 = ambient_data["mean"][
            ambient_data["label"] == LABELS_MAP[self.data_name][self.x0_label]
        ]

    def _load_latent_space(self) -> None:
        self.data_train = torch.load(self.train_latent_save_path)
        self.data_val = torch.load(self.val_latent_save_path)

    def _get_latent_space(self) -> None:
        self.base_train_dataset = self.image_base_dataset("train", self.latent_transform)
        self.base_val_dataset = self.image_base_dataset("val", self.latent_transform)

        self.data_train = self._process_and_get_latent(
            self.base_train_dataset, self.train_latent_save_path
        )
        self.data_val = self._process_and_get_latent(
            self.base_val_dataset, self.val_latent_save_path
        )

    def _get_ambient_space(self) -> None:
        self.base_train_dataset = self.image_base_dataset("train", self.ambient_transform)
        self.base_val_dataset = self.image_base_dataset("val", self.ambient_transform)

        self.data_train = self._process_and_save_ambient(
            self.base_train_dataset, self.train_ambient_save_path
        )
        self.data_val = self._process_and_save_ambient(
            self.base_val_dataset, self.val_ambient_save_path
        )

    def _process_and_save_ambient(self, dataset, save_path):
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)
        images, labels = [], []
        with torch.no_grad():
            for image, batch_labels in tqdm(dataloader, desc="Processing Dataset"):
                images.append(image)
                labels.append(batch_labels)
        image_tensor = torch.cat(images, dim=0)
        label_tensor = torch.cat(labels, dim=0)
        ambient_dict = {"mean": image_tensor, "label": label_tensor}
        torch.save(ambient_dict, save_path)
        print(f"Ambient data saved at: {save_path}")
        return ambient_dict

    def _process_and_get_latent(self, dataset, save_path):
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        means, labels = [], []
        with torch.no_grad():
            for images, batch_labels in tqdm(data_loader, desc="Processing Dataset"):
                images = self.process.preprocess(images).to(self.device)
                outputs = self.vae.to(self.device).encode(images).latent_dist
                means.append(outputs.mean.detach().cpu())
                labels.append(batch_labels)
        mean_tensor = torch.cat(means, dim=0)
        label_tensor = torch.cat(labels, dim=0)
        latent_dict = {"mean": mean_tensor, "label": label_tensor}
        torch.save(latent_dict, save_path)
        print(f"Latent data saved at: {save_path}")
        return latent_dict

    def _prepare_data(self, gaussian_mode: bool = False) -> None:
        if gaussian_mode:
            # gaussian のとき：train_x0/val_x0 はガウシアン latent、x1 は通常の latent
            train_x0 = self.ambient_x0  # [N, C, H, W] のガウシアン
            train_x1 = self.data_train["mean"][
                self.data_train["label"] == LABELS_MAP[self.data_name][self.x1_label]
            ]
            val_x1 = self.data_val["mean"][
                self.data_val["label"] == LABELS_MAP[self.data_name][self.x1_label]
            ]

            # 形状そろえる
            self.dim = train_x1.shape[1:]  # (C,H,W) e.g. (4,64,64)

            # val_x0 は件数を val_x1 に合わせてスライス
            need = val_x1.shape[0]
            if train_x0.shape[0] < need:
                reps = (need + train_x0.shape[0] - 1) // train_x0.shape[0]
                train_x0 = train_x0.repeat((reps, 1, 1, 1))
            val_x0 = train_x0[:need]

            self.train_x0 = train_x0
            self.train_x1 = train_x1
            self.val_x0 = val_x0
            self.val_x1 = val_x1
        else:
            train_x0 = self.data_train["mean"][
                self.data_train["label"] == LABELS_MAP[self.data_name][self.x0_label]
            ]
            train_x1 = self.data_train["mean"][
                self.data_train["label"] == LABELS_MAP[self.data_name][self.x1_label]
            ]
            train_wild = self.data_train["mean"][
                self.data_train["label"] == LABELS_MAP[self.data_name]["wild"]
            ]
            val_x0 = self.data_val["mean"][
                self.data_val["label"] == LABELS_MAP[self.data_name][self.x0_label]
            ]
            val_x1 = self.data_val["mean"][
                self.data_val["label"] == LABELS_MAP[self.data_name][self.x1_label]
            ]
            self.all_data = torch.cat([train_x0, train_x1, train_wild], dim=0)

            self.dim = train_x0.shape[1:]
            self.train_x0 = train_x0
            self.train_x1 = train_x1
            self.val_x0 = val_x0
            self.val_x1 = val_x1

        self.train_dataloaders = [
            DataLoader(self.train_x0, batch_size=self.batch_size, shuffle=True, drop_last=True),
            DataLoader(self.train_x1, batch_size=self.batch_size, shuffle=True, drop_last=True),
        ]
        self.val_dataloaders = [
            DataLoader(self.val_x0, batch_size=min(self.batch_size, self.val_x0.shape[0]),
                       shuffle=False, drop_last=True),
            DataLoader(self.val_x1, batch_size=min(self.batch_size, self.val_x0.shape[0]),
                       shuffle=False, drop_last=True),
        ]
        self.metric_samples_dataloaders = [
            DataLoader(torch.Tensor([0]), batch_size=1, shuffle=False, drop_last=False),
            DataLoader(torch.Tensor([0]), batch_size=1, shuffle=False, drop_last=False),
        ]

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