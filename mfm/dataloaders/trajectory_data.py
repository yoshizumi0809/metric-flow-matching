import numpy as np
import pytorch_lightning as pl
import scanpy as sc
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchcfm.optimal_transport import OTPlanSampler
from pytorch_lightning.utilities.combined_loader import CombinedLoader


class TemporalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args,
        skipped_datapoint=-1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_type = args.data_type
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.split_ratios = args.split_ratios
        self.max_dim = args.dim
        self.whiten = args.whiten
        self.skipped_datapoint = skipped_datapoint
        self._prepare_data()

    def _prepare_data(self):
        self.train_dataloaders = []
        self.val_dataloaders = []
        self.test_dataloaders = []
        self.metric_samples_dataloaders = []

        if self.data_type == "scrna":
            ds, labels, unique_labels = custom_load_dataset(
                self.data_path,
                max_dim=self.max_dim,
            )
        elif self.data_type == "arch":
            ds, labels, unique_labels = generate_arch_data()
        elif self.data_type == "sphere":
            ds, labels, unique_labels = generate_sphere_data()
        else:
            raise ValueError("Data type not recognized")
        if self.whiten:
            self.scaler = StandardScaler()
            ds = self.scaler.fit_transform(ds)

        ds_tensor = torch.tensor(ds, dtype=torch.float32)
        label_to_numeric = {label: idx for idx, label in enumerate(unique_labels)}
        frame_indices = {
            label_to_numeric[label]: (labels == label).nonzero()[0]
            for label in unique_labels
        }
        self.num_timesteps = len(unique_labels)

        min_frame_size = min([len(indices) for indices in frame_indices.values()])
        for label, indices in frame_indices.items():
            frame_data = ds_tensor[indices]
            split_index = int(len(frame_data) * self.split_ratios[0])

            if len(frame_data) - split_index < self.batch_size:
                split_index = len(frame_data) - self.batch_size
            shuffled_indices = torch.randperm(len(frame_data))
            frame_data = frame_data[shuffled_indices]
            train_data = frame_data[:split_index]
            val_data = frame_data[split_index:]
            self.train_dataloaders.append(
                DataLoader(
                    train_data,
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True,
                )
            )
            self.val_dataloaders.append(
                DataLoader(
                    val_data,
                    batch_size=self.batch_size,
                    shuffle=False,
                    drop_last=True,
                )
            )
            self.test_dataloaders.append(
                DataLoader(
                    frame_data,
                    batch_size=frame_data.shape[0],
                    shuffle=False,
                    drop_last=False,
                )
            )
            self.metric_samples_dataloaders.append(
                DataLoader(
                    frame_data,
                    batch_size=min_frame_size,
                    shuffle=True,
                    drop_last=False,
                )
            )

    def train_dataloader(self):
        combined_loaders = {
            "train_samples": CombinedLoader(self.train_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(
                self.metric_samples_dataloaders, mode="min_size"
            ),
        }
        return CombinedLoader(combined_loaders, mode="max_size_cycle")

    def val_dataloader(self):
        combined_loaders = {
            "val_samples": CombinedLoader(self.val_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(
                self.metric_samples_dataloaders, mode="min_size"
            ),
        }

        return CombinedLoader(combined_loaders, mode="max_size_cycle")

    def test_dataloader(self):
        return CombinedLoader(self.test_dataloaders, "max_size")


def adata_dataset(
    path: str,
    embed_name: str = "X_pca",
    label_name: str = "day",
    max_dim: int = 100,
):
    """Load Single Cell dataset from h5ad file using scanpy."""

    adata = sc.read_h5ad(path)
    labels = adata.obs[label_name].astype("category")
    ulabels = labels.cat.categories
    data = adata.obsm[embed_name][:, :max_dim]

    return (data, labels.to_numpy(), ulabels.to_numpy())


def tnet_dataset(
    path: str,
    embed_name: str = "pcs",
    label_name: str = "sample_labels",
    max_dim: int = 100,
):
    """Load Single Cell dataset from npz file."""

    data_dict = np.load(path, allow_pickle=True)
    data = data_dict[embed_name][:, :max_dim]
    labels = data_dict[label_name]
    unique_labels = np.unique(labels)
    return data, labels, unique_labels


def custom_load_dataset(path: str, max_dim: int = 100):
    if path.endswith("h5ad"):
        return adata_dataset(path, max_dim=max_dim)
    if path.endswith("npz"):
        return tnet_dataset(path, max_dim=max_dim)
    raise NotImplementedError(f"File extension not supported for path: {path}")


def generate_arch_data(num_points: int = 5000):
    """Generate synthetic data for the arch dataset."""

    time_0_samples = np.abs(
        np.random.normal(loc=0, scale=1 / (2 * np.pi), size=num_points)
    )
    time_2_samples = 1 - np.abs(
        np.random.normal(loc=0, scale=1 / (2 * np.pi), size=num_points)
    )

    x0_ot, x1_ot = OTPlanSampler(method="exact").sample_plan(
        torch.tensor(time_0_samples).unsqueeze(0),
        torch.tensor(time_2_samples).unsqueeze(0),
        replace=False,
    )
    x0_ot, x1_ot = x0_ot.numpy().flatten(), x1_ot.numpy().flatten()
    time_1_samples = (x0_ot + x1_ot) / 2

    # Mapping to a semi-circle
    angles_0 = np.pi * (1 - time_0_samples)
    angles_1 = np.pi * (1 - time_1_samples)
    angles_2 = np.pi * (1 - time_2_samples)

    x_0 = np.cos(angles_0)
    y_0 = np.sin(angles_0)
    x_1 = np.cos(angles_1)
    y_1 = np.sin(angles_1)
    x_2 = np.cos(angles_2)
    y_2 = np.sin(angles_2)

    # Adding Gaussian noise
    radius_noise_0 = np.random.normal(0, 0.1, size=num_points)
    radius_noise_1 = np.random.normal(0, 0.1, size=num_points)
    radius_noise_2 = np.random.normal(0, 0.1, size=num_points)

    x_0 = (1 + radius_noise_0) * x_0
    y_0 = (1 + radius_noise_0) * y_0
    x_1 = (1 + radius_noise_1) * x_1
    y_1 = (1 + radius_noise_1) * y_1
    x_2 = (1 + radius_noise_2) * x_2
    y_2 = (1 + radius_noise_2) * y_2

    # Combining points and creating labels
    points_0 = np.column_stack((x_0, y_0))
    points_1 = np.column_stack((x_1, y_1))
    points_2 = np.column_stack((x_2, y_2))

    points = np.concatenate([points_0, points_1, points_2])
    labels = np.array([0] * num_points + [1] * num_points + [2] * num_points)

    # Returning the dataset, labels, and unique labels
    unique_labels = np.unique(labels)
    return points, labels, unique_labels


def generate_sphere_data(num_points: int = 5000):
    time_0_samples = np.abs(
        np.random.normal(loc=0, scale=1 / (2 * np.pi), size=num_points)
    )
    time_2_samples = 1 - np.abs(
        np.random.normal(loc=0, scale=1 / (2 * np.pi), size=num_points)
    )

    x0_ot, x1_ot = OTPlanSampler(method="exact").sample_plan(
        torch.tensor(time_0_samples).unsqueeze(0),
        torch.tensor(time_2_samples).unsqueeze(0),
        replace=False,
    )
    x0_ot, x1_ot = x0_ot.numpy().flatten(), x1_ot.numpy().flatten()
    time_1_samples = (x0_ot + x1_ot) / 2

    phi_0 = np.pi * time_0_samples
    phi_1 = np.pi * time_1_samples
    phi_2 = np.pi * time_2_samples

    theta_0 = 2 * np.pi * np.random.rand(num_points)
    theta_1 = 2 * np.pi * np.random.rand(num_points)
    theta_2 = 2 * np.pi * np.random.rand(num_points)

    x_0 = np.sin(phi_0) * np.cos(theta_0)
    y_0 = np.sin(phi_0) * np.sin(theta_0)
    z_0 = np.cos(phi_0)
    x_1 = np.sin(phi_1) * np.cos(theta_1)
    y_1 = np.sin(phi_1) * np.sin(theta_1)
    z_1 = np.cos(phi_1)
    x_2 = np.sin(phi_2) * np.cos(theta_2)
    y_2 = np.sin(phi_2) * np.sin(theta_2)
    z_2 = np.cos(phi_2)

    # Combining points and creating labels
    points_0 = np.column_stack((x_0, y_0, z_0))
    points_1 = np.column_stack((x_1, y_1, z_1))
    points_2 = np.column_stack((x_2, y_2, z_2))

    points = np.concatenate([points_0, points_1, points_2])
    labels = np.array([0] * num_points + [1] * num_points + [2] * num_points)

    unique_labels = np.unique(labels)
    return points, labels, unique_labels