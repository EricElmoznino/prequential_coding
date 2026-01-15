from typing import Any, Literal, get_args

import numpy as np
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class PCLDataset(Dataset):
    Mode = Literal["train", "val", "encode", "all_train"]

    def __init__(
        self,
        dataset: Dataset,
        min_data_size: int,
        val_ratio: float,
        data_increment: int | None = None,
        data_increment_n_log_intervals: int | None = None,
    ):
        """Dataset wrapper for prequential coding, which increments the dataset size correctly.

        Args:
            dataset (Dataset): The full dataset to wrap.
            min_data_size (int): Minimum data size on which to train the initial model.
            val_ratio (float): Ratio of the dataset to use as a validation set.
            batch_size (int): Batch size.
            data_increment (int | None, optional): Static amount to increase the dataset by each iteration. Smaller values more accurately estimate PCL, but fit more models. Argument is mutually exclusive with `data_increment_n_log_intervals`. Defaults to None.
            data_increment_n_log_intervals (int | None, optional): Number of logarithmic intervals to increase the dataset by each iteration. Argument is mutually exclusive with `data_increment`. Defaults to None.
        """
        if data_increment is None:
            assert data_increment_n_log_intervals is not None
        else:
            assert data_increment_n_log_intervals is None

        super().__init__()
        self.dataset = dataset
        self.min_data_size = min_data_size
        self.data_increment = data_increment
        self.data_increment_n_log_intervals = data_increment_n_log_intervals

        self.total_size = len(self.dataset)
        self.train_size = self.total_size - int(self.total_size * val_ratio)
        self.val_size = self.total_size - self.train_size
        if data_increment is not None:
            self.data_sizes = np.arange(
                min_data_size, self.train_size + 1, data_increment
            )
        else:
            self.data_sizes = np.logspace(
                np.log10(min_data_size),
                np.log10(self.train_size),
                data_increment_n_log_intervals,
                endpoint=True,
                base=10,
                dtype=int,
            )
        self.data_size_idx = 0

        self.mode = "train"
        assert (
            len(self.data_sizes) > 1
        ), "Data size too small; must be incremented at least once"

        idx_scrambled = list(range(self.total_size))
        np.random.shuffle(idx_scrambled)
        self._idx_map = lambda idx: idx_scrambled[idx]

    def set_mode(self, mode: Mode):
        assert mode in get_args(PCLDataset.Mode)
        if mode == "encode":
            assert self.data_size_idx > 0
        elif mode == "all_train":
            assert self.done
        self.mode = mode

    @property
    def done(self) -> bool:
        return self.data_size_idx >= len(self.data_sizes) - 1

    @property
    def data_encoded(self) -> int:
        return self.data_sizes[self.data_size_idx]

    @property
    def prev_data_encoded(self) -> int:
        if self.data_size_idx == 0:
            return 0
        return self.data_sizes[self.data_size_idx - 1]

    def increment_data_size(self):
        assert not self.done
        self.data_size_idx += 1

    def __len__(self) -> int:
        # Only data currently being trained on
        if self.mode == "train":
            return self.data_sizes[self.data_size_idx]
        # Data that is unseen and will never be encoded, used to decide when to move to the next data interval
        elif self.mode == "val":
            return self.val_size
        # Only next data interval, used for prequential code length
        elif self.mode == "encode":
            return (
                self.data_sizes[self.data_size_idx]
                - self.data_sizes[self.data_size_idx - 1]
            )
        elif self.mode == "all_train":
            return self.train_size
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def __getitem__(self, index) -> Any:
        if isinstance(index, slice):
            index = list(range(*index.indices(len(self))))
        else:
            index = [index]
        if self.mode == "val":
            index = [i + self.train_size for i in index]
        elif self.mode == "encode":
            index = [i + self.data_sizes[self.data_size_idx - 1] for i in index]
        index = [self._idx_map(i) for i in index]
        if len(index) == 1:
            index = index[0]
        return self.dataset[index]


class PCLDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        min_data_size: int,
        val_ratio: float,
        batch_size: int,
        data_increment: int | None = None,
        data_increment_n_log_intervals: int | None = None,
        num_workers: int = 0,
    ):
        """Lightning data module for prequential coding, which increments the dataset size correctly.

        Args:
            dataset (Dataset): The full dataset to wrap.
            min_data_size (int): Minimum data size on which to train the initial model.
            val_ratio (float): Ratio of the dataset to use as a validation set.
            batch_size (int): Batch size.
            data_increment (int | None, optional): Static amount to increase the dataset by each iteration. Smaller values more accurately estimate PCL, but fit more models. Argument is mutually exclusive with `data_increment_n_log_intervals`. Defaults to None.
            data_increment_n_log_intervals (int | None, optional): Number of logarithmic intervals to increase the dataset by each iteration. Argument is mutually exclusive with `data_increment`. Defaults to None.
            num_workers (int, optional): Number of worker processes for data loading. Defaults to 0.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["dataset"])
        self.base_dataset = dataset

    def setup(self, stage: str) -> None:
        self.dataset = PCLDataset(
            dataset=self.base_dataset,
            min_data_size=self.hparams.min_data_size,
            val_ratio=self.hparams.val_ratio,
            data_increment=self.hparams.data_increment,
            data_increment_n_log_intervals=self.hparams.data_increment_n_log_intervals,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
