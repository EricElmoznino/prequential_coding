from typing import Any, Literal, Sequence, get_args

import jax
import jax.numpy as jnp
import numpy as np


def stack_batch(samples: Sequence[Any]) -> Any:
    if not samples:
        raise ValueError("Cannot stack an empty batch.")
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *samples)


class ArrayDataset:
    """Simple dataset wrapper for numpy-like arrays."""

    def __init__(self, *arrays: np.ndarray):
        if len(arrays) == 0:
            raise ValueError("ArrayDataset requires at least one array.")
        size = len(arrays[0])
        for array in arrays:
            if len(array) != size:
                raise ValueError("All arrays must have the same length.")
        self._arrays = [np.asarray(array) for array in arrays]
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: Any) -> Any:
        if len(self._arrays) == 1:
            return self._arrays[0][index]
        return tuple(array[index] for array in self._arrays)


class PCLDataset:
    Mode = Literal["train", "val", "encode", "all_train"]

    def __init__(
        self,
        dataset: Any,
        min_data_size: int,
        val_ratio: float,
        data_increment: int | None = None,
        data_increment_n_log_intervals: int | None = None,
    ):
        """Dataset wrapper for prequential coding, which increments the dataset size correctly.

        Args:
            dataset (Any): The full dataset to wrap. Must implement __len__ and __getitem__.
            min_data_size (int): Minimum data size on which to train the initial model.
            val_ratio (float): Ratio of the dataset to use as a validation set.
            data_increment (int | None, optional): Static amount to increase the dataset by each iteration. Smaller values more accurately estimate PCL, but fit more models. Argument is mutually exclusive with `data_increment_n_log_intervals`. Defaults to None.
            data_increment_n_log_intervals (int | None, optional): Number of logarithmic intervals to increase the dataset by each iteration. Argument is mutually exclusive with `data_increment`. Defaults to None.
        """
        if data_increment is None:
            assert data_increment_n_log_intervals is not None
        else:
            assert data_increment_n_log_intervals is None

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

        self.mode: PCLDataset.Mode = "train"
        assert (
            len(self.data_sizes) > 1
        ), "Data size too small; must be incremented at least once"

        idx_scrambled = list(range(self.total_size))
        np.random.shuffle(idx_scrambled)
        self._idx_map = lambda idx: idx_scrambled[idx]

    def set_mode(self, mode: Mode) -> None:
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
        return int(self.data_sizes[self.data_size_idx])

    @property
    def prev_data_encoded(self) -> int:
        if self.data_size_idx == 0:
            return 0
        return int(self.data_sizes[self.data_size_idx - 1])

    def increment_data_size(self) -> None:
        assert not self.done
        self.data_size_idx += 1

    def __len__(self) -> int:
        if self.mode == "train":
            return int(self.data_sizes[self.data_size_idx])
        elif self.mode == "val":
            return int(self.val_size)
        elif self.mode == "encode":
            return int(
                self.data_sizes[self.data_size_idx]
                - self.data_sizes[self.data_size_idx - 1]
            )
        elif self.mode == "all_train":
            return int(self.train_size)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            index = list(range(*index.indices(len(self))))
        elif isinstance(index, (list, tuple, np.ndarray)):
            index = list(index)
        else:
            index = [int(index)]

        if self.mode == "val":
            index = [i + self.train_size for i in index]
        elif self.mode == "encode":
            index = [i + self.data_sizes[self.data_size_idx - 1] for i in index]

        index = [self._idx_map(int(i)) for i in index]

        if len(index) == 1:
            return self.dataset[index[0]]

        try:
            batch = self.dataset[index]
            if isinstance(batch, list):
                return stack_batch(batch)
            return jax.tree_util.tree_map(jnp.asarray, batch)
        except Exception:
            samples = [self.dataset[i] for i in index]
            return stack_batch(samples)


class BatchIterator:
    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        rng: np.random.Generator | None = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = rng

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            rng = self.rng if self.rng is not None else np.random.default_rng()
            rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            end = start + self.batch_size
            if end > len(indices) and self.drop_last:
                break
            batch_indices = indices[start:end]
            batch = self.dataset[batch_indices]
            if isinstance(batch, list):
                batch = stack_batch(batch)
            else:
                batch = jax.tree_util.tree_map(jnp.asarray, batch)
            yield batch


class PCLDataModule:
    def __init__(
        self,
        dataset: Any,
        min_data_size: int,
        val_ratio: float,
        batch_size: int,
        data_increment: int | None = None,
        data_increment_n_log_intervals: int | None = None,
        num_workers: int = 0,
    ):
        """Data module for prequential coding, which increments the dataset size correctly.

        Args:
            dataset (Any): The full dataset to wrap. Must implement __len__ and __getitem__.
            min_data_size (int): Minimum data size on which to train the initial model.
            val_ratio (float): Ratio of the dataset to use as a validation set.
            batch_size (int): Batch size.
            data_increment (int | None, optional): Static amount to increase the dataset by each iteration. Smaller values more accurately estimate PCL, but fit more models. Argument is mutually exclusive with `data_increment_n_log_intervals`. Defaults to None.
            data_increment_n_log_intervals (int | None, optional): Number of logarithmic intervals to increase the dataset by each iteration. Argument is mutually exclusive with `data_increment`. Defaults to None.
            num_workers (int, optional): Number of worker processes for data loading. Ignored for JAX. Defaults to 0.
        """
        self.base_dataset = dataset
        self.min_data_size = min_data_size
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.data_increment = data_increment
        self.data_increment_n_log_intervals = data_increment_n_log_intervals
        self.num_workers = num_workers
        self.dataset: PCLDataset | None = None

    def setup(self) -> None:
        self.dataset = PCLDataset(
            dataset=self.base_dataset,
            min_data_size=self.min_data_size,
            val_ratio=self.val_ratio,
            data_increment=self.data_increment,
            data_increment_n_log_intervals=self.data_increment_n_log_intervals,
        )

    def train_dataloader(
        self,
        shuffle: bool = True,
        drop_last: bool = False,
        rng: np.random.Generator | None = None,
    ) -> BatchIterator:
        if self.dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return BatchIterator(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            rng=rng,
        )

    def val_dataloader(
        self,
        shuffle: bool = False,
        drop_last: bool = False,
        rng: np.random.Generator | None = None,
    ) -> BatchIterator:
        if self.dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return BatchIterator(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            rng=rng,
        )
