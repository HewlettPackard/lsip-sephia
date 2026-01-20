import os, pandas as pd, pickle, numpy as np
import torch
from pathlib import Path
from ..scripts.utils import Config
from ..config import LOGGER, get_global_random
from torch.utils.data import DataLoader, Subset
from PIL import Image
from torchvision import datasets, transforms
import random
from sklearn.decomposition import PCA


### New (leaner) implementation
class ReducedMNIST:
    def __init__(
        self,
        batch_size,
        size_sqrt=6,
        N_samples_train=1024
        // 4,  # 4096*4,  # same length is used for validation,therefore this can be max (total_MNIST_samples/2)
        N_samples_val=1024 // 4,  # 4096*4,
        N_samples_test=256 // 4,  # 1024*4
        specific_classes={
            0,
            1,
            # 2,
            # 3,
            4,
            5,
            # 6,
            # 7,
            # 8,
            # 9,
        },  # Example: select only digits 0, 1, and 2
        flatten_img_dims=True,
        test_ds_only=False,  # Generates only test set, no train or val. Can provide speed-up for testing purposes.
        data_path="/tmp/data/mnist",
    ):
        self.size_sqrt = size_sqrt
        self.test_ds_only = test_ds_only

        assert N_samples_train + N_samples_val <= 60000, (
            f"Too many samples requested for training and validation combined {N_samples_train}+{N_samples_val}. The MNIST dataset has only 60,000 training samples."
        )

        #########################################

        self.initiate_transform(flatten_img_dims=flatten_img_dims)
        # Convert specific_classes to a tensor for efficient filtering
        specific_classes_tensor = torch.tensor(list(specific_classes))

        if not self.test_ds_only:
            mnist_train = datasets.MNIST(
                data_path, train=True, download=True, transform=self.transform
            )

            # Get indices of specific digits in one tensor operation
            train_indices = torch.where(
                torch.isin(mnist_train.targets, specific_classes_tensor)
            )[0].tolist()

            # Optional: shuffle indices for better randomization
            random_class.shuffle(train_indices)

            # Create the subsets using the filtered indices
            mnist_train_subset = Subset(mnist_train, train_indices[:N_samples_train])
            mnist_val_subset = Subset(
                mnist_train,
                train_indices[N_samples_train : N_samples_train + N_samples_val],
            )

            # Create DataLoaders
            self.train_loader = DataLoader(
                mnist_train_subset, batch_size=batch_size, shuffle=True, drop_last=True
            )

            self.val_loader = DataLoader(
                mnist_val_subset, batch_size=batch_size, shuffle=True, drop_last=True
            )
        mnist_test = datasets.MNIST(
            data_path, train=False, download=True, transform=self.transform
        )
        # Get indices of specific digits in one tensor operation
        test_indices = torch.where(
            torch.isin(mnist_test.targets, specific_classes_tensor)
        )[0].tolist()

        # Optional: shuffle indices for better randomization
        random_class.shuffle(test_indices)

        # Create the subsets using the filtered indices
        mnist_test_subset = Subset(mnist_test, test_indices[:N_samples_test])

        # Create DataLoaders
        self.test_loader = DataLoader(
            mnist_test_subset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        self.classes = list(specific_classes)

        print(f"FFT-R-MNIST scale: {size_sqrt}x{size_sqrt}")
        if not self.test_ds_only:
            print(
                f"Train set size: {len(self.train_loader)} batches, each of {batch_size} samples -> {len(self.train_loader) * batch_size} samples"
            )
            print(
                f"Val set size: {len(self.val_loader)} batches, each of {batch_size} samples -> {len(self.val_loader) * batch_size} samples"
            )
            self.calculate_train_class_weights()
            self.calculate_val_class_counts()

        print(
            f"Test set size: {len(self.test_loader)} batches, each of {batch_size} samples -> {len(self.test_loader) * batch_size} samples"
        )
        self.calculate_test_class_counts()

        print("Used classes: ", self.classes)
        print("OK.")

    def initiate_transform(self, flatten_img_dims=True):
        transforms_list = [
            transforms.CenterCrop(24),
            transforms.Lambda(self._fft_cutoff),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
        ]

        if flatten_img_dims:
            transforms_list.append(
                transforms.Lambda(lambda x: x.view(int(self.size_sqrt**2)))
            )

        self.transform = transforms.Compose(transforms_list)

    def calculate_train_class_weights(self):
        """
        Calculate weights for each class based on their frequency in the training dataset.
        Returns weights that are inversely proportional to class frequencies.
        """
        # Count instances of each class in the training dataset
        class_counts = torch.zeros(len(self.classes))

        for _, y in self.train_loader:
            for label in RemapTargets(y, self.classes):
                class_counts[label] += 1

        self.train_class_counts = class_counts.cpu().numpy()

        # Calculate weights (inverse of frequency)
        total_samples = class_counts.sum()
        class_weights = total_samples / (len(self.classes) * class_counts)

        # Normalize weights
        self.train_class_weights = (
            class_weights / class_weights.sum() * len(self.classes)
        )

    def calculate_val_class_counts(self):
        """
        Calculate weights for each class based on their frequency in the training dataset.
        Returns weights that are inversely proportional to class frequencies.
        """
        # Count instances of each class in the training dataset
        class_counts = torch.zeros(len(self.classes))

        for _, y in self.val_loader:
            for label in RemapTargets(y, self.classes):
                class_counts[label] += 1

        self.val_class_counts = class_counts.cpu().numpy()

    def calculate_test_class_counts(self):
        """
        Calculate weights for each class based on their frequency in the training dataset.
        Returns weights that are inversely proportional to class frequencies.
        """
        # Count instances of each class in the training dataset
        class_counts = torch.zeros(len(self.classes))

        for _, y in self.test_loader:
            for label in RemapTargets(y, self.classes):
                class_counts[label] += 1

        self.test_class_counts = class_counts.cpu().numpy()

    # Define a FFT cutoff function
    def _fft_cutoff(self, image):
        ret = np.fft.fftshift(np.fft.fft2(image))[
            image.size[0] // 2 - self.size_sqrt // 2 : image.size[0] // 2
            + self.size_sqrt // 2,
            image.size[1] // 2 - self.size_sqrt // 2 : image.size[1] // 2
            + self.size_sqrt // 2,
        ]
        return Image.fromarray(np.abs(ret) / np.max(np.abs(ret)) * 255)


### PCA-reduced FMNIST
class ReducedFMNIST:
    def __init__(
        self,
        batch_size,
        n_features=36,
        N_samples_train=21760,
        N_samples_val=2048,
        N_samples_test=3900,
        specific_classes={
            0,
            1,
            # 2,
            3,
            # 4,
            # 5,
            # 6,
            7,
            # 8,
            # 9,
        },
        test_ds_only=False,
        normalize_features=True,  # Added normalization parameter
        data_path="/tmp/data/fashion-mnist",
    ):
        """Classes:
        0: T-shirt/top
        1: Trouser
        2: Pullover
        3: Dress
        4: Coat
        5: Sandal
        6: Shirt
        7: Sneaker
        8: Bag
        9: Ankle boot"""

        random_class = get_global_random()

        self.test_ds_only = test_ds_only
        self.n_components = n_features
        self.normalize_features = normalize_features

        assert N_samples_train + N_samples_val <= len(specific_classes)*6000, (
            f"Too many samples requested for training and validation combined {N_samples_train}+{N_samples_val}. "
            f"The Fashion MNIST dataset for classes {specific_classes} has only {len(specific_classes)*6000} samples."
        )

        # Load dataset without transformations first to fit PCA
        temp_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1)),  # Flatten for PCA
            ]
        )

        temp_dataset = datasets.FashionMNIST(
            data_path, train=True, download=True, transform=temp_transform
        )

        # Fit PCA on a subset of training data
        self._fit_pca(temp_dataset)

        # Now initialize transform pipeline including PCA
        self._setup_transform()

        # Convert specific_classes to a tensor for efficient filtering
        specific_classes_tensor = torch.tensor(list(specific_classes))

        if not self.test_ds_only:
            fashion_train = datasets.FashionMNIST(
                data_path, train=True, download=True, transform=self.transform
            )

            # Get indices of specific classes
            train_indices = torch.where(
                torch.isin(fashion_train.targets, specific_classes_tensor)
            )[0].tolist()

            # Shuffle indices for randomization
            random_class.shuffle(train_indices)

            # Create subsets using filtered indices
            fashion_train_subset = Subset(
                fashion_train, train_indices[:N_samples_train]
            )
            fashion_val_subset = Subset(
                fashion_train,
                train_indices[N_samples_train : N_samples_train + N_samples_val],
            )

            # Create DataLoaders
            self.train_loader = DataLoader(
                fashion_train_subset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            )

            self.val_loader = DataLoader(
                fashion_val_subset, batch_size=batch_size, shuffle=True, drop_last=True
            )

        fashion_test = datasets.FashionMNIST(
            data_path, train=False, download=True, transform=self.transform
        )

        # Get indices of specific classes
        test_indices = torch.where(
            torch.isin(fashion_test.targets, specific_classes_tensor)
        )[0].tolist()

        # Shuffle for randomization
        random_class.shuffle(test_indices)

        # Create subset
        fashion_test_subset = Subset(fashion_test, test_indices[:N_samples_test])

        # Create DataLoader
        self.test_loader = DataLoader(
            fashion_test_subset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        self.classes = list(specific_classes)

        if self.normalize_features:
            print(f"Features normalized to range [0, 1].")

        if not self.test_ds_only:
            print(
                f"Train set size: {len(self.train_loader)} batches, each of {batch_size} samples -> {len(self.train_loader) * batch_size} samples"
            )
            print(
                f"Val set size: {len(self.val_loader)} batches, each of {batch_size} samples -> {len(self.val_loader) * batch_size} samples"
            )
            self.calculate_train_class_weights()
            self.calculate_val_class_counts()

        print(
            f"Test set size: {len(self.test_loader)} batches, each of {batch_size} samples -> {len(self.test_loader) * batch_size} samples"
        )
        self.calculate_test_class_counts()

        print("Used classes: ", self.classes)
        print("OK.")

    def _fit_pca(self, dataset):
        """Fit PCA on a sample of the dataset and calculate normalization bounds."""
        # Sample a subset to fit PCA (for memory efficiency)
        # n_samples = 60000#min(20000, len(dataset))
        # indices = random.sample(range(len(dataset)), n_samples)
        # Extract images and stack them
        X = torch.stack([dataset[i][0] for i in range(len(dataset))]).numpy()

        # Fit PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)

        # Calculate normalization bounds
        if self.normalize_features:
            # Transform a sample to find min/max values
            X_transformed = self.pca.transform(X)
            self.feature_min = np.min(X_transformed, axis=0)
            self.feature_max = np.max(X_transformed, axis=0)
            # Prevent division by zero in case a feature has constant value
            self.feature_range = np.maximum(self.feature_max - self.feature_min, 1e-10)

        LOGGER.info(
            f"ReducedFMNIST using PCA: {self.n_components} components explain {sum(self.pca.explained_variance_ratio_) * 100:.2f}% of variance."
        )

    def _apply_pca(self, img_tensor):
        """Apply PCA transformation to a tensor and normalize if requested."""
        # Ensure tensor is flattened
        img_flat = img_tensor.view(-1).numpy()

        # Apply PCA transformation
        img_reduced = self.pca.transform(img_flat.reshape(1, -1))[0]

        # Normalize to [0, 1] if requested
        if self.normalize_features:
            img_reduced = (img_reduced - self.feature_min) / self.feature_range

        # Convert back to tensor
        return torch.from_numpy(img_reduced).float()

    def _setup_transform(self):
        """Set up the transformation pipeline."""
        if self.normalize_features:
            transforms_list = [
                transforms.ToTensor(),
                # Note: No longer using standard normalization as we're normalizing after PCA
                transforms.Lambda(self._apply_pca),
            ]
        else:
            transforms_list = [
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),  # Fashion MNIST stats
                transforms.Lambda(self._apply_pca),
            ]

        self.transform = transforms.Compose(transforms_list)

    def calculate_train_class_weights(self):
        """
        Calculate weights for each class based on their frequency in the training dataset.
        Returns weights that are inversely proportional to class frequencies.
        """
        class_counts = torch.zeros(len(self.classes))

        for _, y in self.train_loader:
            for label in RemapTargets(y, self.classes):
                class_counts[label] += 1

        self.train_class_counts = class_counts.cpu().numpy()

        # Calculate weights (inverse of frequency)
        total_samples = class_counts.sum()
        class_weights = total_samples / (len(self.classes) * class_counts)

        # Normalize weights
        self.train_class_weights = (
            class_weights / class_weights.sum() * len(self.classes)
        )

    def calculate_val_class_counts(self):
        """Calculate class distribution in the validation dataset."""
        class_counts = torch.zeros(len(self.classes))

        for _, y in self.val_loader:
            for label in RemapTargets(y, self.classes):
                class_counts[label] += 1

        self.val_class_counts = class_counts.cpu().numpy()

    def calculate_test_class_counts(self):
        """Calculate class distribution in the test dataset."""
        class_counts = torch.zeros(len(self.classes))

        for _, y in self.test_loader:
            for label in RemapTargets(y, self.classes):
                class_counts[label] += 1

        self.test_class_counts = class_counts.cpu().numpy()

    def visualize(self, ax, x=np.arange(1, 150, 1)):
        covered_variances = []
        for i in x:
            covered_variances.append(self.pca.explained_variance_ratio_[:i].sum())
        ax.plot(
            x,
            covered_variances,
            color="xkcd:dark blue",
            # marker='o'
        )
        evr = self.pca.explained_variance_ratio_[: self.n_components].sum()
        ax.scatter(
            self.n_components,
            evr,
            label=f"Expl. ratio = {evr * 100:.2f}%",
            color="xkcd:purple",
            s=50,
            marker="x",
        )
        ax.set_xlabel("Number of F-MNIST\nprincipal components")
        ax.set_ylabel("Explained Variance Ratio")
        ax.legend(frameon=False)
        ax.set_ylim((0.2, 1.0))
        ax.grid(linestyle="--", alpha=0.5)


### Utility functions
def RemapTargets(targets, classes, src_class_count=10, device="cpu"):
    """If using a subset of classes in the MNIST, remap the targets (like 0,1,5,9) to consecutive integers (0,1,2,3) for the loss function."""

    remap = torch.zeros(src_class_count, dtype=torch.long, device=device)
    for i_v, v in enumerate(classes):
        remap[v] = i_v
    return remap[targets]
