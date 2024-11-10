import datetime
import os
import time
from pathlib import Path
from typing import Union, Tuple, List, Dict, Optional, Callable, Any

import kagglehub
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader, make_dataset, IMG_EXTENSIONS
from torchvision.transforms import transforms

img_width, img_height = 250, 250

# Download latest version
if __name__ == "__main__":
    path = kagglehub.dataset_download("csafrit2/plant-leaves-for-image-classification")
else:
    path = "..."

train_data_dir = path + r"\Plants_2\train"
validation_data_dir = path + r"\Plants_2\valid"
epochs = 250
batch_size = 16
classes = ("healthy", "unhealthy")


class HealthNetwork(nn.Module):
    def __init__(self):
        super(HealthNetwork, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 32, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Flatten(),
            nn.Linear(57600, 65),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(65, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.seq(x)


class FilteredDataFolder(VisionDataset):
    """A generic data loader.

    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.

    Args:
        root (str or ``pathlib.Path``): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
        allow_empty(bool, optional): If True, empty folders are considered to be valid classes.
            An error is raised on empty folders if False (default).

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: Union[str, Path],
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            allow_empty: bool = False,
            allowed_classes: Optional[tuple[str]] = None
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        _classes, class_to_idx = self.find_classes(self.root, allowed_classes)
        samples = self.make_dataset(
            self.root,
            class_to_idx=class_to_idx,
            extensions=extensions,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
        )

        self.loader = loader
        self.extensions = extensions

        self.classes = _classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
            directory: Union[str, Path],
            class_to_idx: Dict[str, int],
            extensions: Optional[Tuple[str, ...]] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            allow_empty: bool = False,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.
            allow_empty(bool, optional): If True, empty folders are considered to be valid classes.
                An error is raised on empty folders if False (default).

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(
            directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file, allow_empty=allow_empty
        )

    @staticmethod
    def find_classes(directory: Union[str, Path], allowed_classes: Optional[tuple[str]] = None) -> Tuple[
        List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``
            allowed_classes(tuple[str]): Whitelist of classes to use in the root directory path

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory, allowed_classes)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        _path, target = self.samples[index]
        sample = self.loader(_path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class FilteredImageFolder(FilteredDataFolder):
    def __init__(
            self,
            root: Union[str, Path],
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            allow_empty: bool = False,
            allowed_classes: Optional[tuple[str]] = None,
    ):
        super().__init__(
            root=root,
            transform=transform,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
            allowed_classes=allowed_classes
        )


def find_classes(
        directory: Union[str, Path],
        allowed_classes: Optional[tuple[str]] = None
) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    _classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    if allowed_classes is not None:
        _new_classes = []
        for _class in _classes:
            if _class in allowed_classes:
                _new_classes.append(_class)
        _classes = _new_classes

    if not _classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(_classes)}
    return _classes, class_to_idx


def train(dataloader, _model, _loss_fn, _optimizer):
    size = len(dataloader.dataset)
    _model.train()
    for batch, (_input, label) in enumerate(dataloader):
        _input, label = _input.to(device), label.to(device)

        # Compute prediction error
        pred = _model(_input)
        loss = _loss_fn(pred, label)

        # Backpropagation
        loss.backward()
        _optimizer.step()
        _optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(_input)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, _model, _loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    _model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = _model(X)
            test_loss += _loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return correct


model = HealthNetwork()

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

# Define the transformations for the training data
train_transforms = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Define the transformations for the testing data
test_transforms = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Define the transformation for the validation data
valid_transforms = transforms.Compose([
    transforms.Resize((250, 250)),
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

if __name__ == "__main__":
    # Load the training dataset
    train_dataset = FilteredImageFolder(root=train_data_dir, transform=train_transforms, allowed_classes=classes)
    print(f"Training classes: " + ", ".join(train_dataset.class_to_idx.keys()))

    # Load the testing dataset
    test_dataset = FilteredImageFolder(root=validation_data_dir, transform=test_transforms, allowed_classes=classes)
    print(f"Testing classes: " + ", ".join(test_dataset.class_to_idx.keys()))

    # Load the validating dataset
    valid_dataset = FilteredImageFolder(root=os.path.join(path, "Plants_2", "test"), transform=valid_transforms,
                                        allowed_classes=classes)
    print(f"Validation classes: " + ", ".join(valid_dataset.class_to_idx.keys()))

    # Create the DataLoader for the training dataset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create the DataLoader for the testing dataset
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Create the DataLoader for the validating dataset
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=True)

    model = model.to(device)
    test_acc = []
    val_acc = []
    tStart = 0
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        tStart = time.time()

        train(train_dataloader, model, loss_fn, optimizer)

        print("Test Error:", end="\n  ")
        test_acc.append(test(test_dataloader, model, loss_fn))
        print("Validation Error:", end="\n  ")
        val_acc.append(test(valid_dataloader, model, loss_fn))

        dTime = time.time() - tStart
        remaining = epochs - (t + 1)
        srem = remaining * dTime
        print(
            f"ETA: {datetime.datetime.fromtimestamp(datetime.datetime.now().timestamp() + srem)}"
            f"\n-------------------------------\n"
        )

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    plt.plot(test_acc, label="Test Data")
    plt.plot(val_acc, label="Validation Data")
    plt.legend()
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.show()

    print("Final accuracy over validation data\n-------------------------------")
    test(
        valid_dataloader,
        model,
        loss_fn
    )
