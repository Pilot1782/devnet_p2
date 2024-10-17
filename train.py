import kagglehub
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

img_width, img_height = 250, 250

# Download latest version
path = kagglehub.dataset_download("csafrit2/plant-leaves-for-image-classification")

print("Path to dataset files:", path)

train_data_dir = path + r"\Plants_2\train"
validation_data_dir = path + r"\Plants_2\valid"
nb_train_samples = 4274
nb_validation_samples = 110
epochs = 10
batch_size = 16


class HealthModel(nn.Module):
    def __init__(self):
        super(HealthModel, self).__init__()

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
            nn.Linear(57600, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.seq(x)


model = HealthModel()

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


# Load the training dataset
train_dataset = ImageFolder(root=train_data_dir, transform=train_transforms)

# Load the testing dataset
test_dataset = ImageFolder(root=validation_data_dir, transform=test_transforms)


# Create the DataLoader for the training dataset
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create the DataLoader for the testing dataset
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

classes = ("healthy", "unhealthy")


def train(dataloader, _model, _loss_fn, _optimizer):
    size = len(dataloader.dataset)
    _model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = _model(X)
        loss = _loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        _optimizer.step()
        _optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
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
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
