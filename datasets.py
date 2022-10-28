import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


# transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# train and test data
download = False  # set to true if running for the first time
train_data = datasets.CIFAR10("data", train=True,
                             download=download, transform=transform)

test_data = datasets.CIFAR10("data", train=False,
                             download=download, transform=transform)

# get validation set
batch_size = 32
val_size = 0.2

num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)

split = int(np.floor(val_size * num_train))
train_idx, val_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

# create dataloaders
train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(train_data, batch_size=batch_size, sampler=val_sampler)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

data_loaders = {
    "train": train_loader, 
    "val": val_loader,
    "test": test_loader
}

data_sizes = {
    "train": len(train_loader.dataset) - split, 
    "val": split,
    "test": len(test_loader.dataset)
}

if __name__=="__main__":
    print(data_sizes)

    data_iter = iter(train_loader)
    # data_iter = iter(test_loader)
    images, labels = data_iter.next()
    images = images.numpy()
    print(images[0].shape)
    print(labels)
