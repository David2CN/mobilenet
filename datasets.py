import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from torch.utils.data import ConcatDataset


def augment_data(data):
    transformed_imgs = []
    labels = []
    data_transforms = [
        (transforms.RandomHorizontalFlip(p=1), 1),
        (transforms.RandomVerticalFlip(p=1), 1),
    ]

    print("Augmenting train data...")
    for i in tqdm(range(len(data))):
        image, label = data[i]
        for transform, count in data_transforms:
            for k in range(count):
                transformed_imgs.append(transform(image))
                labels.append(label)
    
    return transformed_imgs, labels


# augmented dataset
class AugDataset(Dataset):
    def __init__(self, images, labels):
        super().__init__()
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        return img, label


# get validation set and loaders
def get_dataloaders(batch_size: int=32, val_size: float=0.1, 
                    download: bool=False, data_dir: str="../data",
                    transform=None, augment: bool=False):
    
    if not transform:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # train and test data
    train_data = datasets.CIFAR10(data_dir, train=True,
                                download=download, transform=transform)

    test_data = datasets.CIFAR10(data_dir, train=False,
                                download=download, transform=transform)

    if augment:
        # augment train data
        aug_train = augment_data(train_data)
        aug_dataset = AugDataset(*aug_train)
        train_data = ConcatDataset((train_data, aug_dataset))

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

    return data_loaders, data_sizes

