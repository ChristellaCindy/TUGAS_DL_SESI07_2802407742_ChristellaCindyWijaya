import random
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import train_test_split

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

transform = transforms.ToTensor()
raw_train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
raw_test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
classes = raw_train.classes

full_data = ConcatDataset([raw_train, raw_test])
all_labels = torch.cat([raw_train.targets, raw_test.targets]).numpy()
idx = np.arange(len(full_data))

# Split 80% Train, 10% Val, 10% Test
train_idx, temp_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=all_labels)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed, stratify=all_labels[temp_idx])

train_data = Subset(full_data, train_idx)
val_data = Subset(full_data, val_idx)
test_data = Subset(full_data, test_idx)

batch_size = 128
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
