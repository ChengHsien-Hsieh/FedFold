import numpy as np
import torch
import os
from collections import defaultdict
from torchvision import transforms
from torchvision.datasets import CIFAR10, SVHN, CIFAR100, ImageFolder
from torch.utils.data import Dataset
import pandas as pd
from model import Conv, MLP, ResNet


def set_parameters(cfg):
    if cfg['dataset'] == 'CIFAR10':
        # model_fn = Conv
        model_fn = ResNet # for CIFAR10 with ResNet
        cfg['n_class'] = 10
        if len(cfg['train_ratio'].split('-')) == 3 or model_fn == ResNet:
            cfg['global_epochs'] = 300
        else:
            cfg['global_epochs'] = 300
        cfg['local_epochs'] = 5
        hidden_size = {
            '16': [64, 128, 256, 512],
            '8': [32, 64, 128, 256],
            '4': [16, 32, 64, 128],
            '2': [8, 16, 32, 64],
            '1': [4, 8, 16, 32]
        }
    elif cfg['dataset'] == 'CIFAR100':
        # model_fn = Conv 
        model_fn = ResNet
        cfg['n_class'] = 100
        cfg['global_epochs'] = 300
        cfg['local_epochs'] = 5
        hidden_size = {
            '16': [64, 128, 256, 512],
            '8': [32, 64, 128, 256],
            '4': [16, 32, 64, 128],
            '2': [8, 16, 32, 64],
            '1': [4, 8, 16, 32]
        }
    elif cfg['dataset'] == 'TinyImageNet':
        # model_fn = Conv 
        model_fn = ResNet
        cfg['n_class'] = 200
        cfg['global_epochs'] = 300
        cfg['local_epochs'] = 5
        hidden_size = {
            '16': [64, 128, 256, 512],
            '8': [32, 64, 128, 256],
            '4': [16, 32, 64, 128],
            '2': [8, 16, 32, 64],
            '1': [4, 8, 16, 32]
        }
    elif cfg['dataset'] == 'Otto':
        model_fn = MLP
        cfg['n_class'] = 9
        cfg['global_epochs'] = 100
        cfg['local_epochs'] = 3
        hidden_size = {
            '16': [128, 64],
            '8': [64, 32],
            '4': [32, 16],
            '2': [16, 8],
            '1': [8, 4]
        }
    elif cfg['dataset'] == 'SVHN':
        model_fn = ResNet
        cfg['n_class'] = 10
        cfg['global_epochs'] = 500
        cfg['local_epochs'] = 5
        hidden_size = {
            '16': [64, 128, 256, 512],
            '8': [32, 64, 128, 256],
            '4': [16, 32, 64, 128],
            '2': [8, 16, 32, 64],
            '1': [4, 8, 16, 32]
        }
    return model_fn, hidden_size

class SplitDataset(Dataset):
    def __init__(self, dataset, data_idx):
        super().__init__()
        self.dataset = dataset
        self.data_idx = data_idx

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        img, label = self.dataset[self.data_idx[idx]]
        return img, label
    
def get_dataset(dataset_name, val_ratio=0.2):
    print(f'fetching dataset: {dataset_name}')
    os.makedirs('/media/massstorage2/b09901055/FL/dataset/', exist_ok=True)
    root = f'/media/massstorage2/b09901055/FL/dataset/{dataset_name}'
    dataset = {}
    labels = {}
    if dataset_name == 'CIFAR10':
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        # train_data = datasets.load_dataset(path='cifar10', split='train')
        # test_data = datasets.load_dataset(path='cifar10', split='test')        

        dataset['train'] = CIFAR10(root, train=True, download=True, transform=train_transforms)
        dataset['test'] = CIFAR10(root, train=False, download=True, transform=test_transforms)

        classes = [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9
                ]
        #personalize or generalize

        test_class = [(img, label) for img, label in dataset['test'] if label in classes] 
        dataset['test'] = torch.utils.data.Subset(test_class, range(len(test_class)))

        val_size = int(len(dataset['train'])*val_ratio)
        train_size = len(dataset['train']) - val_size
        dataset['train'], dataset['val'] = torch.utils.data.random_split(dataset['train'], [train_size, val_size])
        labels['train'] = [dataset['train'].dataset.targets[i] for i in dataset['train'].indices]
        labels['val'] = [dataset['val'].dataset.targets[i] for i in dataset['val'].indices]
        # labels['test'] = dataset['test'].targets
        labels['test'] = [dataset['test'].dataset[i][1] for i in range(len(dataset['test']))]
    elif dataset_name == 'CIFAR100':
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        dataset['train'] = CIFAR100(root, train=True, download=True, transform=train_transforms)
        dataset['test'] = CIFAR100(root, train=False, download=True, transform=test_transforms)
        val_size = int(len(dataset['train'])*val_ratio)
        train_size = len(dataset['train']) - val_size
        dataset['train'], dataset['val'] = torch.utils.data.random_split(dataset['train'], [train_size, val_size])
        labels['train'] = [dataset['train'].dataset.targets[i] for i in dataset['train'].indices]
        labels['val'] = [dataset['val'].dataset.targets[i] for i in dataset['val'].indices]
        labels['test'] = dataset['test'].targets
    elif dataset_name == 'TinyImageNet':
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        ])

        dataset['train'] = ImageFolder(root=os.path.join(root, 'tiny-imagenet-200', 'train'),
                                                transform=train_transforms)
        dataset['test'] = ImageFolder(root=os.path.join(root, 'tiny-imagenet-200', 'val'),
                                            transform=test_transforms)

        val_size = int(len(dataset['train']) * val_ratio)
        train_size = len(dataset['train']) - val_size
        dataset['train'], dataset['val'] = torch.utils.data.random_split(dataset['train'], [train_size, val_size])

        labels['train'] = [dataset['train'].dataset.targets[i] for i in dataset['train'].indices]
        labels['val'] = [dataset['val'].dataset.targets[i] for i in dataset['val'].indices]
        # labels['test'] = [label for _, label in dataset['test'].samples]
        labels['test'] = dataset['test'].targets
        # if hasattr(dataset['test'], 'targets') and len(dataset['test'].targets) > 0:
        #     print("Labels are present for the test dataset.")
        # else:
        #     print("No labels found for the test dataset.")
    elif dataset_name == 'Otto':
        # df = pd.read_csv(f'{root}/data.csv').sample(frac=1).reset_index(drop=True)
        # class_labels = df['target'].unique()

        # # split into train and test set
        # train_data = []
        # test_data = []
        # for label in class_labels:
        #     class_data = df[df['target'] == label]
        #     num = int(len(class_data)*0.8)
        #     train_data.append(class_data[:num])
        #     test_data.append(class_data[num:])

        # train_df = pd.concat(train_data)
        # test_df = pd.concat(test_data)
        # train_df.to_csv(f'{root}/train.csv', index=False)
        # test_df.to_csv(f'{root}/test.csv', index=False)

        train_df = pd.read_csv(f'{root}/train.csv').sample(frac=1).reset_index(drop=True)
        test_df = pd.read_csv(f'{root}/test.csv')
        class_labels = train_df['target'].unique()

        # split into train and validation set
        train_data = []
        valid_data = []
        for label in class_labels:
            class_data = train_df[train_df['target'] == label]
            num = int(len(class_data)*(1-val_ratio))
            train_data.append(class_data[:num])
            valid_data.append(class_data[num:])

        train_df = pd.concat(train_data)
        valid_df = pd.concat(valid_data)

        # build dataset
        dataset = {}
        labels = {}
        dataset['train'], labels['train'] = process_df(train_df)
        dataset['val'], labels['val'] = process_df(valid_df)
        dataset['test'], labels['test'] = process_df(test_df)
    elif dataset_name == 'SVHN':
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4380, 0.4440, 0.4730), (0.1751, 0.1771, 0.1744))
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4380, 0.4440, 0.4730), (0.1751, 0.1771, 0.1744))
        ])
        dataset['train'] = SVHN(root, split='train', download=True, transform=train_transforms)
        dataset['test'] = SVHN(root, split='test', download=True, transform=test_transforms)
        val_size = int(len(dataset['train'])*val_ratio)
        train_size = len(dataset['train']) - val_size
        dataset['train'], dataset['val'] = torch.utils.data.random_split(dataset['train'], [train_size, val_size])
        labels['train'] = [dataset['train'].dataset.labels[i] for i in dataset['train'].indices]
        labels['val'] = [dataset['val'].dataset.labels[i] for i in dataset['val'].indices]
        labels['test'] = dataset['test'].labels
    print(f"Train: {len(labels['train'])}, Val: {len(labels['val'])}, Test: {len(labels['test'])}")
    print('data ready')
    return dataset, labels

def process_df(df):
    data_y = []
    target_labels = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
    for i in df['target']:
        data_y.append(target_labels.index(i))
    data_x = torch.tensor(np.array(df)[:, 1:-1].astype(np.float32))
    return list(zip(data_x, data_y)), data_y

def split_dataset(labels, n_device, n_class, n_split, label_split=None):
    data_split = {i: [] for i in range(n_device)} # {client: [data idx]}

    label_data_idx = defaultdict(list) # {class: [data idx]}
    for i in range(len(labels)):
        label_data_idx[labels[i]].append(i)

    # number of devices can be assigned by each class
    device_per_label = n_split*n_device // n_class # device per label: 20
    device_per_label_list = [device_per_label for _ in range(n_class)]
    remain = np.random.choice(n_class, n_split*n_device % n_class, replace=False)
    # print(remain)
    for i in remain:
        device_per_label_list[i] += 1

    # split label_data_idx to number of device_per_label
    for label, data_idx in label_data_idx.items():
        # label_data_idx[label] = np.array(data_idx).reshape((device_per_label, -1)).tolist()
        num_leftover = len(data_idx) % device_per_label_list[label]
        leftover = data_idx[-num_leftover:] if num_leftover > 0 else []
        tmp = np.array(data_idx[:-num_leftover]) if num_leftover > 0 else np.array(data_idx)
        tmp = tmp.reshape((device_per_label_list[label], -1)).tolist()
        for i, leftover_data_idx in enumerate(leftover):
            tmp[i] = np.concatenate([tmp[i], [leftover_data_idx]])
        label_data_idx[label] = tmp
    
    # split label to number of n_split
    if label_split == None:
        label_split = []
        for _ in range(device_per_label):
            tmp = list(range(n_class)) # [0,1, ... , 9]
            tmp = torch.tensor(tmp)[torch.randperm(len(tmp))].tolist()
            label_split.append(tmp)
        label_split = np.array(label_split).reshape(-1).tolist()
        for i in remain:
            label_split.append(i)
        label_split = np.array(label_split).reshape((n_device, -1)).tolist()
        label_split = torch.tensor(label_split)[torch.randperm(len(label_split))].tolist()
        print(label_split)

    # split data idx to each client
    for i in range(n_device):
        for label in label_split[i]: # [[0, 1], [2, 3]...], len=100
            idx = torch.arange(len(label_data_idx[label]))[torch.randperm(len(label_data_idx[label]))[0]].item()
            data_split[i].extend(label_data_idx[label].pop(idx))
        # print(len(data_split[i]))
    return data_split, label_split