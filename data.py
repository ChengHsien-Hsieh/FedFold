import numpy as np
import torch
import os
from collections import defaultdict
from torchvision import transforms
from torchvision.datasets import CIFAR10, SVHN, CIFAR100, ImageFolder
from torch.utils.data import Dataset
import pandas as pd
from model import Conv, MLP, ResNet, TCN


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
    # ============================================================
    # NLP Datasets with TCN
    # ============================================================
    elif cfg['dataset'] == 'AGNews':
        model_fn = TCN
        cfg['n_class'] = 4  # AGNews 有 4 個類別：World, Sports, Business, Sci/Tech
        cfg['global_epochs'] = 100
        cfg['local_epochs'] = 3
        cfg['vocab_size'] = 30000
        cfg['embed_dim'] = 128
        cfg['max_seq_len'] = 256
        hidden_size = {
            '16': [64, 128, 256, 512],
            '8': [32, 64, 128, 256],
            '4': [16, 32, 64, 128],
            '2': [8, 16, 32, 64],
            '1': [4, 8, 16, 32]
        }
    elif cfg['dataset'] == 'SST2':
        model_fn = TCN
        cfg['n_class'] = 2  # SST-2: Positive / Negative
        cfg['global_epochs'] = 100
        cfg['local_epochs'] = 3
        cfg['vocab_size'] = 20000
        cfg['embed_dim'] = 128
        cfg['max_seq_len'] = 128
        hidden_size = {
            '16': [64, 128, 256, 512],
            '8': [32, 64, 128, 256],
            '4': [16, 32, 64, 128],
            '2': [8, 16, 32, 64],
            '1': [4, 8, 16, 32]
        }
    elif cfg['dataset'] == 'IMDB':
        model_fn = TCN
        cfg['n_class'] = 2  # IMDB: Positive / Negative
        cfg['global_epochs'] = 50
        cfg['local_epochs'] = 3
        cfg['vocab_size'] = 30000
        cfg['embed_dim'] = 128
        cfg['max_seq_len'] = 512  # IMDB 評論較長
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
    # ============================================================
    # NLP Datasets
    # ============================================================
    elif dataset_name == 'AGNews':
        dataset, labels = load_agnews_dataset(root, val_ratio)
    elif dataset_name == 'SST2':
        dataset, labels = load_sst2_dataset(root, val_ratio)
    elif dataset_name == 'IMDB':
        dataset, labels = load_imdb_dataset(root, val_ratio)
    print(f"Train: {len(labels['train'])}, Val: {len(labels['val'])}, Test: {len(labels['test'])}")
    print('data ready')
    return dataset, labels


# ============================================================
# NLP Dataset Classes and Loaders
# ============================================================

class TextDataset(Dataset):
    """
    通用文本資料集類別，用於 NLP 任務
    存儲 tokenized 的文本序列和標籤
    """
    def __init__(self, texts, labels, vocab, max_seq_len=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize and convert to indices
        tokens = simple_tokenize(text)
        indices = [self.vocab.get(token, self.vocab.get('<unk>', 1)) for token in tokens]
        
        # Padding or truncation
        if len(indices) < self.max_seq_len:
            indices = indices + [0] * (self.max_seq_len - len(indices))  # 0 = <pad>
        else:
            indices = indices[:self.max_seq_len]
        
        return torch.tensor(indices, dtype=torch.long), label


def simple_tokenize(text):
    """
    簡單的分詞器：小寫化、移除標點、按空格分詞
    """
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    return tokens


def build_vocab(texts, max_vocab_size=30000):
    """
    從文本建立詞彙表
    """
    from collections import Counter
    
    word_counts = Counter()
    for text in texts:
        tokens = simple_tokenize(text)
        word_counts.update(tokens)
    
    # 保留最常見的詞
    most_common = word_counts.most_common(max_vocab_size - 2)
    
    vocab = {'<pad>': 0, '<unk>': 1}
    for word, _ in most_common:
        vocab[word] = len(vocab)
    
    return vocab


def load_agnews_dataset(root, val_ratio=0.2, max_vocab_size=30000, max_seq_len=256):
    """
    載入 AG News 資料集
    使用 torchtext 或從 CSV 載入
    
    AG News 類別：
    0: World
    1: Sports
    2: Business
    3: Sci/Tech
    """
    try:
        from datasets import load_dataset
        
        print("Loading AG News from HuggingFace datasets...")
        raw_dataset = load_dataset('ag_news')
        
        train_texts = raw_dataset['train']['text']
        train_labels = raw_dataset['train']['label']
        test_texts = raw_dataset['test']['text']
        test_labels = raw_dataset['test']['label']
        
    except ImportError:
        # 備用方案：從 CSV 載入
        print("HuggingFace datasets not found, trying to load from CSV...")
        csv_path = os.path.join(root, 'ag_news_train.csv')
        if os.path.exists(csv_path):
            import csv
            train_texts, train_labels = [], []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    train_labels.append(int(row[0]) - 1)  # AG News CSV 標籤從 1 開始
                    train_texts.append(row[1] + ' ' + row[2])
            
            test_csv_path = os.path.join(root, 'ag_news_test.csv')
            test_texts, test_labels = [], []
            with open(test_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    test_labels.append(int(row[0]) - 1)
                    test_texts.append(row[1] + ' ' + row[2])
        else:
            raise FileNotFoundError(
                f"AG News dataset not found. Please install 'datasets' library:\n"
                f"  pip install datasets\n"
                f"Or download CSV from: https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset"
            )
    
    # 建立詞彙表
    print("Building vocabulary...")
    vocab = build_vocab(train_texts, max_vocab_size)
    print(f"Vocabulary size: {len(vocab)}")
    
    # 分割訓練集和驗證集
    val_size = int(len(train_texts) * val_ratio)
    train_size = len(train_texts) - val_size
    
    indices = torch.randperm(len(train_texts)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_texts_split = [train_texts[i] for i in train_indices]
    train_labels_split = [train_labels[i] for i in train_indices]
    val_texts = [train_texts[i] for i in val_indices]
    val_labels = [train_labels[i] for i in val_indices]
    
    # 建立資料集
    dataset = {}
    labels = {}
    
    dataset['train'] = TextDataset(train_texts_split, train_labels_split, vocab, max_seq_len)
    dataset['val'] = TextDataset(val_texts, val_labels, vocab, max_seq_len)
    dataset['test'] = TextDataset(test_texts, test_labels, vocab, max_seq_len)
    
    labels['train'] = train_labels_split
    labels['val'] = val_labels
    labels['test'] = list(test_labels)
    
    # 保存詞彙表供後續使用
    dataset['vocab'] = vocab
    
    return dataset, labels


def load_sst2_dataset(root, val_ratio=0.2, max_vocab_size=20000, max_seq_len=128):
    """
    載入 SST-2 (Stanford Sentiment Treebank) 資料集
    二元情感分類：Positive / Negative
    """
    try:
        from datasets import load_dataset
        
        print("Loading SST-2 from HuggingFace datasets...")
        raw_dataset = load_dataset('glue', 'sst2')
        
        train_texts = raw_dataset['train']['sentence']
        train_labels = raw_dataset['train']['label']
        # SST-2 的 test set 沒有標籤，使用 validation 作為 test
        test_texts = raw_dataset['validation']['sentence']
        test_labels = raw_dataset['validation']['label']
        
    except ImportError:
        raise ImportError(
            "Please install 'datasets' library:\n  pip install datasets"
        )
    
    # 建立詞彙表
    print("Building vocabulary...")
    vocab = build_vocab(train_texts, max_vocab_size)
    print(f"Vocabulary size: {len(vocab)}")
    
    # 分割訓練集和驗證集
    val_size = int(len(train_texts) * val_ratio)
    train_size = len(train_texts) - val_size
    
    indices = torch.randperm(len(train_texts)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_texts_split = [train_texts[i] for i in train_indices]
    train_labels_split = [train_labels[i] for i in train_indices]
    val_texts = [train_texts[i] for i in val_indices]
    val_labels = [train_labels[i] for i in val_indices]
    
    dataset = {}
    labels = {}
    
    dataset['train'] = TextDataset(train_texts_split, train_labels_split, vocab, max_seq_len)
    dataset['val'] = TextDataset(val_texts, val_labels, vocab, max_seq_len)
    dataset['test'] = TextDataset(test_texts, test_labels, vocab, max_seq_len)
    
    labels['train'] = train_labels_split
    labels['val'] = val_labels
    labels['test'] = list(test_labels)
    dataset['vocab'] = vocab
    
    return dataset, labels


def load_imdb_dataset(root, val_ratio=0.2, max_vocab_size=30000, max_seq_len=512):
    """
    載入 IMDB 電影評論資料集
    二元情感分類：Positive / Negative
    """
    try:
        from datasets import load_dataset
        
        print("Loading IMDB from HuggingFace datasets...")
        raw_dataset = load_dataset('imdb')
        
        train_texts = raw_dataset['train']['text']
        train_labels = raw_dataset['train']['label']
        test_texts = raw_dataset['test']['text']
        test_labels = raw_dataset['test']['label']
        
    except ImportError:
        raise ImportError(
            "Please install 'datasets' library:\n  pip install datasets"
        )
    
    # 建立詞彙表
    print("Building vocabulary...")
    vocab = build_vocab(train_texts, max_vocab_size)
    print(f"Vocabulary size: {len(vocab)}")
    
    # 分割訓練集和驗證集
    val_size = int(len(train_texts) * val_ratio)
    train_size = len(train_texts) - val_size
    
    indices = torch.randperm(len(train_texts)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_texts_split = [train_texts[i] for i in train_indices]
    train_labels_split = [train_labels[i] for i in train_indices]
    val_texts = [train_texts[i] for i in val_indices]
    val_labels = [train_labels[i] for i in val_indices]
    
    dataset = {}
    labels = {}
    
    dataset['train'] = TextDataset(train_texts_split, train_labels_split, vocab, max_seq_len)
    dataset['val'] = TextDataset(val_texts, val_labels, vocab, max_seq_len)
    dataset['test'] = TextDataset(test_texts, test_labels, vocab, max_seq_len)
    
    labels['train'] = train_labels_split
    labels['val'] = val_labels
    labels['test'] = list(test_labels)
    dataset['vocab'] = vocab
    
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