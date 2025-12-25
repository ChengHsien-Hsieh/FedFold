import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride):
        super(Block, self).__init__()
        # self.n1 = nn.BatchNorm2d(in_planes, momentum=None)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.n2 = nn.BatchNorm2d(planes, momentum=None)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.scaler = nn.Identity()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(x)
        # out = F.relu(self.n1(self.scaler(x)))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(out))
        out += shortcut
        return out
    
class ResNet(nn.Module):
    def __init__(self, hidden_size, input_channel=3, n_class=10, block=Block, num_blocks=[2, 2, 2, 2]):
        super(ResNet, self).__init__()
        self.in_planes = hidden_size[0]
        self.conv1 = nn.Conv2d(input_channel, hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], stride=2)
        # self.n4 = nn.BatchNorm2d(hidden_size[3] * block.expansion, momentum=None)
        self.scaler = nn.Identity()
        self.linear = nn.Linear(hidden_size[3] * block.expansion, n_class)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input):
        out = self.conv1(input)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(out)
        # out = F.relu(self.n4(self.scaler(out)))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class Conv(nn.Module):
    def __init__(self, hidden_size, input_channel=3, n_class=10):
        super().__init__()
        blocks = [nn.Conv2d(input_channel, hidden_size[0], 3, 1, 1),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(2)]
        for i in range(len(hidden_size) - 1):
            blocks.extend([nn.Conv2d(hidden_size[i], 
                           hidden_size[i + 1], 3, 1, 1),
                           nn.ReLU(inplace=True),
                           nn.MaxPool2d(2)])
        blocks = blocks[:-1]
        blocks.extend([nn.AdaptiveAvgPool2d(1),
                       nn.Flatten(),
                       nn.Linear(hidden_size[-1], n_class)])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        output = self.blocks(input)
        return output

class MLP(nn.Module):
    def __init__(self, hidden_size, input_channel=93, n_class=9):
        super().__init__()
        blocks = [nn.Linear(input_channel, hidden_size[0]),
                  nn.ReLU(inplace=True)]
        for i in range(len(hidden_size) - 1):
            blocks.extend([nn.Linear(hidden_size[i], hidden_size[i + 1]),
                           nn.ReLU(inplace=True)])
        blocks = blocks[:-1]
        blocks.extend([nn.Linear(hidden_size[-1], n_class)])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        output = self.blocks(input)
        return output

class LocalMaskCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, n_class):
        super(LocalMaskCrossEntropyLoss, self).__init__()
        self.n_class = n_class
        
    def forward(self, input, target):
        labels = torch.unique(target)
        mask = torch.zeros_like(input)
        for c in range(self.n_class):
            if c in labels:
                mask[:, c] = 1
        return F.cross_entropy(input*mask, target, reduction='mean')


# ============================================================
# TCN (Temporal Convolutional Network) for NLP Tasks
# Reference: Bai et al., "An Empirical Evaluation of Generic 
#            Convolutional and Recurrent Networks for Sequence Modeling"
# ============================================================

class Chomp1d(nn.Module):
    """
    移除因果卷積（Causal Convolution）產生的額外 padding，
    確保輸出序列長度等於輸入序列長度。
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    TCN 的基本構建塊（Residual Block），結構類似 ResNet 的 Block。
    
    結構：
    x -> Conv1d -> Chomp1d -> ReLU -> Dropout ->
         Conv1d -> Chomp1d -> ReLU -> Dropout -> (+) -> out
                                                  |
    x ----------------- (1x1 Conv if needed) -----+
    
    這與 ResNet 的 Block 結構完全同構，因此 FedFold 的寬度分割邏輯可直接套用。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # 第一個卷積層：Conv1d -> Chomp1d -> ReLU -> Dropout
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二個卷積層：Conv1d -> Chomp1d -> ReLU -> Dropout
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 網路主分支（與 ResNet 的 conv1->conv2 類似）
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        # Shortcut 連接（與 ResNet 相同邏輯）
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for NLP Text Classification
    
    與 ResNet 的設計理念完全對應：
    - ResNet: Conv2d -> [layer1, layer2, layer3, layer4] -> AdaptiveAvgPool -> FC
    - TCN:    Embedding -> [block1, block2, block3, block4] -> AdaptiveAvgPool -> FC
    
    這使得 FedFold 的寬度分割（width splitting）可以直接套用：
    - hidden_size['16'] = [64, 128, 256, 512]   # 強設備
    - hidden_size['1']  = [4, 8, 16, 32]        # 弱設備
    
    Arguments:
        hidden_size: list[int], 各層的通道數，例如 [64, 128, 256, 512]
        vocab_size: int, 詞彙表大小
        embed_dim: int, 詞嵌入維度
        n_class: int, 分類類別數
        kernel_size: int, 卷積核大小
        dropout: float, dropout 比例
    """
    def __init__(self, hidden_size, vocab_size=30000, embed_dim=128, n_class=4, 
                 kernel_size=3, dropout=0.2, max_seq_len=256):
        super(TCN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Word Embedding Layer（不參與寬度分割）
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 建立 TCN 層（與 ResNet 的 layer1-4 對應）
        # 從 embed_dim 逐漸擴展到 hidden_size[-1]
        layers = []
        num_levels = len(hidden_size)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # 擴大感受野
            in_channels = embed_dim if i == 0 else hidden_size[i - 1]
            out_channels = hidden_size[i]
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size, padding=padding,
                dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
        
        # 全局池化 + 分類器（與 ResNet 相同）
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size[-1], n_class)

    def forward(self, x):
        """
        Input:
            x: (batch_size, seq_len) - token indices
        Output:
            (batch_size, n_class) - logits
        """
        # Embedding: (batch, seq_len) -> (batch, seq_len, embed_dim)
        x = self.embedding(x)
        
        # 轉換為 Conv1d 格式: (batch, embed_dim, seq_len)
        x = x.transpose(1, 2)
        
        # TCN forward: (batch, embed_dim, seq_len) -> (batch, hidden_size[-1], seq_len)
        x = self.network(x)
        
        # Global pooling: (batch, hidden_size[-1], seq_len) -> (batch, hidden_size[-1], 1)
        x = self.global_pool(x)
        
        # Flatten: (batch, hidden_size[-1], 1) -> (batch, hidden_size[-1])
        x = x.squeeze(-1)
        
        # Classifier
        x = self.fc(x)
        
        return x


class TCNForHAR(nn.Module):
    """
    TCN for Human Activity Recognition (HAR) - 用於時間序列分類
    
    這個版本適用於類似 HAR 的時間序列資料，輸入是連續的感測器數據而非文字 token。
    
    Arguments:
        hidden_size: list[int], 各層的通道數
        input_channels: int, 輸入特徵維度（例如 HAR 的 9 軸感測器）
        n_class: int, 分類類別數
        kernel_size: int, 卷積核大小
        dropout: float, dropout 比例
    """
    def __init__(self, hidden_size, input_channels=9, n_class=6, 
                 kernel_size=3, dropout=0.2):
        super(TCNForHAR, self).__init__()
        
        layers = []
        num_levels = len(hidden_size)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else hidden_size[i - 1]
            out_channels = hidden_size[i]
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size, padding=padding,
                dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size[-1], n_class)

    def forward(self, x):
        """
        Input:
            x: (batch_size, input_channels, seq_len) - 時間序列資料
               或 (batch_size, seq_len, input_channels) - 會自動轉換
        Output:
            (batch_size, n_class) - logits
        """
        # 如果輸入格式是 (batch, seq_len, channels)，轉換為 (batch, channels, seq_len)
        if x.dim() == 3 and x.size(1) > x.size(2):
            x = x.transpose(1, 2)
        
        x = self.network(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        
        return x