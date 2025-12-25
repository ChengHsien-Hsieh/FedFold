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