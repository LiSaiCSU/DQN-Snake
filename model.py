import torch
import torch.nn as nn
import torchvision.models as models


def create_modified_resnet18():
    # 加载一个预训练的ResNet-18模型（也可以不预训练）
    model = models.resnet18(
        weights=None
    )  # weights=None表示不使用预训练权重，随机初始化

    # 1. 修改输入层
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # 2. 替换输出层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)  # 3个动作

    return model


# 使用
q_network = create_modified_resnet18()
