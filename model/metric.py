import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

device_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def Accuracy(x: torch.Tensor, y: torch.Tensor, model: nn.Module, device: str = device_) -> float:
    """
    Classification 모델 평가 함수

    :param x: Validation 입력 데이터 셋
    :param y: Validation 라벨 데이터
    :param model: Classification 모델
    :param device: GPU 설정셋
    
    :return: Accuracy
    """
    x = x.to(device)
    y = y.to(device)
    preds = torch.argmax(torch.softmax(model(x), dim=1), dim=1)
    num_correct = (preds == y).sum()

    accuracy = (num_correct / len(y)) * 100
    return accuracy.cpu().item()