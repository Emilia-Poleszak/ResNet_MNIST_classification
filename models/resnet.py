import torch
import torch.nn as nn
from torchvision import models
import yaml
from scripts.load_config import load_config

def build_model():
    cfg = load_config()
    model = models.resnet18(pretrained=cfg['model']['pretrained'])
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, cfg['model']['num_classes'])
    return model

if __name__ == "__main__":
    model = build_model()