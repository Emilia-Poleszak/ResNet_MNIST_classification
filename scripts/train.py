from scripts.load_config import load_config
from models.resnet import build_model
import torch.optim as optim
from torchvision import datasets, transforms

def train():
    cfg = load_config()
    model = build_model()

    transform = transforms.Compose([
        transforms.Resize((cfg["data"]["img_size"], cfg["data"]["img_size"])),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # import MNIST dataset
    data = datasets.MNIST(cfg["data"]["root"], train=True, download=True, transform=transform)

    optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    # main training loop
    for epoch in range(cfg["train"]["epochs"]):
        model.train()