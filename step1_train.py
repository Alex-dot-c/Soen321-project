# step1_train.py
"""
Train a VGG-style CNN on CIFAR-10 subset and save the best model.
"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder


def get_loaders(root_train="./training_images", batch=64):
    """Load training data with CIFAR-10 normalization."""
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    tf_train = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    ds_tr = ImageFolder(root_train, transform=tf_train)
    classes = ds_tr.classes
    ld_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, num_workers=0)
    return ld_tr, classes


class VGG11(nn.Module):
    """Original VGG11: 8 convolutional layers"""
    def __init__(self, num_classes=10, k=3, bn=True):
        super().__init__()
        p = k // 2

        def block(in_c, out_c, n):
            layers = []
            for i in range(n):
                layers += [nn.Conv2d(in_c if i == 0 else out_c, out_c, k,
                                     padding=p, bias=not bn)]
                if bn:
                    layers += [nn.BatchNorm2d(out_c)]
                layers += [nn.ReLU(inplace=True)]
            layers += [nn.MaxPool2d(2, 2)]
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(3,   64, 1),
            block(64, 128, 1),
            block(128, 256, 2),
            block(256, 512, 2),
            block(512, 512, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
        self.depth_name = "original"
        self.num_conv_layers = 8

    def forward(self, x):
        return self.classifier(self.features(x))


class VGG_Shallow(nn.Module):
    """Shallow CNN with 5 convolutional layers"""
    def __init__(self, num_classes=10, k=3, bn=True):
        super().__init__()
        p = k // 2

        def block(in_c, out_c, n):
            layers = []
            for i in range(n):
                layers += [nn.Conv2d(in_c if i == 0 else out_c, out_c, k,
                                     padding=p, bias=not bn)]
                if bn:
                    layers += [nn.BatchNorm2d(out_c)]
                layers += [nn.ReLU(inplace=True)]
            layers += [nn.MaxPool2d(2, 2)]
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(3,   64, 1),
            block(64, 128, 2),
            block(128, 256, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 2048), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
        )
        self.depth_name = "shallow"
        self.num_conv_layers = 5

    def forward(self, x):
        return self.classifier(self.features(x))


class VGG_Deep(nn.Module):
    """Deep CNN with 13 convolutional layers"""
    def __init__(self, num_classes=10, k=3, bn=True):
        super().__init__()
        p = k // 2

        def block(in_c, out_c, n):
            layers = []
            for i in range(n):
                layers += [nn.Conv2d(in_c if i == 0 else out_c, out_c, k,
                                     padding=p, bias=not bn)]
                if bn:
                    layers += [nn.BatchNorm2d(out_c)]
                layers += [nn.ReLU(inplace=True)]
            layers += [nn.MaxPool2d(2, 2)]
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(3,   64, 2),
            block(64, 128, 2),
            block(128, 256, 3),
            block(256, 512, 3),
            block(512, 512, 3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
        self.depth_name = "deep"
        self.num_conv_layers = 13

    def forward(self, x):
        return self.classifier(self.features(x))


def train_model(epochs=40, lr=0.01, k=3, depth="original", batch=64):
    """Train the model and save the best checkpoint."""
    # Setup device
    device = ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create output directory
    out_dir = Path("./outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading training data...")
    train_loader, classes = get_loaders(batch=batch)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Classes: {classes}")

    # Initialize model
    if depth == "shallow":
        model = VGG_Shallow(num_classes=len(classes), k=k).to(device)
    elif depth == "deep":
        model = VGG_Deep(num_classes=len(classes), k=k).to(device)
    else:
        model = VGG11(num_classes=len(classes), k=k).to(device)

    print(f"\nArchitecture: {model.depth_name}")
    print(f"Kernel size: {k}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup training
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(epochs // 3, 1), gamma=0.1)

    save_path = out_dir / f"vgg_{depth}_k{k}.pt"

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 60)

    best_train_acc = 0.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        total = correct = 0
        loss_sum = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()

        sched.step()

        train_loss = loss_sum / total
        train_acc = correct / total

        print(f"Epoch {ep:3d}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        # Save best model based on training accuracy
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_state = model.state_dict().copy()

    # Save the best model
    torch.save(best_state, save_path)
    print("=" * 60)
    print(f"\nTraining complete!")
    print(f"Best training accuracy: {best_train_acc:.4f}")
    print(f"Model saved to: {save_path}")

    # Save model metadata
    meta_path = out_dir / f"vgg_{depth}_k{k}_meta.txt"
    with open(meta_path, "w") as f:
        f.write(f"Architecture: {depth}\n")
        f.write(f"Kernel size: {k}\n")
        f.write(f"Conv layers: {model.num_conv_layers}\n")
        f.write(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"Epochs trained: {epochs}\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(f"Batch size: {batch}\n")
        f.write(f"Best training accuracy: {best_train_acc:.4f}\n")
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train VGG-style CNN on CIFAR-10")
    ap.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    ap.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    ap.add_argument("--kernel", type=int, default=3, help="Convolution kernel size")
    ap.add_argument("--depth", type=str, default="original",
                    choices=["shallow", "original", "deep"],
                    help="Model architecture depth")
    ap.add_argument("--batch", type=int, default=64, help="Batch size")
    args = ap.parse_args()

    print("=" * 60)
    print(f"STEP 1: TRAINING VGG CNN")
    print(f"Depth: {args.depth} | Kernel: {args.kernel}")
    print("=" * 60)

    train_model(epochs=args.epochs, lr=args.lr, k=args.kernel,
                depth=args.depth, batch=args.batch)
