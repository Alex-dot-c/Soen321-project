# step2_test.py
"""
Load a trained model and evaluate it on the test set.
Produces confusion matrix and classification report.
"""
import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


def get_test_loader(root_test="./test_images", batch=64):
    """Load test data with CIFAR-10 normalization."""
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    tf_test = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    ds_te = ImageFolder(root_test, transform=tf_test)
    classes = ds_te.classes
    ld_te = DataLoader(ds_te, batch_size=batch, shuffle=False, num_workers=0)
    return ld_te, classes


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

    def forward(self, x):
        return self.classifier(self.features(x))


def evaluate_model(model_path, k=3, depth="original", batch=64):
    """Load model and evaluate on test set."""
    # Setup device
    device = ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create output directory
    out_dir = Path("./outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    print("\nLoading test data...")
    test_loader, classes = get_test_loader(batch=batch)
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Classes: {classes}")

    # Initialize model architecture
    if depth == "shallow":
        model = VGG_Shallow(num_classes=len(classes), k=k).to(device)
    elif depth == "deep":
        model = VGG_Deep(num_classes=len(classes), k=k).to(device)
    else:
        model = VGG11(num_classes=len(classes), k=k).to(device)

    # Load trained weights
    print(f"\nLoading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Architecture: {model.depth_name}")
    print(f"Kernel size: {k}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    preds, gts = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds.append(model(x).argmax(1).cpu().numpy())
            gts.append(y.cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(gts)

    # Calculate accuracy
    test_acc = accuracy_score(y_true, y_pred)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # Generate classification report
    print("\nClassification Report:")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    # Save classification report
    report_path = out_dir / f"classification_report_{depth}_k{k}.txt"
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write("=" * 60 + "\n")
        f.write(classification_report(y_true, y_pred, target_names=classes, digits=4))
    print(f"\nClassification report saved to: {report_path}")

    # Generate and save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(title=f"Confusion Matrix - {depth.capitalize()} VGG (k={k})",
           xlabel="Predicted Label",
           ylabel="True Label")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)

    # Add text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, cm[i, j],
                          ha="center", va="center",
                          color="white" if cm[i, j] > cm.max() / 2 else "black")

    cm_path = out_dir / f"confusion_matrix_{depth}_k{k}.png"
    fig.tight_layout()
    fig.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate trained VGG model on test set")
    ap.add_argument("--model", type=str, required=True,
                    help="Path to trained model (.pt file)")
    ap.add_argument("--kernel", type=int, default=3,
                    help="Convolution kernel size (must match training)")
    ap.add_argument("--depth", type=str, default="original",
                    choices=["shallow", "original", "deep"],
                    help="Model architecture depth (must match training)")
    ap.add_argument("--batch", type=int, default=64,
                    help="Batch size for evaluation")
    args = ap.parse_args()

    print("=" * 60)
    print(f"STEP 2: TESTING VGG CNN")
    print(f"Model: {args.model}")
    print(f"Depth: {args.depth} | Kernel: {args.kernel}")
    print("=" * 60)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\nError: Model file not found at {model_path}")
        print("Please run step1_train.py first to train a model.")
        exit(1)

    evaluate_model(model_path, k=args.kernel, depth=args.depth, batch=args.batch)
