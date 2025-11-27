# step3_attack.py
"""
Load a trained model and evaluate its robustness against FGSM adversarial attacks.
Produces adversarial accuracy curves and example visualizations.
"""
import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
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


def fgsm_attack(model, device, data_loader, epsilon: float, classes):
    """
    Perform FGSM attack and return adversarial accuracy.
    Also returns some example adversarial images for visualization.
    """
    crit = nn.CrossEntropyLoss()
    model.eval()

    correct = 0
    total = 0

    # Store examples for visualization (first batch only)
    examples_stored = False
    example_data = None

    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        x.requires_grad = True

        # Forward pass
        outputs = model(x)
        loss = crit(outputs, y)

        # Backward pass to get gradients
        model.zero_grad()
        loss.backward()

        # Generate adversarial examples using FGSM
        data_grad = x.grad.data
        x_adv = x + epsilon * data_grad.sign()
        x_adv = torch.clamp(x_adv, -3.0, 3.0)  # Clamp to normalized range

        # Evaluate on adversarial examples
        with torch.no_grad():
            outputs_adv = model(x_adv)
        pred_adv = outputs_adv.argmax(1)

        correct += (pred_adv == y).sum().item()
        total += y.size(0)

        # Store first batch for visualization
        if not examples_stored and batch_idx == 0:
            # Get original predictions
            with torch.no_grad():
                outputs_orig = model(x)
            pred_orig = outputs_orig.argmax(1)

            # Store up to 5 examples
            num_examples = min(5, x.size(0))
            example_data = {
                'original': x[:num_examples].cpu(),
                'adversarial': x_adv[:num_examples].cpu(),
                'perturbation': (x_adv - x)[:num_examples].cpu(),
                'true_labels': y[:num_examples].cpu(),
                'orig_preds': pred_orig[:num_examples].cpu(),
                'adv_preds': pred_adv[:num_examples].cpu(),
                'classes': classes,
                'epsilon': epsilon
            }
            examples_stored = True

    accuracy = correct / total
    return accuracy, example_data


def visualize_attack_examples(example_data, save_path):
    """Visualize original, perturbation, and adversarial images side by side."""
    # Denormalize for visualization
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

    original = example_data['original'] * std + mean
    adversarial = example_data['adversarial'] * std + mean
    perturbation = example_data['perturbation']

    # Clip to [0, 1]
    original = torch.clamp(original, 0, 1)
    adversarial = torch.clamp(adversarial, 0, 1)

    # Amplify perturbation for visibility
    perturbation_vis = perturbation * 10 + 0.5
    perturbation_vis = torch.clamp(perturbation_vis, 0, 1)

    num_examples = original.size(0)
    fig, axes = plt.subplots(num_examples, 3, figsize=(10, 3 * num_examples))

    if num_examples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_examples):
        # Original image
        axes[i, 0].imshow(original[i].permute(1, 2, 0).numpy())
        true_label = example_data['classes'][example_data['true_labels'][i]]
        orig_pred = example_data['classes'][example_data['orig_preds'][i]]
        axes[i, 0].set_title(f"Original\nTrue: {true_label}\nPred: {orig_pred}")
        axes[i, 0].axis('off')

        # Perturbation (amplified)
        axes[i, 1].imshow(perturbation_vis[i].permute(1, 2, 0).numpy())
        axes[i, 1].set_title(f"Perturbation (×10)\nε={example_data['epsilon']}")
        axes[i, 1].axis('off')

        # Adversarial image
        axes[i, 2].imshow(adversarial[i].permute(1, 2, 0).numpy())
        adv_pred = example_data['classes'][example_data['adv_preds'][i]]
        success = "✓" if adv_pred != true_label else "✗"
        axes[i, 2].set_title(f"Adversarial\nPred: {adv_pred} {success}")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def attack_model(model_path, k=3, depth="original", batch=64,
                epsilons=None):
    """Load model and perform FGSM attacks with different epsilon values."""
    if epsilons is None:
        epsilons = [0.0, 0.01, 0.03, 0.05, 0.1]

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

    # Perform FGSM attacks with different epsilon values
    print("\n" + "=" * 60)
    print("FGSM ADVERSARIAL ATTACK")
    print("=" * 60)
    print(f"Testing with epsilon values: {epsilons}")
    print()

    adv_accs = []
    all_examples = []

    for eps in epsilons:
        print(f"Running FGSM attack with ε = {eps:.3f}...", end=" ")
        adv_acc, examples = fgsm_attack(model, device, test_loader, eps, classes)
        adv_accs.append(adv_acc)
        all_examples.append(examples)
        print(f"Accuracy: {adv_acc:.4f}")

    # Save results to text file
    results_path = out_dir / f"attack_results_{depth}_k{k}.txt"
    with open(results_path, "w") as f:
        f.write("FGSM Adversarial Attack Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {depth} VGG (k={k})\n")
        f.write(f"Model path: {model_path}\n\n")
        f.write("Epsilon\tAccuracy\tAccuracy Drop\n")
        f.write("-" * 60 + "\n")
        for i, eps in enumerate(epsilons):
            drop = adv_accs[0] - adv_accs[i]
            f.write(f"{eps:.3f}\t{adv_accs[i]:.4f}\t\t{drop:.4f}\n")
    print(f"\nResults saved to: {results_path}")

    # Plot adversarial accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, adv_accs, marker="o", linewidth=2, markersize=8)
    plt.xlabel("Epsilon (ε)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(f"FGSM Adversarial Robustness - {depth.capitalize()} VGG (k={k})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])

    # Add accuracy values as labels
    for i, (eps, acc) in enumerate(zip(epsilons, adv_accs)):
        plt.annotate(f'{acc:.3f}',
                    xy=(eps, acc),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9)

    adv_plot_path = out_dir / f"fgsm_curve_{depth}_k{k}.png"
    plt.tight_layout()
    plt.savefig(adv_plot_path, dpi=200)
    plt.close()
    print(f"Adversarial curve saved to: {adv_plot_path}")

    # Visualize attack examples for a few epsilon values
    vis_epsilons = [0.03, 0.05, 0.1]  # Choose meaningful epsilons
    for i, eps in enumerate(epsilons):
        if eps in vis_epsilons and all_examples[i] is not None:
            vis_path = out_dir / f"attack_examples_{depth}_k{k}_eps{eps:.3f}.png"
            visualize_attack_examples(all_examples[i], vis_path)
            print(f"Attack examples (ε={eps:.3f}) saved to: {vis_path}")

    # Summary
    print("\n" + "=" * 60)
    print("ATTACK SUMMARY")
    print("=" * 60)
    print(f"Clean accuracy (ε=0.00):  {adv_accs[0]:.4f}")
    print(f"Worst accuracy (ε={max(epsilons):.2f}): {min(adv_accs):.4f}")
    print(f"Accuracy drop:             {adv_accs[0] - min(adv_accs):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Perform FGSM adversarial attacks on trained VGG model")
    ap.add_argument("--model", type=str, required=True,
                    help="Path to trained model (.pt file)")
    ap.add_argument("--kernel", type=int, default=3,
                    help="Convolution kernel size (must match training)")
    ap.add_argument("--depth", type=str, default="original",
                    choices=["shallow", "original", "deep"],
                    help="Model architecture depth (must match training)")
    ap.add_argument("--batch", type=int, default=64,
                    help="Batch size for evaluation")
    ap.add_argument("--epsilons", type=str, default="0.0,0.01,0.03,0.05,0.1",
                    help="Comma-separated epsilon values for FGSM attack")
    args = ap.parse_args()

    print("=" * 60)
    print(f"STEP 3: ADVERSARIAL ATTACK (FGSM)")
    print(f"Model: {args.model}")
    print(f"Depth: {args.depth} | Kernel: {args.kernel}")
    print("=" * 60)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\nError: Model file not found at {model_path}")
        print("Please run step1_train.py first to train a model.")
        exit(1)

    # Parse epsilon values
    epsilons = [float(e.strip()) for e in args.epsilons.split(',')]

    attack_model(model_path, k=args.kernel, depth=args.depth,
                batch=args.batch, epsilons=epsilons)
