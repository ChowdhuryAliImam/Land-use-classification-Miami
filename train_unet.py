
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from dataset import make_loaders  


def compute_metrics(logits, labels, n_classes):
    with torch.no_grad():
        preds = logits.argmax(dim=1) 
        mask = (labels != 255)

        correct = (preds[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        overall_acc = correct / total if total > 0 else 0.0

        per_class_correct = torch.zeros(n_classes, dtype=torch.long, device=preds.device)
        per_class_total   = torch.zeros(n_classes, dtype=torch.long, device=preds.device)

        for c in range(n_classes):
            class_mask = (labels == c)
            per_class_correct[c] = (preds[class_mask] == labels[class_mask]).sum()
            per_class_total[c]   = class_mask.sum()

        per_class_acc = []
        for c in range(n_classes):
            if per_class_total[c] > 0:
                per_class_acc.append((c, (per_class_correct[c] / per_class_total[c]).item()))
            else:
                per_class_acc.append((c, None))

    return overall_acc, per_class_acc


def run_epoch(model, loader, criterion, optimizer, device, n_classes, train=True):
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    all_acc = []

    for imgs, labels in tqdm(loader, desc="train" if train else "val"):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad()

        logits = model(imgs)
        loss = criterion(logits, labels)
        epoch_loss += loss.item() * imgs.size(0)

        overall_acc, _ = compute_metrics(logits, labels, n_classes)
        all_acc.append(overall_acc)

        if train:
            loss.backward()
            optimizer.step()

    epoch_loss /= len(loader.dataset)
    mean_acc = sum(all_acc) / len(all_acc) if all_acc else 0.0
    return epoch_loss, mean_acc


#HYPERPARAMETERS 
ENCODER_NAME = "resnet50"         
ENCODER_WEIGHTS = "imagenet"      
FREEZE_ENCODER = True           
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-4            
BATCH_SIZE = 16
NUM_EPOCHS = 40
NUM_WORKERS = 0              
USE_SCHEDULER = True


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    #data
    meta_csv = r"tiles/tiles_meta_split_balanced_prop.csv"

    train_loader, val_loader, test_loader, n_classes = make_loaders(
        meta_csv, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    print("n_classes:", n_classes)

    # model
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=n_classes,
    ).to(device)

    if FREEZE_ENCODER:
        for p in model.encoder.parameters():
            p.requires_grad = False
        #unfreeze last few layers
        for name, p in list(model.encoder.named_parameters())[-20:]:
            p.requires_grad = True

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    if USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            verbose=True,
        )
    else:
        scheduler = None

    # training loop
    best_val_acc = 0.0
    history = {"train_loss": [],"val_loss": [], "train_acc": [],"val_acc": [],"lr": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        current_lr = optimizer.param_groups[0]["lr"]
        print("Current LR:", current_lr)

        history["lr"].append(current_lr)  # <-- store LR each epoch

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, n_classes, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, n_classes, train=False
        )

        print(f"Train loss: {train_loss:.4f}, overall acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f}, overall acc: {val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if scheduler is not None:
            scheduler.step(val_loss)  

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "unet_best.pth")
            print("Saved new best model with val acc:", best_val_acc)

    # PLOTS AFTER TRAINING
    epochs = range(1, NUM_EPOCHS + 1)

    # Loss curve
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"],   label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)

    # Accuracy curve
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train accuracy")
    plt.plot(epochs, history["val_acc"],   label="Val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Overall pixel accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_curve.png", dpi=150)

    # Learning rate curve
    plt.figure()
    plt.plot(epochs, history["lr"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Learning rate schedule")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lr_curve.png", dpi=150)

    print("Saved curves: loss_curve.png, accuracy_curve.png, lr_curve.png")


if __name__ == "__main__":
    main()

#AI Involvement: ChatGPT was used for debugging and code optimization