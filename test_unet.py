
import os
import json
import numpy as np
import torch
import segmentation_models_pytorch as smp
import pandas as pd
import matplotlib.pyplot as plt
from dataset import make_loaders

ENCODER_NAME = "resnet50"     
ENCODER_WEIGHTS = None        
MODEL_WEIGHTS_PATH = "unet_best.pth"
META_CSV = r"tiles/tiles_meta_split_balanced_prop.csv"

BATCH_SIZE_TEST = 8
NUM_WORKERS_TEST = 0          


def build_model(n_classes, device):
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=n_classes,
    ).to(device)

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"Model weights not found: {MODEL_WEIGHTS_PATH}")

    state = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def accumulate_confusion(conf_mat, true_labels, pred_labels, n_classes):

    true = true_labels.astype(np.int64)
    pred = pred_labels.astype(np.int64)
    idx = true * n_classes + pred
    binc = np.bincount(idx, minlength=n_classes * n_classes)
    conf_mat += binc.reshape((n_classes, n_classes))
    return conf_mat


def compute_metrics_from_confusion(conf_mat):

    n_classes = conf_mat.shape[0]
    tp = np.diag(conf_mat)
    support_true = conf_mat.sum(axis=1)  
    support_pred = conf_mat.sum(axis=0) 
    total = conf_mat.sum()

    overall_acc = tp.sum() / total if total > 0 else 0.0

    per_class_records = []
    for c in range(n_classes):
        st = support_true[c]
        sp = support_pred[c]
        tpc = tp[c]

        acc_c = tpc / st if st > 0 else None
        prec_c = tpc / sp if sp > 0 else None
        rec_c = tpc / st if st > 0 else None

        if prec_c is not None and rec_c is not None and (prec_c + rec_c) > 0:
            f1_c = 2 * prec_c * rec_c / (prec_c + rec_c)
        else:
            f1_c = None

        per_class_records.append({
            "class_id": c,
            "accuracy": acc_c,
            "precision": prec_c,
            "recall": rec_c,
            "f1": f1_c,
            "support_true": int(st),
            "support_pred": int(sp),
        })

    per_class_df = pd.DataFrame(per_class_records)

    # macro-average over classes that actually appear in ground truth
    valid = per_class_df["support_true"] > 0
    macro_precision = per_class_df.loc[valid, "precision"].mean()
    macro_recall = per_class_df.loc[valid, "recall"].mean()
    macro_f1 = per_class_df.loc[valid, "f1"].mean()

    return overall_acc, per_class_df, macro_precision, macro_recall, macro_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, n_classes = make_loaders(
        META_CSV,
        batch_size=BATCH_SIZE_TEST,
        num_workers=NUM_WORKERS_TEST,
    )
    print("n_classes:", n_classes)
    print("Test batches:", len(test_loader))

    model = build_model(n_classes, device)
    conf_mat = np.zeros((n_classes, n_classes), dtype=np.int64)

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            # ignore_index=255
            mask = (labels != 255)
            if not mask.any():
                continue

            logits = model(imgs)
            preds = logits.argmax(dim=1)

            # moving to CPU numpy for confusion matrix
            labels_np = labels[mask].cpu().numpy().ravel()
            preds_np  = preds[mask].cpu().numpy().ravel()
            labels_np = np.clip(labels_np, 0, n_classes - 1)
            preds_np  = np.clip(preds_np,  0, n_classes - 1)
            conf_mat = accumulate_confusion(conf_mat, labels_np, preds_np, n_classes)

    overall_acc, per_class_df, macro_prec, macro_rec, macro_f1 = compute_metrics_from_confusion(conf_mat)

    print("\n=== TEST SET METRICS ===")
    print(f"Overall pixel accuracy: {overall_acc:.4f}")
    print(f"Macro precision:{macro_prec:.4f}")
    print(f"Macro recall:{macro_rec:.4f}")
    print(f"Macro F1:{macro_f1:.4f}")

    print("\nPer-class metrics:")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 120):
        print(per_class_df)

    #saving per-class metrics to CSV
    per_class_df.to_csv("new_test_per_class_metrics.csv", index=False)
    print("\nSaved per-class metrics to test_per_class_metrics.csv")
    
    #Confusion matrix heatmap
    try:
        with open("landuse_class_mapping.json", "r") as f:
            name_to_id = json.load(f)
        id_to_name = {v: k for k, v in name_to_id.items()}
        class_labels = [id_to_name.get(i, str(i)) for i in range(n_classes)]
    except FileNotFoundError:
        class_labels = [str(i) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(conf_mat, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticklabels(class_labels)

    # Annotating each cell with the raw count
    max_val = conf_mat.max() if conf_mat.size > 0 else 0
    for i in range(n_classes):
        for j in range(n_classes):
            val = int(conf_mat[i, j])
            if max_val > 0:
                color = "white" if val > 0.5 * max_val else "black"
            else:
                color = "black"
            ax.text(
                j, i, str(val),
                ha="center", va="center",
                fontsize=6,
                color=color,
            )

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200)
    print("Saved confusion matrix heatmap to confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()

#AI Involvement: ChatGPT was used for debugging and code optimization