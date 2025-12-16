
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

class LandUseTileDataset(Dataset):
    def __init__(self, meta_csv, split, augment=False):
        self.meta = pd.read_csv(meta_csv)
        self.meta = self.meta[self.meta["split"] == split].reset_index(drop=True)
        self.split = split
        self.augment = augment

        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img = np.load(row["img_path"])   
        lbl = np.load(row["lbl_path"])  

        if img.shape[0] > 3:
            img = img[:3, :, :]

        img_hwc = np.transpose(img, (1, 2, 0))  

        if self.transform is not None:
            augmented = self.transform(image=img_hwc, mask=lbl)
            img_hwc = augmented["image"]
            lbl = augmented["mask"]

        img = np.transpose(img_hwc, (2, 0, 1))  # (C, H, W)

        if img.max() > 1.5:
            img = img / 255.0

        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std  = np.array([0.229, 0.224, 0.225])[:, None, None]
        img = (img - mean) / std

        # labels: nodata (-1) to 255
        lbl = lbl.astype(np.int64)
        lbl[lbl < 0] = 255

        img = torch.from_numpy(img.astype(np.float32))
        lbl = torch.from_numpy(lbl)

        return img, lbl

def make_loaders(meta_csv, batch_size=8, num_workers=0):
    meta = pd.read_csv(meta_csv, header=0)

    # inferring n_classes from ALL label tiles
    max_label = -1
    for p in meta["lbl_path"]:
        arr = np.load(p)
        arr = arr[arr >= 0] 
        if arr.size > 0:
            m = int(arr.max())
            if m > max_label:
                max_label = m
    n_classes = max_label + 1
    print("Inferred n_classes from label files:", n_classes)

    train_ds = LandUseTileDataset(meta_csv, split="train", augment=True)
    val_ds   = LandUseTileDataset(meta_csv, split="val", augment=False)
    test_ds  = LandUseTileDataset(meta_csv, split="test", augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, n_classes

#AI Involvement: ChatGPT was used for debugging and code optimization