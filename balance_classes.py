
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

META_IN = "tiles/tiles_meta_split.csv"                 
META_OUT = "tiles/tiles_meta_split_balanced_prop.csv" 

# Max factor by which any class's tile count can grow
CAP_FACTOR = 3.0   

def main():
    meta = pd.read_csv(META_IN)

    train_meta = meta[meta["split"] == "train"].reset_index(drop=True)
    val_meta   = meta[meta["split"] == "val"].reset_index(drop=True)
    test_meta  = meta[meta["split"] == "test"].reset_index(drop=True)

    #computiing dominant class + per-class pixel counts ----
    dom_classes = []
    class_pixel_counts = {}  

    print("Scanning train tiles to compute dominant class and pixel counts...")
    for i, row in train_meta.iterrows():
        lbl = np.load(row["lbl_path"])  
        lbl = lbl[lbl >= 0]            

        if lbl.size == 0:
            dom_classes.append(-1)
            continue

        vals, counts = np.unique(lbl, return_counts=True)

        # dominant class for this tile (most pixels)
        dom_c = int(vals[counts.argmax()])
        dom_classes.append(dom_c)

        # pixel counts per class
        for v, c in zip(vals, counts):
            v = int(v)
            class_pixel_counts[v] = class_pixel_counts.get(v, 0) + int(c)

    train_meta["dom_class"] = dom_classes
    # drop tiles with no valid labels at all
    train_meta = train_meta[train_meta["dom_class"] >= 0].reset_index(drop=True)

    # sort classes 
    class_ids = sorted(class_pixel_counts.keys())
    print("\nTotal pixel counts per class in TRAIN:")
    for cid in class_ids:
        print(f"  class {cid}: {class_pixel_counts[cid]} pixels")

    # defining target number of tiles per class
    max_pixels = max(class_pixel_counts.values())
    print("\nMax pixels among classes:", max_pixels)

    # current number of tiles per dominant class
    tile_counts = train_meta["dom_class"].value_counts().to_dict()
    print("\nCurrent tile counts per dominant class:")
    for cid in sorted(tile_counts.keys()):
        print(f"  class {cid}: {tile_counts[cid]} tiles")

    sampling_strategy = {}
    for cid in class_ids:
        pixels_c = class_pixel_counts[cid]
        base_n_tiles = tile_counts.get(cid, 0)

        if pixels_c == 0 or base_n_tiles == 0:
            continue

        # how many times smaller than the max-pixel class
        multiplier = max_pixels / float(pixels_c)

        # target tiles = base tiles * multiplier
        target_tiles = int(round(base_n_tiles * multiplier))

        # capping to avoid extreme upsampling
        max_cap = int(base_n_tiles * CAP_FACTOR)
        target_tiles = min(target_tiles, max_cap)

        # only upsample or keep same
        target_tiles = max(target_tiles, base_n_tiles)

        sampling_strategy[cid] = target_tiles

    print("\nSampling strategy (target tiles per class, proportionally increased):")
    for cid in sorted(sampling_strategy.keys()):
        print(f"  class {cid}: from {tile_counts.get(cid, 0)} -> {sampling_strategy[cid]} tiles")

    # RandomOverSampler
    X = train_meta.index.values.reshape(-1, 1)
    y = train_meta["dom_class"].values

    ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
    X_res, y_res = ros.fit_resample(X, y)

    print("\nResampled train dominant-class distribution:")
    unique, counts = np.unique(y_res, return_counts=True)
    for cid, cnt in zip(unique, counts):
        print(f"  class {cid}: {cnt} tiles")

    # building balanced train_meta by duplicating rows ----
    idx_resampled = X_res.ravel()
    train_meta_balanced = train_meta.iloc[idx_resampled].reset_index(drop=True)
    train_meta_balanced = train_meta_balanced.drop(columns=["dom_class"])

    # combine with unchanged val,test 
    balanced_meta = pd.concat(
        [train_meta_balanced, val_meta, test_meta],
        ignore_index=True)

    balanced_meta.to_csv(META_OUT, index=False)
    print(f"\nSaved proportionally balanced metadata to {META_OUT}")
    print("New train size:", len(train_meta_balanced))
    print("Val size:", len(val_meta), "Test size:", len(test_meta))


if __name__ == "__main__":
    main()

#AI Involvement: ChatGPT was used for debugging and code optimization