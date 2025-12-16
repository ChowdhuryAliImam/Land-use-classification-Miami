import os
import json
import numpy as np
import rasterio
import pandas as pd

ortho_path = "orthophoto_3km_georef.tif"
label_path = "landuse_labels_3km.tif"

out_img_dir = "tiles/images"
out_lbl_dir = "tiles/labels"
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_lbl_dir, exist_ok=True)

tile_size = 128
min_valid_ratio = 0.1   

meta_rows = []

with rasterio.open(ortho_path) as src_img, rasterio.open(label_path) as src_lbl:
    assert src_img.width == src_lbl.width
    assert src_img.height == src_lbl.height
    assert src_img.crs == src_lbl.crs
    assert src_img.transform == src_lbl.transform

    C = src_img.count
    H = src_img.height
    W = src_img.width

    tile_id = 0
    for y in range(0, H - tile_size + 1, tile_size):
        iy = y // tile_size
        for x in range(0, W - tile_size + 1, tile_size):
            ix = x // tile_size

            # reading window
            img_tile = src_img.read(window=((y, y + tile_size), (x, x + tile_size)))  
            lbl_tile = src_lbl.read(1, window=((y, y + tile_size), (x, x + tile_size)))  

            # skipping tiles that are mostly nodata (-1)
            valid = (lbl_tile != -1)
            valid_ratio = valid.mean()
            if valid_ratio < min_valid_ratio:
                continue

            img_path = os.path.join(out_img_dir, f"img_{tile_id:05d}.npy")
            lbl_path = os.path.join(out_lbl_dir, f"lbl_{tile_id:05d}.npy")

            np.save(img_path, img_tile.astype(np.float32))
            np.save(lbl_path, lbl_tile.astype(np.int16))

            meta_rows.append({"tile_id": tile_id, "img_path": img_path, "lbl_path": lbl_path,"x_pix": x,
                "y_pix": y, "ix": ix,"iy": iy,"valid_ratio": float(valid_ratio),})

            tile_id += 1

meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv("tiles/tiles_meta_raw.csv", index=False)
print("Saved", len(meta_df), "tiles to tiles/ and metadata to tiles/tiles_meta_raw.csv")


#AI Involvement: ChatGPT was used for debugging and code optimization