import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


#Load label raster
label_path = "landuse_labels_3km.tif"

with rasterio.open(label_path) as src:
    labels = src.read(1)  # first (and only) band
    print("dtype:", labels.dtype)
    unique_vals = np.unique(labels)
    print("Unique values:", unique_vals)

    vals, counts = np.unique(labels, return_counts=True)
    print("Value counts:")
    for v, c in zip(vals, counts):
        print(f"value {v}: {c} pixels")

#integer type
labels = labels.astype(int)

# Masking nodata for plotting 
masked = np.ma.masked_equal(labels, -1)


# 2. Define class mapping and colors

class_to_id = {
    'CIVIC_INSTITUTIONAL': 0,'COMMERCIAL_MIXED': 1,'INDUSTRIAL': 2,'OTHER':3,'PARKS_OPEN_SPACE':4,
    'RESIDENTIAL':5,'TRANSPORT_INFRA':6,'UTILITIES_INFRA': 7,'VACANT':8,'WATER': 9}

class_colors = {'CIVIC_INSTITUTIONAL': "#1f77b4" ,  'COMMERCIAL_MIXED': "#ff7f0e", 'INDUSTRIAL': "#d62728", 'PARKS_OPEN_SPACE':  "#2ca02c",  'RESIDENTIAL':         "#bcbd22",   'TRANSPORT_INFRA':     "#7f7f7f",  'UTILITIES_INFRA':     "#9467bd", 
               'VACANT': "#8c564b", 'WATER': "#17becf",  'OTHER': "#e377c2" }

#Sorting classes by numeric id (0..9)
sorted_classes = sorted(class_to_id.items(), key=lambda kv: kv[1])

# Building color list in order of class_id
color_list = [class_colors[cname] for cname, cid in sorted_classes]

# Building discrete colormap 
cmap = ListedColormap(color_list)
max_id = max(class_to_id.values())

# boundaries: [-0.5, 0.5, 1.5, ..., max_id+0.5]
boundaries = np.arange(-0.5, max_id + 1.5, 1)
norm = BoundaryNorm(boundaries, cmap.N)

# Plotting
plt.figure(figsize=(6, 6))
im = plt.imshow(masked, cmap=cmap, norm=norm)
plt.axis("off")

# Colorbar with integer ticks (0..max_id)
cbar = plt.colorbar(im, boundaries=boundaries, ticks=np.arange(0, max_id + 1))
cbar.ax.set_yticklabels([str(i) for i in range(0, max_id + 1)])
plt.tight_layout()
plt.show()


#AI Involvement: ChatGPT was used for debugging and code optimization