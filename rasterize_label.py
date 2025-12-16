import rasterio
import geopandas as gpd
from rasterio import features
import numpy as np

ortho_path = "orthophoto_3km_georef.tif"
landuse_path = "landuse_3km_simplified.gpkg"

# Load orthophoto
with rasterio.open(ortho_path) as src:
    ortho_img = src.read()
    ortho_meta = src.meta.copy()
    transform = src.transform
    height = src.height
    width = src.width
    raster_crs = src.crs
    raster_bounds = src.bounds

print("Ortho shape:", ortho_img.shape)
print("Raster CRS:", raster_crs)
print("Raster bounds:", raster_bounds)

# Load land use
landuse_aoi = gpd.read_file(landuse_path, layer="landuse_3km_simplified")
print("Original landuse CRS:", landuse_aoi.crs)
print("Original landuse bounds:", landuse_aoi.total_bounds)

landuse_aoi = landuse_aoi.set_crs(epsg=4326, allow_override=True)

# Reproject to raster CRS (EPSG:2236)
if landuse_aoi.crs != raster_crs:
    landuse_aoi = landuse_aoi.to_crs(raster_crs)

print("Reprojected landuse CRS:", landuse_aoi.crs)
print("Reprojected landuse bounds:", landuse_aoi.total_bounds)
#class id
assert "class_id" in landuse_aoi.columns, "class_id column missing"
print("class_id nunique:", landuse_aoi["class_id"].nunique())

#shape list
shape_list = []
for geom, cid in zip(landuse_aoi.geometry, landuse_aoi["class_id"]):
    if geom is None or geom.is_empty:
        continue
    if cid is None or (isinstance(cid, float) and np.isnan(cid)):
        continue
    shape_list.append((geom, int(cid)))

print("Number of shapes to rasterize:", len(shape_list))

label_raster = features.rasterize(
    shapes=shape_list,
    out_shape=(height, width),
    transform=transform,
    fill=-1,
    dtype="int16",
)

print("Unique labels after rasterize:", np.unique(label_raster))

label_meta = ortho_meta.copy()
label_meta.update({
    "driver": "GTiff",   # critical!
    "count": 1,
    "dtype": "int16",
    "nodata": -1,
})

label_path = "landuse_labels_3km.tif"

with rasterio.open(label_path, "w", **label_meta) as dst:
    dst.write(label_raster, 1)

print("Saved label raster to:", label_path)

# checking labels
with rasterio.open(label_path) as src:
    labels = src.read(1)
    print("dtype:", labels.dtype)
    print("Unique values:", np.unique(labels))

#AI Involvement: ChatGPT was used for debugging and code optimization