import rasterio
from rasterio.transform import Affine

in_path = "orthophoto_3km.tif"
out_path = "orthophoto_3km_georef.tif"

with rasterio.open(in_path) as src:
    arr = src.read()
    height = src.height
    width = src.width
    profile = src.profile

xres = (907935.9877176675 - 898093.4677176675) / width
yres = (549579.4040555587 - 539736.8840555587) / height

transform = Affine(xres, 0, 898093.4677176675, 0, -yres,549579.4040555587)

profile.update({
    "crs": "EPSG:2236",
    "transform": transform
})

with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(arr)

#AI Involvement: ChatGPT was used for debugging and code optimization