import requests
from pyproj import Transformer

# Defining  AOI center in lon/lat
center_lon = -80.24953543355088
center_lat = 25.83013736691271

# transformer: WGS84 -> EPSG:2236 
transformer = Transformer.from_crs("EPSG:4326", "EPSG:2236", always_xy=True)
cx_feet, cy_feet = transformer.transform(center_lon, center_lat)

# Compute 3 km box in feet
meters_to_feet = 3.28084
half_side_m = 1500.0
half_side_ft = half_side_m * meters_to_feet  

xmin = cx_feet - half_side_ft
xmax = cx_feet + half_side_ft
ymin = cy_feet - half_side_ft
ymax = cy_feet + half_side_ft

# Call the MDC MapServer 
export_url = "https://gisweb.miamidade.gov/arcgis/rest/services/MapCache/MDCImagery/MapServer/export"

# image size
width = 4096
height = 4096

params = {
    "bbox": f"{xmin},{ymin},{xmax},{ymax}",
    "bboxSR": 2236,
    "imageSR": 2236,    
    "size": f"{width},{height}",
    "format": "tiff",   
    "f": "image",     
}
#saving response
response = requests.get(export_url, params=params)
response.raise_for_status()  

with open("orthophoto_3km.tif", "wb") as f:
    f.write(response.content)

print("Saved orthophoto_3km.tif")

#AI Involvement: ChatGPT was used for debugging and code optimization
