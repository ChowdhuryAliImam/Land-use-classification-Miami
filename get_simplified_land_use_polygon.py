import requests
import geopandas as gpd
from shapely.geometry import box
import numpy as np
import geopandas as gpd

LAND_USE_FEATURESERVER = "https://services.arcgis.com/8Pc9XBTAsYuxx9Ny/arcgis/rest/services/LUMALanduse_gdb/FeatureServer/0"


def fetch_landuse_aoi(xmin, ymin, xmax, ymax, out_crs_epsg=2236):
    geometry = {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "spatialReference": {"wkid": out_crs_epsg},
    }

    params = {
        "where": "1=1",
        "geometry": str(geometry).replace("'", '"'),   
        "geometryType": "esriGeometryEnvelope",
        "inSR": out_crs_epsg,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "f": "geojson",
    }

    resp = requests.get(f"{LAND_USE_FEATURESERVER}/query", params=params)
    resp.raise_for_status()

    data = resp.json()
    gdf = gpd.GeoDataFrame.from_features(data["features"])
    gdf.set_crs(epsg=out_crs_epsg, inplace=True)
    return gdf

landuse_aoi = fetch_landuse_aoi(898093.4677176675, 539736.8840555587,907935.9877176675, 549579.4040555587, out_crs_epsg=2236)
print(landuse_aoi.head())
print(landuse_aoi.shape)
print(len(landuse_aoi['DESCR'].unique()))

def simplify_descr(desc):
    d = str(desc).lower()

    # 1. Water
    if "water" in d or "lakes" in d or "ponds" in d:
        return "WATER"

    # 2. Vacant
    if "vacant" in d:
        return "VACANT"

    # 3. Parks / open space / cemeteries
    if "park" in d or "cemeter" in d:
        return "PARKS_OPEN_SPACE"

    # 4. Transport infrastructure (roads, parking, rail, terminals, drives)
    if ("street" in d or "roads" in d or "railroad" in d
        or "parking" in d or "terminal" in d
        or "bus/truck" in d or "freight" in d
        or "private drives" in d or "right-of-way" in d):
        return "TRANSPORT_INFRA"

    # 5. Industrial
    if "industrial" in d or "junk yard" in d:
        return "INDUSTRIAL"

    # 6. Utilities / infrastructure
    if ("sewerage" in d or "treatment plant" in d or "electric power" in d
        or "generator" in d or "substation" in d or "communications" in d
        or "oil and gas storage" in d or "tank farms" in d
        or "motor pools" in d or "maintenance and storage yards" in d):
        return "UTILITIES_INFRA"

    # 7. Civic / institutional
    if ("school" in d or "hospital" in d or "clinic" in d or "medical" in d
        or "nursing home" in d or "assisted living" in d
        or "social services" in d or "charitable" in d
        or "governmental/public administration" in d
        or "houses of worship" in d or "religious" in d):
        return "CIVIC_INSTITUTIONAL"

    # 8. Residential 
    if "office" in d and "residential" in d:
        return "COMMERCIAL_MIXED"

    if ("single-family" in d or "two-family" in d or "multi-family" in d
        or "mobile home" in d or "residential sf" in d
        or "residential mf" in d
        or ("residential" in d and "transient" not in d)):
        return "RESIDENTIAL"

    # 9. Commercial / mixed
    if ("sales and services" in d or "shopping center" in d
        or "strip commercial" in d or "office building" in d
        or "business" in d or "hotel" in d or "motel" in d
        or "transient-residential" in d):
        return "COMMERCIAL_MIXED"

    # Fallback
    return "OTHER"
landuse_aoi["simple_class"] = landuse_aoi["DESCR"].apply(simplify_descr)

print(landuse_aoi["simple_class"].value_counts())
print("Num simplified classes:", landuse_aoi["simple_class"].nunique())

simple_classes = sorted(landuse_aoi["simple_class"].unique())
class_to_id = {c: i for i, c in enumerate(simple_classes)}
id_to_class = {i: c for c, i in class_to_id.items()}

landuse_aoi["class_id"] = landuse_aoi["simple_class"].map(class_to_id)

print("Class mapping:", class_to_id)
print(landuse_aoi[["DESCR", "simple_class", "class_id"]].head())

import json
with open("landuse_class_mapping.json", "w") as f:
    json.dump(class_to_id, f, indent=2)

# After adding simple_class and class_id
landuse_aoi.to_file(
    "landuse_3km_simplified.gpkg",
    layer="landuse_3km_simplified",
    driver="GPKG"
)

#AI Involvement: ChatGPT was used for debugging and code optimization