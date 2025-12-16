import pandas as pd

meta = pd.read_csv("tiles/tiles_meta_raw.csv")

# iy is the row index (0 at top)
ny = meta["iy"].max() + 1
print("Number of rows:", ny)

# defining fractions for vertical split 
test_frac  = 0.2  
val_frac   = 0.1   
test_y_max = int(ny * test_frac)        
val_y_max  = int(ny * (test_frac + val_frac))  

def assign_split(iy):
    if iy < test_y_max:
        return "test"    
    elif iy < val_y_max:
        return "val"    
    else:
        return "train"  

meta["split"] = meta["iy"].apply(assign_split)

print(meta["split"].value_counts())
meta.to_csv("tiles/tiles_meta_split.csv", index=False)
print("Saved row-wise split to tiles/tiles_meta_split.csv")

#AI Involvement: ChatGPT was used for debugging and code optimization