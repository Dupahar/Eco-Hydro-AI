import laspy
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
import multiprocessing
import os
import zipfile
import shutil
import traceback
import time

INPUT_DIR = Path("../../Data/Point-Cloud")
if not INPUT_DIR.exists(): INPUT_DIR = Path("Data/Point-Cloud")

OUTPUT_DIR = Path("Processed_Data/FullScale")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("Temp_Unzip")

def process_laz(las_path):
    try:
        out_path = OUTPUT_DIR / (las_path.stem + ".laz")
        if out_path.exists(): 
            print(f"⏩ Skipping {las_path.name} (Exists)")
            return

        print(f"Reading {las_path.name}...")
        las = laspy.read(las_path)
        
        # Extract coordinates
        points = np.vstack((las.x, las.y, las.z)).transpose()
        
        # Statistical Outlier Removal (SOR)
        # mean_k=8, multiplier=3.0
        print(f"Filtering {las_path.name} (SOR)...")
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=9, workers=1) # k=9 because self is included
        mean_dists = np.mean(dists[:, 1:], axis=1) # Exclude self
        
        global_mean = np.mean(mean_dists)
        global_std = np.std(mean_dists)
        
        mask = mean_dists < (global_mean + 3.0 * global_std)
        
        # Save filtered
        new_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
        new_las.points = las.points[mask]
        new_las.write(out_path)
        
        print(f"✅ Ingested {las_path.stem} -> {len(points)} to {np.sum(mask)} pts")
            
    except Exception as e:
        print(f"❌ Error {las_path.name}: {e}")
        traceback.print_exc()

def worker_task(file_path):
    if file_path.suffix == '.zip':
        try:
            with zipfile.ZipFile(file_path, 'r') as z:
                extracted = []
                for name in z.namelist():
                    if name.lower().endswith(('.las', '.laz')):
                        z.extract(name, TEMP_DIR)
                        extracted.append(TEMP_DIR / name)
                
                for f in extracted: process_laz(f)
        except Exception as e:
            print(f"❌ Zip Error {file_path}:")
            traceback.print_exc()
    else:
        process_laz(file_path)

if __name__ == "__main__":
    if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir()
    
    files = list(INPUT_DIR.rglob("*.zip")) + list(INPUT_DIR.rglob("*.las")) + list(INPUT_DIR.rglob("*.laz"))
    print(f"Found {len(files)} source files.")
    
    # Process sequentially or parallel? Scipy cKDTree is memory hungry.
    # We will use fewer processes to be safe on RAM.
    with multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count()//4)) as pool:
        pool.map(worker_task, files)
        
    if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
