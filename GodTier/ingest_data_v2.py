import laspy
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
import multiprocessing
import os
import zipfile
import shutil
import traceback
import gc

INPUT_DIR = Path("../Point-Cloud")
OUTPUT_DIR = Path("Processed_Data/GodTier_V2/Clouds")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("Temp_Unzip_V2")

TILE_SIZE = 100.0
MAX_WORKERS = 2
K_NEIGHBORS = 12
SOR_THRESHOLD = 2.5

def process_tile(points, indices, k=K_NEIGHBORS):
    if len(points) < k + 1:
        return indices
    try:
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=k, workers=1)
        mean_dists = np.mean(dists[:, 1:], axis=1)
        global_mean = np.mean(mean_dists)
        global_std = np.std(mean_dists)
        mask = mean_dists < (global_mean + SOR_THRESHOLD * global_std)
        return indices[mask]
    except Exception as e:
        print(f"    [WARN] Tile filter failed: {e}")
        return indices

def process_laz_tiled(las_path):
    try:
        out_path = OUTPUT_DIR / (las_path.stem + ".laz")
        if out_path.exists():
            print(f"[SKIP] {las_path.name}")
            return
        print(f"Loading {las_path.name}...")
        las = laspy.read(las_path)
        n_points = len(las.points)
        print(f"  Total points: {n_points:,}")
        x = np.array(las.x)
        y = np.array(las.y)
        z = np.array(las.z)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        n_tiles_x = int(np.ceil((x_max - x_min) / TILE_SIZE))
        n_tiles_y = int(np.ceil((y_max - y_min) / TILE_SIZE))
        total_tiles = n_tiles_x * n_tiles_y
        print(f"  Processing {total_tiles} tiles ({n_tiles_x}x{n_tiles_y})...")
        valid_indices = []
        tiles_done = 0
        for i in range(n_tiles_x):
            for j in range(n_tiles_y):
                tx_min = x_min + i * TILE_SIZE - 1.0
                tx_max = x_min + (i + 1) * TILE_SIZE + 1.0
                ty_min = y_min + j * TILE_SIZE - 1.0
                ty_max = y_min + (j + 1) * TILE_SIZE + 1.0
                tile_mask = (x >= tx_min) & (x < tx_max) & (y >= ty_min) & (y < ty_max)
                tile_indices = np.where(tile_mask)[0]
                if len(tile_indices) > 0:
                    tile_points = np.column_stack((x[tile_indices], y[tile_indices], z[tile_indices]))
                    filtered_indices = process_tile(tile_points, tile_indices)
                    valid_indices.extend(filtered_indices.tolist())
                tiles_done += 1
                if tiles_done % 50 == 0:
                    print(f"    Tiles: {tiles_done}/{total_tiles}")
                if tiles_done % 100 == 0:
                    gc.collect()
        valid_indices = np.unique(valid_indices)
        print(f"  Writing {len(valid_indices):,} / {n_points:,} points...")
        new_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
        new_las.points = las.points[valid_indices]
        new_las.write(out_path)
        del las, x, y, z, valid_indices
        gc.collect()
        print(f"[OK] Ingested {las_path.stem}")
    except Exception as e:
        print(f"[FAIL] {las_path.name}: {e}")
        traceback.print_exc()

def worker_task(file_path):
    if file_path.suffix == '.zip':
        try:
            worker_temp = TEMP_DIR / f"worker_{os.getpid()}"
            worker_temp.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(file_path, 'r') as z:
                extracted = []
                for name in z.namelist():
                    if name.lower().endswith(('.las', '.laz')):
                        z.extract(name, worker_temp)
                        extracted.append(worker_temp / name)
                for f in extracted:
                    process_laz_tiled(f)
                    try: f.unlink()
                    except: pass
            try: shutil.rmtree(worker_temp)
            except: pass
        except Exception as e:
            print(f"[FAIL] ZIP {file_path.name}: {e}")
            traceback.print_exc()
    else:
        process_laz_tiled(file_path)

if __name__ == "__main__":
    if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir()
    files = list(INPUT_DIR.rglob("*.zip")) + list(INPUT_DIR.rglob("*.las")) + list(INPUT_DIR.rglob("*.laz"))
    print(f"Found {len(files)} files.")
    print(f"Spawning {MAX_WORKERS} workers for Memory-Safe Ingestion...")
    print(f"Tile Size: {TILE_SIZE}m | K-Neighbors: {K_NEIGHBORS}")
    with multiprocessing.Pool(processes=MAX_WORKERS) as pool:
        pool.map(worker_task, files)
    if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
    print("\n[DONE] Ingestion Complete.")
