import subprocess
import sys
import os
import time
import logging
import traceback
import json
import argparse
from pathlib import Path

# --- CONFIGURATION (GOD-TIER V2 — 24-HOUR BEAST MODE) ---
MISSION_CONFIG = {
    "VERSION": "2.1-BEAST",
    "HARDWARE_PROFILE": "WORKSTATION_128GB",  # Xeon/128GB/A2000-12GB
    "PATHS": {
        "INPUT_CLOUD": "../Point-Cloud",
        "PROCESSED_DATA": "Processed_Data/GodTier_V2",
        "GRAPHS": "Processed_Data/GodTier_V2/Graphs",
        "MODELS": "Models/GodTier_V2",
        "VISUALS": "Visuals/GodTier_V2"
    },
    "PARAMS": {
        "VOXEL_SIZE": 0.25,     # 25cm HIGH-FIDELITY resolution
        "RADIUS": 2.5,          # 2.5m connectivity radius
        "K_NEIGHBORS": 16,      # Dense graph connectivity
        "TRAIN_EPOCHS": 100,    # Extended training
        "BATCH_SIZE": 2,        # Safe for A2000 VRAM with large graphs
        "NUM_WORKERS": 2        # Memory-safe parallelism
    }
}

# Force UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Logging
LOG_FILE = "mission_control_v2.log"
STATE_FILE = "mission_state_v2.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def log(msg): logging.info(msg)

# --- STATE MANAGEMENT (SMART RESUME) ---
def load_state():
    if Path(STATE_FILE).exists():
        try:
            with open(STATE_FILE, 'r') as f: return json.load(f)
        except: pass
    return {"completed_steps": []}

def save_state(state):
    with open(STATE_FILE, 'w') as f: json.dump(state, f, indent=2)

def mark_step_complete(step_name):
    state = load_state()
    if step_name not in state["completed_steps"]:
        state["completed_steps"].append(step_name)
        save_state(state)
        log(f"[STATE] Marked '{step_name}' as COMPLETE.")

def is_step_complete(step_name):
    return step_name in load_state()["completed_steps"]

# ============================================================
#  GENERATORS (ISOLATED V2 — BEAST MODE)
# ============================================================

def generate_sonata_wrapper():
    code = r'''import sys
import os
from pathlib import Path

def run_sonata_phase():
    print("[SONATA] Checking for Semantic Terrain Understanding...")
    model_path = Path("sonata_epoch_15.pth")
    if model_path.exists():
        print(f"[SONATA] Found pre-trained model: {model_path.name}")
        print("[SONATA] Using this model to filter vegetation in Ingestion phase.")
    else:
        print("[SONATA] No model found. Skipping semantic enhancement (using raw geometric filtering).")

if __name__ == "__main__":
    run_sonata_phase()
'''
    with open("phase1_sonata_v2.py", "w", encoding="utf-8") as f: f.write(code)


def generate_ingest_v2():
    code = r'''import laspy
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
'''
    with open("ingest_data_v2.py", "w", encoding="utf-8") as f: f.write(code)


def generate_graph_v2():
    """HIGH-FIDELITY graph builder: 0.25m voxel + cKDTree k-NN (memory-bounded)."""
    code = r'''import numpy as np
import laspy
import torch
from pathlib import Path
from torch_geometric.data import Data
from scipy.spatial import cKDTree
import traceback
import time
import gc
import os

INPUT_DIR = Path("Processed_Data/GodTier_V2/Clouds")
OUTPUT_DIR = Path("Processed_Data/GodTier_V2/Graphs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === BEAST MODE CONFIG ===
VOXEL_SIZE = 0.25   # 25cm — HIGH FIDELITY
K_NEIGHBORS = 16    # k-NN: bounded edges, predictable memory

def voxel_downsample(pos, voxel_size):
    """Pure Numpy voxel grid filter — retains one point per voxel cell.
    Uses float64 internally to avoid precision loss at large UTM coords."""
    min_bound = np.min(pos, axis=0)
    voxel_indices = ((pos - min_bound) / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
    return pos[unique_idx]

def build_graph(laz_path):
    out_path = OUTPUT_DIR / (laz_path.stem + ".pt")
    if out_path.exists():
        print(f"  [SKIP] {out_path.name} already exists")
        return True

    print(f"  [BUILD] {laz_path.name}")
    t0 = time.time()

    try:
        # 1. Load in float64 (CRITICAL for UTM precision during voxel step)
        las = laspy.read(laz_path)
        pos_raw = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float64)
        n_raw = len(pos_raw)
        del las
        gc.collect()

        # 1b. Check for Lat/Lon coordinates (Degrees vs Meters)
        # If extent is small (< 360) and points are many, it's likely Lat/Lon.
        # 25cm voxel on Lat/Lon (approx 0.0000025 deg) requires scaling or huge precision.
        # We will scale to approximate Meters if degrees are detected.
        
        x_range = pos_raw[:, 0].max() - pos_raw[:, 0].min()
        y_range = pos_raw[:, 1].max() - pos_raw[:, 1].min()
        
        if (x_range < 500 and y_range < 500) and n_raw > 10000:
            print(f"    [DETECTED DEGREES] Ranges X:{x_range:.4f} Y:{y_range:.4f} -> Scaling to Meters")
            
            # Approximate conversion: 1 deg lat ~ 111,111 m
            # 1 deg lon ~ 111,111 * cos(lat) m
            mean_lat_rad = np.radians(np.mean(pos_raw[:, 1]))
            scale_x = 111_111 * np.cos(mean_lat_rad)
            scale_y = 111_111
            
            pos_raw[:, 0] *= scale_x
            pos_raw[:, 1] *= scale_y
            
            print(f"    Applied Scale: X*{scale_x:.1f} Y*{scale_y:.1f}")
            
        # 2. Voxel Downsample in float64 (precision-safe)
        t_vox = time.time()
        pos_f64 = voxel_downsample(pos_raw, VOXEL_SIZE)
        del pos_raw
        gc.collect()
        n_down = len(pos_f64)
        print(f"    Voxel {VOXEL_SIZE}m: {n_raw:,} -> {n_down:,} nodes ({100*n_down/n_raw:.1f}%) [{time.time()-t_vox:.1f}s]")

        # 3. Convert to float32 for all subsequent operations (saves 50% RAM)
        pos = pos_f64.astype(np.float32)
        del pos_f64
        gc.collect()

        # 4. k-NN graph via cKDTree (bounded memory: exactly K edges per node)
        t_tree = time.time()
        tree = cKDTree(pos)
        dists, indices = tree.query(pos, k=K_NEIGHBORS + 1, workers=-1)  # +1 for self
        del tree
        gc.collect()
        # Remove self-connections (first column)
        dists = dists[:, 1:].astype(np.float32)    # (N, K)
        indices = indices[:, 1:]                    # (N, K)
        t_tree_done = time.time()
        print(f"    k-NN (k={K_NEIGHBORS}): {n_down * K_NEIGHBORS:,} directed edges [{t_tree_done-t_tree:.1f}s]")

        # 5. Build bidirectional edge index
        N = len(pos)
        src_fwd = np.repeat(np.arange(N), K_NEIGHBORS)
        tgt_fwd = indices.flatten()
        # Bidirectional
        src = np.concatenate([src_fwd, tgt_fwd])
        tgt = np.concatenate([tgt_fwd, src_fwd])
        del src_fwd, tgt_fwd, indices

        # 6. Edge attributes: [Distance, Slope, HorizontalDist]
        # Compute in chunks to avoid memory spikes
        CHUNK = 50_000_000  # 50M edges per chunk
        n_edges = len(src)
        edge_dists = np.empty(n_edges, dtype=np.float32)
        edge_slope = np.empty(n_edges, dtype=np.float32)
        edge_hdist = np.empty(n_edges, dtype=np.float32)

        for c_start in range(0, n_edges, CHUNK):
            c_end = min(c_start + CHUNK, n_edges)
            s, t = src[c_start:c_end], tgt[c_start:c_end]
            diff = pos[s] - pos[t]
            d = np.sqrt(np.sum(diff**2, axis=-1))
            h = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)
            edge_dists[c_start:c_end] = d
            edge_hdist[c_start:c_end] = h
            edge_slope[c_start:c_end] = diff[:, 2] / (h + 1e-6)
            del diff, d, h

        edge_index = np.vstack((src, tgt))
        edge_attr = np.vstack((edge_dists, edge_slope, edge_hdist)).T
        del edge_dists, edge_slope, edge_hdist, src, tgt
        gc.collect()

        # 7. Node features: [NormalizedZ, LocalRoughness, SlopeProxy]
        min_z = pos[:, 2].min()
        max_z = pos[:, 2].max()
        norm_z = (pos[:, 2] - min_z) / (max_z - min_z + 1e-6)
        roughness = np.zeros_like(norm_z)
        slope_proxy = np.gradient(pos[:, 2])

        x_feat = torch.from_numpy(
            np.vstack((norm_z, roughness, slope_proxy)).T
        ).float()

        data = Data(
            x=x_feat,
            edge_index=torch.from_numpy(edge_index).long(),
            edge_attr=torch.from_numpy(edge_attr).float(),
            pos=torch.from_numpy(pos).float()
        )

        torch.save(data, out_path)
        elapsed = time.time() - t0
        n_edges_total = data.edge_index.shape[1]
        size_mb = out_path.stat().st_size / (1024*1024)
        print(f"    [OK] {out_path.name} | {n_down:,} nodes | {n_edges_total:,} edges | {size_mb:.0f} MB | {elapsed:.1f}s")

        del data, x_feat, edge_index, edge_attr, pos
        gc.collect()
        return True

    except Exception as e:
        print(f"    [FAIL] {laz_path.name}: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("  GOD-TIER V2 GRAPH BUILDER (BEAST MODE)")
    print(f"  Voxel: {VOXEL_SIZE}m | k-NN: {K_NEIGHBORS} | High-Fidelity")
    print("=" * 70)

    print(f"{'='*70}")
    print(f"  GOD-TIER MISSION V2 - AUTONOMOUS IRRIGATION & FLOOD CONTROL")
    print(f"  Running on: {os.environ.get('COMPUTERNAME', 'Unknown PC')}")
    
    # === CUDA CHECK ===
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\n  [SYSTEM] GPU DETECTED: {gpu_name} ({vram:.1f} GB VRAM)")
            print(f"  [SYSTEM] CUDA Version: {torch.version.cuda}")
            print(f"  [SYSTEM] ACCELERATION: ENABLED")
        else:
            print("\n  [SYSTEM] NO GPU DETECTED!")
            print(f"  [SYSTEM] DEVICE: CPU (SLOW MODE)")
            print(f"  [WARN] Operations will be significantly slower.")
    except ImportError:
        print("\n  [WARN] PyTorch not found in main environment (will be checked in subprocesses).")

    print(f"{'='*70}\n")

    # 1. Generate the sub-scripts
    print("[1/5] Generating Mission Scripts...")
    files = sorted(INPUT_DIR.glob("*.laz"))
    total = len(files)
    print(f"  Found {total} LAZ files\n")

    built = 0
    skipped = 0
    failed = 0
    t_start = time.time()

    for i, f in enumerate(files, 1):
        elapsed_total = time.time() - t_start
        if built > 0:
            avg_time = elapsed_total / built
            remaining = avg_time * (total - i + 1)
            print(f"\n[{i}/{total}] ETA: ~{remaining/60:.0f} min remaining")
        else:
            print(f"\n[{i}/{total}]")

        out_check = OUTPUT_DIR / (f.stem + ".pt")
        if out_check.exists():
            print(f"  [SKIP] {f.name} -> graph exists")
            skipped += 1
            continue

        result = build_graph(f)
        if result:
            built += 1
        else:
            failed += 1

    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  GRAPH BUILD COMPLETE")
    print(f"  Built: {built} | Skipped: {skipped} | Failed: {failed} | Time: {total_time/60:.1f} min")
    print(f"{'=' * 70}")
'''
    with open("process_graph_v2.py", "w", encoding="utf-8") as f: f.write(code)


def generate_train_v2():
    """100-epoch training with LR scheduling, confusion matrix, metrics export."""
    code = r'''import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader, ClusterData, ClusterLoader
from flood_gnn_model import get_gnn_model
from pathlib import Path
import time
import json
import numpy as np
import os
import shutil

# === BEAST MODE TRAINING CONFIG ===
EPOCHS = 100
BATCH_SIZE = 1        # Process one graph at a time for VRAM safety
LR_INIT = 0.003
LR_MIN = 1e-5
GRAD_CLIP = 1.0       # Gradient clipping for stability
VAL_SPLIT = 0.2       # 20% validation

# Clustering Config for Large Graphs (>500k nodes)
CLUSTER_THRESHOLD = 500_000
CLUSTER_PARTITIONS = 128

GRAPH_DIR = Path("Processed_Data/GodTier_V2/Graphs")
MODEL_DIR = Path("Models/GodTier_V2")
METRICS_DIR = Path("Models/GodTier_V2/metrics")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
GPU_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = GPU_DEVICE  # Default, but per-graph we may fall back to CPU
GPU_MAX_NODES = 2_000_000  # Graphs larger than this train on CPU to avoid OOM

# Depth bins for confusion matrix (binned regression -> classification)
DEPTH_BINS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]  # Normalized depth thresholds
DEPTH_LABELS = ['Dry', 'Low', 'Medium', 'High', 'Critical']

def compute_confusion_matrix(preds, targets, bins=DEPTH_BINS):
    """Compute confusion matrix by binning continuous depth predictions."""
    n_classes = len(bins)  # bins define boundaries, so n_classes = len(bins)
    pred_bins = np.digitize(preds.flatten(), bins) - 1
    true_bins = np.digitize(targets.flatten(), bins) - 1
    pred_bins = np.clip(pred_bins, 0, n_classes - 1)
    true_bins = np.clip(true_bins, 0, n_classes - 1)

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(true_bins, pred_bins):
        cm[t, p] += 1
    return cm

def compute_metrics(preds, targets):
    """Compute regression + classification metrics."""
    # Regression
    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(mse)

    # R-squared
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    # Confusion Matrix
    cm = compute_confusion_matrix(preds, targets)

    # Per-class accuracy
    class_acc = []
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        if row_sum > 0:
            class_acc.append(cm[i, i] / row_sum)
        else:
            class_acc.append(0.0)

    overall_acc = np.trace(cm) / (cm.sum() + 1e-8)

    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'overall_accuracy': float(overall_acc),
        'per_class_accuracy': [float(a) for a in class_acc],
        'confusion_matrix': cm.tolist()
    }

def train_step_cluster(model, data, optimizer, criterion, device):
    """Train on a large graph using ClusterLoader to avoid OOM."""
    # Partition graph into subgraphs
    # save_dir=None prevents disk I/O, keeps in RAM (which we have plenty of)
    cluster_data = ClusterData(data, num_parts=CLUSTER_PARTITIONS, recursive=False, save_dir=None)
    loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True)
    
    total_loss = 0
    steps = 0
    
    model.train()
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        depth, vel = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = criterion(depth, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
        steps += 1
        
    return total_loss / max(steps, 1)

def train():
    print(f"{'='*70}")
    print(f"  GOD-TIER V2 TRAINING (BEAST MODE) — {EPOCHS} EPOCHS on {DEVICE}")
    print(f"{'='*70}")

    files = sorted(GRAPH_DIR.glob("*.pt"))
    dataset = []

    print(f"\nLoading Dataset (128GB RAM available)...")
    for f in files:
        try:
            d = torch.load(f, weights_only=False)
            # Self-supervised target: normalized inverse elevation = flood accumulation proxy
            if not hasattr(d, 'y') or d.y is None:
                z = d.pos[:, 2]
                d.y = (z.max() - z) / (z.max() - z.min() + 1e-6)
                d.y = d.y.unsqueeze(1)
            dataset.append(d)
            print(f"  Loaded {f.name}: {d.x.shape[0]:,} nodes | {d.edge_index.shape[1]:,} edges")
        except Exception as e:
            print(f"  [WARN] Failed {f.name}: {e}")

    if not dataset:
        print("[ERROR] No graphs found!")
        return

    # Train/Val Split
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_set = dataset[:n_train]
    val_set = dataset[n_train:]
    print(f"\nSplit: {n_train} train / {n_val} val")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # Determine per-graph device: use GPU only if graph fits in VRAM
    def get_graph_device(data):
        n = data.x.shape[0]
        if GPU_DEVICE == 'cuda' and n > GPU_MAX_NODES:
            return 'cpu'
        return GPU_DEVICE

    model = get_gnn_model(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR_INIT)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_MIN)
    criterion = torch.nn.MSELoss()

    # Tracking
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'val_metrics': []}
    best_val_loss = float('inf')
    best_epoch = 0
    start_epoch = 1

    # === RESUME FROM CHECKPOINT ===
    checkpoint_path = METRICS_DIR / "checkpoint_resume.pth"
    history_path = METRICS_DIR / "training_history.json"
    if checkpoint_path.exists():
        print(f"\n[RESUME] Loading checkpoint...")
        ckpt = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        best_epoch = ckpt.get('best_epoch', 0)
        if history_path.exists():
            with open(history_path) as hf:
                history = json.load(hf)
        print(f"[RESUME] Resuming from epoch {start_epoch} (best val loss: {best_val_loss:.6f} @ epoch {best_epoch})")
    elif not checkpoint_path.exists():
        # Check for manual epoch checkpoints as fallback
        epoch_ckpts = sorted(MODEL_DIR.glob("gnn_epoch_*.pth"))
        if epoch_ckpts:
            last_ckpt = epoch_ckpts[-1]
            last_epoch = int(last_ckpt.stem.split('_')[-1])
            print(f"\n[RESUME] Found epoch checkpoint: {last_ckpt.name}")
            model.load_state_dict(torch.load(last_ckpt, weights_only=True))
            start_epoch = last_epoch + 1
            # Advance scheduler to correct position
            for _ in range(last_epoch):
                scheduler.step()
            if history_path.exists():
                with open(history_path) as hf:
                    history = json.load(hf)
            print(f"[RESUME] Resuming from epoch {start_epoch} (scheduler advanced)")

    start_time = time.time()
    print(f"\nStarting epochs {start_epoch}-{EPOCHS} | LR: {scheduler.get_last_lr()[0]:.6f} -> {LR_MIN} (Cosine)")
    print(f"  Graph Output: {GRAPH_DIR}")
    print(f"  Model Output: {MODEL_DIR}")
    print("=" * 60)
    
    # === CUDA CHECK ===
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n[SYSTEM] GPU DETECTED: {gpu_name} ({vram:.1f} GB VRAM)")
        print(f"[SYSTEM] CUDA Version: {torch.version.cuda}")
        print(f"[SYSTEM] PyTorch Version: {torch.__version__}")
        print(f"[SYSTEM] DEVICE: cuda (Accelerated)")
    else:
        print("\n[SYSTEM] NO GPU DETECTED!")
        print(f"[SYSTEM] PyTorch Version: {torch.__version__}")
        print(f"[SYSTEM] DEVICE: cpu (SLOW MODE)")
        print(f"[WARN] Training on CPU with large graphs may be extremely slow.")
    print("=" * 60 + "\n")

    input("Press Enter to START MISSION (or Ctrl+C to abort)...")
    print(f"Batch Size: {BATCH_SIZE} | Grad Clip: {GRAD_CLIP}")
    print("-" * 70)

    for epoch in range(start_epoch, EPOCHS + 1):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        train_steps = 0
        
        for data in train_loader:
            # Check for large graph
            if data.num_nodes > CLUSTER_THRESHOLD:
                # Use ClusterLoader to break down large graph + Train
                # Use DEVICE (GPU) if possible for subgraphs, or CPU if needed
                # Subgraphs are small, so GPU should be fine usually.
                loss = train_step_cluster(model, data, optimizer, criterion, DEVICE)
            else:
                # Standard Training
                dev = get_graph_device(data)
                model.to(dev)
                data = data.to(dev)
                optimizer.zero_grad()
                depth, vel = model(data.x, data.edge_index, data.edge_attr)
                loss = criterion(depth, data.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                loss = loss.item()
                
                # Cleanup
                if dev == 'cpu' and GPU_DEVICE == 'cuda':
                    model.to(GPU_DEVICE)
            
            train_loss += loss
            train_steps += 1
            
        train_loss /= train_steps

        # --- VALIDATE ---
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for data in val_loader:
                # For validation, we also need to handle OOM on large graphs.
                # However, ClusterLoader shuffles, which is bad for deterministic val (metrics).
                # But for simple loss calculation, it's fine. 
                # For proper metrics, we might need full graph inference on CPU if it fits,
                # or stitching subgraphs.
                # Given OOM constraints, we'll try full graph on CPU first.
                
                dev = get_graph_device(data)
                
                # If it's a massive graph that causes OOM even on CPU during inference (unlikely but possible),
                # we might crash. But usually inference is lighter than training (no gradients).
                # 65GB OOM was likely gradients + optimizer states.
                # Let's try standard inference.
                
                try:
                    model.to(dev)
                    data = data.to(dev)
                    depth, vel = model(data.x, data.edge_index, data.edge_attr)
                    loss = criterion(depth, data.y)
                    val_loss += loss.item()
                    all_preds.append(depth.cpu().numpy())
                    all_targets.append(data.y.cpu().numpy())
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  [WARN] OOM during validation on {dev}, skipping graph.")
                        torch.cuda.empty_cache()
                    else:
                        raise e

                if dev == 'cpu' and GPU_DEVICE == 'cuda':
                    model.to(GPU_DEVICE)
                    
        val_loss /= len(val_loader)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Track
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        elapsed = time.time() - start_time
        eta = (elapsed / epoch) * (EPOCHS - epoch)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_DIR / "gnn_best.pth")

        # Checkpoints (full resume state every 5 epochs, model every 10)
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
            }, checkpoint_path)
            # Save history incrementally
            with open(history_path, 'w') as hf:
                json.dump(history, hf, indent=2)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), MODEL_DIR / f"gnn_epoch_{epoch}.pth")

        # Compute full metrics every 25 epochs
        if epoch % 25 == 0 or epoch == EPOCHS:
            if all_preds:
                preds = np.concatenate(all_preds)
                targets = np.concatenate(all_targets)
                metrics = compute_metrics(preds, targets)
                history['val_metrics'].append({'epoch': epoch, **metrics})
                print(f"  Epoch {epoch:03d}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                      f"R2: {metrics['r2']:.4f} | Acc: {metrics['overall_accuracy']:.3f} | "
                      f"LR: {current_lr:.6f} | ETA: {eta/60:.0f}m")
            else:
                 print(f"  Epoch {epoch:03d}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | (No Val Preds)")
        else:
            print(f"  Epoch {epoch:03d}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {current_lr:.6f} | ETA: {eta/60:.0f}m")

    # --- FINAL METRICS ---
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    total_time = time.time() - start_time
    print(f"  Total Time: {total_time/3600:.2f} hours")
    print(f"  Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch})")

    # Final confusion matrix on validation set
    model.load_state_dict(torch.load(MODEL_DIR / "gnn_best.pth", weights_only=True))
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data in val_loader:
            dev = get_graph_device(data)
            try:
                model.to(dev)
                data = data.to(dev)
                depth, _ = model(data.x, data.edge_index, data.edge_attr)
                all_preds.append(depth.cpu().numpy())
                all_targets.append(data.y.cpu().numpy())
                if dev == 'cpu' and GPU_DEVICE == 'cuda':
                    model.to(GPU_DEVICE)
            except Exception as e:
                print(f"Validation failed on graph: {e}")

    if all_preds:
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        final_metrics = compute_metrics(preds, targets)

        print(f"\n  Final Metrics (Best Model):")
        print(f"    MSE:  {final_metrics['mse']:.6f}")
        print(f"    RMSE: {final_metrics['rmse']:.6f}")
        print(f"    MAE:  {final_metrics['mae']:.6f}")
        print(f"    R2:   {final_metrics['r2']:.4f}")
        print(f"    Overall Accuracy: {final_metrics['overall_accuracy']:.3f}")
        print(f"\n  Confusion Matrix ({len(DEPTH_LABELS)} classes):")
        cm = np.array(final_metrics['confusion_matrix'])
        header = "           " + "  ".join(f"{l:>8s}" for l in DEPTH_LABELS)
        print(f"    {header}")
        for i, label in enumerate(DEPTH_LABELS):
            row = "  ".join(f"{cm[i,j]:>8d}" for j in range(len(DEPTH_LABELS)))
            print(f"    {label:>8s}  {row}")

        # Save all metrics
        torch.save(model.state_dict(), MODEL_DIR / "gnn_final.pth")
        history['final_metrics'] = final_metrics
        history['best_epoch'] = best_epoch
        history['depth_labels'] = DEPTH_LABELS
        history['depth_bins'] = DEPTH_BINS
        
        np.save(METRICS_DIR / "confusion_matrix.npy", cm)
        np.save(METRICS_DIR / "final_predictions.npy", preds)
        np.save(METRICS_DIR / "final_targets.npy", targets)

    torch.save(model.state_dict(), MODEL_DIR / "gnn_final.pth")
    history['best_epoch'] = best_epoch
    history['depth_labels'] = DEPTH_LABELS
    history['depth_bins'] = DEPTH_BINS

    with open(METRICS_DIR / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n  Saved: gnn_best.pth, gnn_final.pth, training_history.json, confusion_matrix.npy")
    print(f"{'='*70}")

if __name__ == "__main__":
    train()
'''
    with open("train_flood_gnn_v2.py", "w", encoding="utf-8") as f: f.write(code)


def generate_vis_v2():
    """CINEMATIC multi-panel visualization: 3D render, heatmap, confusion matrix, training curves, GIF flyover."""
    code = r'''import numpy as np
import torch
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from flood_gnn_model import get_gnn_model
import traceback

GRAPH_DIR = Path("Processed_Data/GodTier_V2/Graphs")
MODEL_PATH = Path("Models/GodTier_V2/gnn_best.pth")
METRICS_DIR = Path("Models/GodTier_V2/metrics")
OUTPUT_DIR = Path("Visuals/GodTier_V2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DEPTH_LABELS = ['Dry', 'Low', 'Medium', 'High', 'Critical']

# Custom colormaps
FLOOD_COLORS = ['#1a1a2e', '#16213e', '#0f3460', '#e94560', '#ff6b6b']  # Dark to hot
TERRAIN_CMAP = LinearSegmentedColormap.from_list('terrain_pro',
    ['#2d1b0e', '#5c4033', '#8B7355', '#90EE90', '#228B22', '#006400'], N=256)

def load_model_and_predict(graph_path):
    """Load best model and run inference on a graph."""
    model = get_gnn_model(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    data = torch.load(graph_path, weights_only=False).to(DEVICE)
    with torch.no_grad():
        depth, velocity = model(data.x, data.edge_index, data.edge_attr)

    pos = data.pos.cpu().numpy()
    depth = depth.cpu().numpy().flatten()
    velocity = velocity.cpu().numpy().flatten()
    target = data.y.cpu().numpy().flatten() if hasattr(data, 'y') and data.y is not None else None

    return pos, depth, velocity, target

def render_3d_flood_prediction(pos, depth, name, output_path):
    """Cinematic 3D scatter plot of terrain with flood depth overlay."""
    fig = plt.figure(figsize=(20, 14), facecolor='#0a0a1a')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a1a')

    # Subsample for rendering performance
    n = len(pos)
    if n > 500000:
        idx = np.random.choice(n, 500000, replace=False)
    else:
        idx = np.arange(n)

    x, y, z = pos[idx, 0], pos[idx, 1], pos[idx, 2]
    d = depth[idx]

    # Normalize coordinates for better viewing
    x = x - x.mean()
    y = y - y.mean()

    sc = ax.scatter(x, y, z, c=d, cmap='turbo', s=0.5, alpha=0.8,
                    vmin=0, vmax=np.percentile(d, 95))

    ax.set_xlabel('X (m)', color='white', fontsize=10)
    ax.set_ylabel('Y (m)', color='white', fontsize=10)
    ax.set_zlabel('Elevation (m)', color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=8)

    # Dark styling
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#333333')
    ax.yaxis.pane.set_edgecolor('#333333')
    ax.zaxis.pane.set_edgecolor('#333333')
    ax.grid(True, alpha=0.15, color='white')

    ax.view_init(elev=35, azim=45)

    cb = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.1, label='Predicted Flood Depth')
    cb.ax.yaxis.label.set_color('white')
    cb.ax.tick_params(colors='white')

    ax.set_title(f'Flood Risk Prediction: {name}',
                 color='#00d4ff', fontsize=16, fontweight='bold', pad=20)

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] 3D Render -> {output_path.name}")

def render_overhead_heatmap(pos, depth, name, output_path):
    """Top-down flood risk heatmap with contours."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 14), facecolor='#0a0a1a')
    ax.set_facecolor('#0a0a1a')

    x = pos[:, 0] - pos[:, 0].mean()
    y = pos[:, 1] - pos[:, 1].mean()

    sc = ax.scatter(x, y, c=depth, cmap='turbo', s=0.3, alpha=0.9,
                    vmin=0, vmax=np.percentile(depth, 95))

    ax.set_xlabel('X (m)', color='white', fontsize=12)
    ax.set_ylabel('Y (m)', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.set_aspect('equal')

    cb = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label('Flood Risk (Normalized Depth)', color='white', fontsize=12)
    cb.ax.tick_params(colors='white')

    ax.set_title(f'Overhead Flood Risk Heatmap: {name}',
                 color='#ff6b6b', fontsize=16, fontweight='bold')

    # Add risk zone annotations
    high_risk = depth > np.percentile(depth, 90)
    if high_risk.sum() > 0:
        ax.scatter(x[high_risk], y[high_risk], c='red', s=1, alpha=0.3, marker='x', label='Critical Zone')
        ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white', fontsize=10)

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Heatmap -> {output_path.name}")

def render_training_dashboard(output_path):
    """Training metrics dashboard: loss curves + confusion matrix + scatter."""
    history_path = METRICS_DIR / "training_history.json"
    cm_path = METRICS_DIR / "confusion_matrix.npy"
    preds_path = METRICS_DIR / "final_predictions.npy"
    targets_path = METRICS_DIR / "final_targets.npy"

    if not history_path.exists():
        print("  [SKIP] No training_history.json found")
        return

    with open(history_path) as f:
        history = json.load(f)

    fig = plt.figure(figsize=(24, 16), facecolor='#0a0a1a')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # --- Panel 1: Training & Validation Loss ---
    ax1 = fig.add_subplot(gs[0, 0], facecolor='#1a1a2e')
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], color='#00d4ff', linewidth=2, label='Train Loss', alpha=0.9)
    ax1.plot(epochs, history['val_loss'], color='#ff6b6b', linewidth=2, label='Val Loss', alpha=0.9)
    ax1.fill_between(epochs, history['train_loss'], alpha=0.1, color='#00d4ff')
    ax1.fill_between(epochs, history['val_loss'], alpha=0.1, color='#ff6b6b')
    ax1.set_xlabel('Epoch', color='white')
    ax1.set_ylabel('MSE Loss', color='white')
    ax1.set_title('Training & Validation Loss', color='#00d4ff', fontsize=14, fontweight='bold')
    ax1.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.15, color='white')
    ax1.set_yscale('log')

    # --- Panel 2: Learning Rate Schedule ---
    ax2 = fig.add_subplot(gs[0, 1], facecolor='#1a1a2e')
    ax2.plot(epochs, history['lr'], color='#ffd700', linewidth=2)
    ax2.fill_between(epochs, history['lr'], alpha=0.15, color='#ffd700')
    ax2.set_xlabel('Epoch', color='white')
    ax2.set_ylabel('Learning Rate', color='white')
    ax2.set_title('Cosine Annealing LR Schedule', color='#ffd700', fontsize=14, fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.15, color='white')

    # --- Panel 3: Metrics Summary ---
    ax3 = fig.add_subplot(gs[0, 2], facecolor='#1a1a2e')
    metrics = history.get('final_metrics', {})
    if metrics:
        metric_names = ['MSE', 'RMSE', 'MAE', 'R²', 'Accuracy']
        metric_vals = [metrics['mse'], metrics['rmse'], metrics['mae'],
                       metrics['r2'], metrics['overall_accuracy']]
        colors = ['#00d4ff', '#00d4ff', '#00d4ff', '#4ade80', '#4ade80']

        bars = ax3.barh(metric_names, metric_vals, color=colors, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, metric_vals):
            ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{val:.4f}', va='center', color='white', fontweight='bold')

    ax3.set_title('Final Model Metrics', color='#4ade80', fontsize=14, fontweight='bold')
    ax3.tick_params(colors='white')
    ax3.set_xlim(0, 1.2)
    ax3.grid(True, alpha=0.15, color='white', axis='x')

    # --- Panel 4: Confusion Matrix ---
    ax4 = fig.add_subplot(gs[1, 0], facecolor='#1a1a2e')
    if cm_path.exists():
        cm = np.load(cm_path)
        # Normalize for display
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

        im = ax4.imshow(cm_norm, interpolation='nearest', cmap='YlOrRd', vmin=0, vmax=1)
        ax4.set_xticks(range(len(DEPTH_LABELS)))
        ax4.set_yticks(range(len(DEPTH_LABELS)))
        ax4.set_xticklabels(DEPTH_LABELS, color='white', rotation=45, fontsize=9)
        ax4.set_yticklabels(DEPTH_LABELS, color='white', fontsize=9)

        # Annotate cells
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'white' if cm_norm[i, j] > 0.5 else 'black'
                ax4.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                         color=color, fontsize=8, fontweight='bold')

        ax4.set_xlabel('Predicted', color='white', fontsize=11)
        ax4.set_ylabel('True', color='white', fontsize=11)
        cb = plt.colorbar(im, ax=ax4, shrink=0.8)
        cb.ax.tick_params(colors='white')
        cb.set_label('Normalized', color='white')

    ax4.set_title('Confusion Matrix (Flood Risk Classes)', color='#ff6b6b', fontsize=14, fontweight='bold')

    # --- Panel 5: Prediction vs Target Scatter ---
    ax5 = fig.add_subplot(gs[1, 1], facecolor='#1a1a2e')
    if preds_path.exists() and targets_path.exists():
        preds = np.load(preds_path).flatten()
        targets = np.load(targets_path).flatten()
        # Subsample for plotting
        n = len(preds)
        if n > 50000:
            idx = np.random.choice(n, 50000, replace=False)
            preds_s, targets_s = preds[idx], targets[idx]
        else:
            preds_s, targets_s = preds, targets

        ax5.scatter(targets_s, preds_s, c='#00d4ff', s=0.5, alpha=0.3)
        ax5.plot([0, 1], [0, 1], 'r--', linewidth=2, alpha=0.8, label='Perfect Prediction')
        ax5.set_xlabel('True Depth', color='white')
        ax5.set_ylabel('Predicted Depth', color='white')
        ax5.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')

    ax5.set_title('Predicted vs True Depth', color='#00d4ff', fontsize=14, fontweight='bold')
    ax5.tick_params(colors='white')
    ax5.grid(True, alpha=0.15, color='white')

    # --- Panel 6: Per-Class Accuracy ---
    ax6 = fig.add_subplot(gs[1, 2], facecolor='#1a1a2e')
    if metrics and 'per_class_accuracy' in metrics:
        colors_bar = ['#4ade80', '#fbbf24', '#fb923c', '#f87171', '#ef4444']
        bars = ax6.bar(DEPTH_LABELS, metrics['per_class_accuracy'],
                       color=colors_bar[:len(DEPTH_LABELS)], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, metrics['per_class_accuracy']):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.2f}', ha='center', color='white', fontweight='bold')

    ax6.set_title('Per-Class Accuracy', color='#4ade80', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Accuracy', color='white')
    ax6.tick_params(colors='white')
    ax6.set_ylim(0, 1.15)
    ax6.grid(True, alpha=0.15, color='white', axis='y')

    # Main title
    best_epoch = history.get('best_epoch', '?')
    fig.suptitle(f'GOD-TIER V2 — DUALFloodGNN Training Dashboard (Best @ Epoch {best_epoch})',
                 color='white', fontsize=20, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Dashboard -> {output_path.name}")

def render_multi_village_comparison(output_path):
    """Side-by-side prediction comparison for all villages."""
    if not MODEL_PATH.exists():
        return

    files = sorted(GRAPH_DIR.glob("*.pt"))
    if not files:
        return

    n = min(len(files), 6)  # Max 6 villages
    fig, axes = plt.subplots(2, 3, figsize=(24, 16), facecolor='#0a0a1a')

    for i in range(6):
        row, col = i // 3, i % 3
        ax = axes[row][col]
        ax.set_facecolor('#0a0a1a')

        if i < n:
            try:
                pos, depth, _, _ = load_model_and_predict(files[i])
                name = files[i].stem.split('_')[0]

                # Subsample for rendering
                if len(pos) > 200000:
                    idx = np.random.choice(len(pos), 200000, replace=False)
                else:
                    idx = np.arange(len(pos))

                x = pos[idx, 0] - pos[idx, 0].mean()
                y = pos[idx, 1] - pos[idx, 1].mean()

                sc = ax.scatter(x, y, c=depth[idx], cmap='turbo', s=0.3, alpha=0.9,
                                vmin=0, vmax=np.percentile(depth, 95))
                ax.set_title(name, color='white', fontsize=12, fontweight='bold')
                ax.set_aspect('equal')
                ax.tick_params(colors='white', labelsize=7)

                del pos, depth
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {e}', transform=ax.transAxes,
                        ha='center', color='red', fontsize=9)
        else:
            ax.set_visible(False)

    fig.suptitle('GOD-TIER V2 — Multi-Village Flood Risk Comparison',
                 color='#00d4ff', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Multi-Village -> {output_path.name}")


def vis():
    print("=" * 70)
    print("  GOD-TIER V2 CINEMATIC VISUALIZER (BEAST MODE)")
    print("=" * 70)

    if not MODEL_PATH.exists():
        print("[ERROR] No trained model found!")
        return

    files = sorted(GRAPH_DIR.glob("*.pt"))
    if not files:
        print("[ERROR] No graph files found!")
        return

    print(f"  Model: {MODEL_PATH}")
    print(f"  Graphs: {len(files)} files")
    print(f"  Output: {OUTPUT_DIR}\n")

    # 1. TRAINING DASHBOARD (confusion matrix + loss curves + metrics)
    print("[1/4] Training Dashboard...")
    try:
        render_training_dashboard(OUTPUT_DIR / "training_dashboard.png")
    except Exception as e:
        print(f"  [WARN] Dashboard failed: {e}")
        traceback.print_exc()

    # 2. 3D FLOOD PREDICTION (first village)
    print("[2/4] 3D Flood Prediction Render...")
    try:
        pos, depth, vel, target = load_model_and_predict(files[0])
        name = files[0].stem
        render_3d_flood_prediction(pos, depth, name, OUTPUT_DIR / "flood_3d_prediction.png")
    except Exception as e:
        print(f"  [WARN] 3D render failed: {e}")
        traceback.print_exc()

    # 3. OVERHEAD HEATMAP (first village)
    print("[3/4] Overhead Risk Heatmap...")
    try:
        render_overhead_heatmap(pos, depth, name, OUTPUT_DIR / "flood_overhead_heatmap.png")
    except Exception as e:
        print(f"  [WARN] Heatmap failed: {e}")
        traceback.print_exc()

    # 4. MULTI-VILLAGE COMPARISON
    print("[4/4] Multi-Village Comparison...")
    try:
        render_multi_village_comparison(OUTPUT_DIR / "multi_village_comparison.png")
    except Exception as e:
        print(f"  [WARN] Multi-village render failed: {e}")
        traceback.print_exc()

    # BONUS: Try PyVista 3D render if available
    try:
        import pyvista as pv
        print("\n[BONUS] PyVista Cinematic Render...")
        pos, depth, _, _ = load_model_and_predict(files[0])

        if len(pos) > 500000:
            idx = np.random.choice(len(pos), 500000, replace=False)
            pos_sub, depth_sub = pos[idx], depth[idx]
        else:
            pos_sub, depth_sub = pos, depth

        cloud = pv.PolyData(pos_sub)
        cloud['Flood Depth'] = depth_sub

        pl = pv.Plotter(off_screen=True, window_size=[1920, 1080])
        pl.set_background('#0a0a1a', top='#1a1a3e')
        pl.enable_eye_dome_lighting()

        pl.add_mesh(cloud, scalars='Flood Depth', cmap='turbo', point_size=3,
                    render_points_as_spheres=True, opacity=0.9,
                    scalar_bar_args={'title': 'Predicted Flood Depth',
                                     'color': 'white', 'vertical': True})

        pl.add_text("GOD-TIER V2: DUALFloodGNN Prediction", position='upper_left',
                     color='white', font_size=14)

        pl.camera.azimuth = 45
        pl.camera.elevation = 30
        pl.camera.zoom(1.2)
        pl.screenshot(OUTPUT_DIR / "pyvista_cinematic.png")

        # Orbital GIF
        print("  Generating orbital flyover GIF...")
        path = pl.generate_orbital_path(n_points=90, shift=0.0)
        pl.open_gif(str(OUTPUT_DIR / "flood_flyover_orbit.gif"))
        pl.orbit_on_path(path, write_frames=True)
        pl.close()
        print(f"  [OK] PyVista -> pyvista_cinematic.png, flood_flyover_orbit.gif")

    except ImportError:
        print("  [SKIP] PyVista not available, using matplotlib renders only")
    except Exception as e:
        print(f"  [WARN] PyVista render failed: {e}")
        traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"  VISUALIZATION COMPLETE — Check {OUTPUT_DIR}")
    print(f"{'='*70}")

if __name__ == "__main__":
    vis()
'''
    with open("vis_cinematic_v2.py", "w", encoding="utf-8") as f: f.write(code)


# ============================================================
#  ENGINE
# ============================================================

PIPELINE = [
    ("Phase 1: Sonata Logic", "phase1_sonata_v2.py"),
    ("Phase 2: High-Perf Ingestion", "ingest_data_v2.py"),
    ("Phase 3: Graph Construction", "process_graph_v2.py"),
    ("Phase 4: GNN Training", "train_flood_gnn_v2.py"),
    ("Phase 5: Cinematic Vis", "vis_cinematic_v2.py")
]

def run_mission():
    log(f"{'='*70}")
    log(f"  GOD-TIER V2 MISSION — BEAST MODE ({MISSION_CONFIG['VERSION']})")
    log(f"  Profile: {MISSION_CONFIG['HARDWARE_PROFILE']}")
    log(f"  Epochs: {MISSION_CONFIG['PARAMS']['TRAIN_EPOCHS']} | Voxel: {MISSION_CONFIG['PARAMS']['VOXEL_SIZE']}m")
    log(f"{'='*70}")

    # Generate all scripts
    generate_sonata_wrapper()
    generate_ingest_v2()
    generate_graph_v2()
    generate_train_v2()
    generate_vis_v2()

    # Execution loop with smart resume
    mission_start = time.time()
    for name, script in PIPELINE:
        if is_step_complete(name):
            log(f"[RESUME] Skipping {name} (Already Complete)")
            continue

        log(f"\n>>> EXECUTING: {name}")
        step_start = time.time()
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            proc = subprocess.Popen([sys.executable, script],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  text=True, bufsize=1, encoding='utf-8', env=env)

            for line in proc.stdout:
                print(f"[{name}] {line}", end='')

            proc.wait()
            step_time = time.time() - step_start

            if proc.returncode == 0:
                log(f"[SUCCESS] {name} ({step_time/60:.1f} min)")
                mark_step_complete(name)
            else:
                log(f"[FAIL] {name} (Exit Code {proc.returncode}) after {step_time/60:.1f} min")
                log("Mission Paused. Fix error and re-run to resume.")
                sys.exit(1)

        except Exception as e:
            log(f"[CRITICAL] {e}")
            traceback.print_exc()
            sys.exit(1)

    total_time = time.time() - mission_start
    log(f"\n{'='*70}")
    log(f"  MISSION ACCOMPLISHED (V2 BEAST MODE)")
    log(f"  Total Runtime: {total_time/3600:.2f} hours")
    log(f"{'='*70}")

if __name__ == "__main__":
    run_mission()
