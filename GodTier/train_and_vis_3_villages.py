import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from flood_gnn_model import get_gnn_model
from pathlib import Path
import time
import json
import numpy as np
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import traceback
import sys

# --- CONFIGURATION ---
EPOCHS = 100
BATCH_SIZE = 1  # Safe for large graphs
LR_INIT = 0.003
LR_MIN = 1e-5
GRAD_CLIP = 1.0
VAL_SPLIT = 0.2

# Output Configuration
OUTPUT_BASE = Path("Processed_Data/GodTier_V2_Contingency")
MODEL_DIR = OUTPUT_BASE / "Models"
METRICS_DIR = MODEL_DIR / "metrics"
VISUALS_DIR = OUTPUT_BASE / "Visuals"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
VISUALS_DIR.mkdir(parents=True, exist_ok=True)

# Graph Paths (Hardcoded as requested)
GRAPH_PATHS = [
    r"E:\DTM\Data\GodTier\Processed_Data\GodTier_V2\Graphs\118118_Dariyapur_POINT CLOUD.pt",
    r"E:\DTM\Data\GodTier\Processed_Data\GodTier_V2\Graphs\118125_devipura urf Dhanna Nagla_POINT CLOUD.pt",
    r"E:\DTM\Data\GodTier\Processed_Data\GodTier_V2\Graphs\118129_Manjhola Khurd_POINT CLOUD.pt"
]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Metrics Metadata
DEPTH_BINS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
DEPTH_LABELS = ['Dry', 'Low', 'Medium', 'High', 'Critical']

# --- METRIC UTILS ---
def compute_confusion_matrix(preds, targets, bins=DEPTH_BINS):
    n_classes = len(bins)
    pred_bins = np.digitize(preds.flatten(), bins) - 1
    true_bins = np.digitize(targets.flatten(), bins) - 1
    pred_bins = np.clip(pred_bins, 0, n_classes - 1)
    true_bins = np.clip(true_bins, 0, n_classes - 1)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(true_bins, pred_bins):
        cm[t, p] += 1
    return cm

def compute_metrics(preds, targets):
    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(mse)
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    cm = compute_confusion_matrix(preds, targets)
    class_acc = []
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        class_acc.append(cm[i, i] / row_sum if row_sum > 0 else 0.0)
    overall_acc = np.trace(cm) / (cm.sum() + 1e-8)

    return {
        'mse': float(mse), 'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2),
        'overall_accuracy': float(overall_acc),
        'per_class_accuracy': [float(a) for a in class_acc],
        'confusion_matrix': cm.tolist()
    }

# --- TRAINING ---
def train_contingency():
    global DEVICE
    print(f"{'='*70}")
    print(f"  GOD-TIER CONTINGENCY TRAINING (3 VILLAGES) — {EPOCHS} EPOCHS")
    print(f"  Device: {DEVICE}")
    print(f"{'='*70}")

    dataset = []
    print(f"\nLoading Graphs...")
    for p_str in GRAPH_PATHS:
        p = Path(p_str)
        try:
            if not p.exists():
                print(f"  [ERROR] File not found: {p}")
                continue
            d = torch.load(p, weights_only=False)
            if not hasattr(d, 'y') or d.y is None:
                z = d.pos[:, 2]
                d.y = (z.max() - z) / (z.max() - z.min() + 1e-6)
                d.y = d.y.unsqueeze(1)
            dataset.append(d)
            print(f"  Loaded {p.name}: {d.x.shape[0]:,} nodes | {d.edge_index.shape[1]:,} edges")
        except Exception as e:
            print(f"  [FAIL] Failed: {e}")

    if not dataset:
        print("[CRITICAL] No graphs loaded. Exiting.")
        return

    # Split (Standard 20% -> 1 graph for 3 total)
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_set = dataset[:n_train]
    val_set = dataset[n_train:]
    print(f"\nSplit: {n_train} Train / {n_val} Val")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = get_gnn_model(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR_INIT)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_MIN)
    criterion = torch.nn.MSELoss()

    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'val_metrics': []}
    best_val_loss = float('inf')
    best_epoch = 0
    start_time = time.time()

    print(f"\nStarting {EPOCHS} Epochs...")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        # TRAIN
        model.train()
        train_loss = 0
        steps = 0
        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            depth, vel = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(depth, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item()
            steps += 1
        train_loss /= steps

        # VAL
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(DEVICE)
                depth, vel = model(data.x, data.edge_index, data.edge_attr)
                loss = criterion(depth, data.y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        elapsed = time.time() - start_time
        avg_time = elapsed / epoch
        remaining = avg_time * (EPOCHS - epoch)
        
        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_DIR / "gnn_best.pth")

        print(f"Epoch {epoch:03d}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"LR: {current_lr:.6f} | ETA: {remaining/60:.1f}m")

    # Final Save
    total_time = time.time() - start_time
    history['best_epoch'] = best_epoch
    with open(METRICS_DIR / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), MODEL_DIR / "gnn_final.pth")
    
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE (Best Val: {best_val_loss:.6f} @ Epoch {best_epoch})")
    print(f"  Total Time: {total_time/60:.1f} min")
    print(f"{'='*70}\n")


# --- VISUALIZATION ---
def vis_contingency():
    print(f"\n[VISUALIZATION] Generating Cinematic Visuals...")
    MODEL_PATH = MODEL_DIR / "gnn_best.pth"
    if not MODEL_PATH.exists():
        print("[ERROR] Best model not found!")
        return

    model = get_gnn_model(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Load History for Dashboard
    with open(METRICS_DIR / "training_history.json") as f:
        history = json.load(f)

    # 1. Training Dashboard
    fig = plt.figure(figsize=(16, 10), facecolor='#0a0a1a')
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1 = fig.add_subplot(2, 1, 1, facecolor='#1a1a2e')
    ax1.plot(epochs, history['train_loss'], color='#00d4ff', label='Train')
    ax1.plot(epochs, history['val_loss'], color='#ff6b6b', label='Val')
    ax1.set_yscale('log')
    ax1.set_title("Training Loss (Contingency Run)", color='white')
    ax1.legend()
    ax1.tick_params(colors='white')
    ax1.grid(alpha=0.1, color='white')

    ax2 = fig.add_subplot(2, 1, 2, facecolor='#1a1a2e')
    ax2.plot(epochs, history['lr'], color='#ffd700')
    ax2.set_title("Learning Rate", color='white')
    ax2.tick_params(colors='white')
    ax2.grid(alpha=0.1, color='white')
    
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "training_dashboard.png", facecolor='#0a0a1a')
    plt.close()
    print("  [OK] Training Dashboard")

    # 2. Render Each Graph
    for p_str in GRAPH_PATHS:
        p = Path(p_str)
        if not p.exists(): continue
        
        try:
            name = p.stem.split('_')[1] # Extract village name
        except:
            name = p.stem

        print(f"  Rendering {name}...")
        data = torch.load(p).to(DEVICE)
        
        with torch.no_grad():
            depth, vel = model(data.x, data.edge_index, data.edge_attr)
        
        pos = data.pos.cpu().numpy()
        d = depth.cpu().numpy().flatten()
        
        # Subsample for plot
        if len(pos) > 200000:
            idx = np.random.choice(len(pos), 200000, replace=False)
            pos, d = pos[idx], d[idx]
            
        x, y = pos[:, 0], pos[:, 1]
        x = x - x.mean()
        y = y - y.mean()

        # Heatmap
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='#0a0a1a')
        sc = ax.scatter(x, y, c=d, cmap='turbo', s=1, vmin=0, vmax=np.percentile(d, 95))
        ax.set_title(f"Flood Risk Heatmap: {name}", color='white', fontsize=16)
        ax.axis('off')
        plt.colorbar(sc, ax=ax, label='Flood Depth').ax.tick_params(colors='white')
        plt.savefig(VISUALS_DIR / f"heatmap_{name}.png", facecolor='#0a0a1a', bbox_inches='tight')
        plt.close()

        # 3D PyVista (Cinematic)
        try:
            import pyvista as pv
            cloud = pv.PolyData(pos)
            cloud['Flood Depth'] = d
            
            pl = pv.Plotter(off_screen=True, window_size=[1920, 1080])
            pl.set_background('#0a0a1a', top='#1a1a3e')
            pl.enable_eye_dome_lighting()
            pl.add_mesh(cloud, scalars='Flood Depth', cmap='turbo', point_size=2, render_points_as_spheres=True)
            pl.add_text(f"GOD-TIER V2: {name}", position='upper_left', color='white', font_size=14)
            pl.camera.azimuth = 45
            pl.camera.elevation = 30
            pl.screenshot(VISUALS_DIR / f"cinematic_3d_{name}.png")
            
            # Orbital GIF
            print(f"    Generating Orbit GIF for {name}...")
            path = pl.generate_orbital_path(n_points=60, shift=0.0)
            pl.open_gif(str(VISUALS_DIR / f"flyover_{name}.gif"))
            pl.orbit_on_path(path, write_frames=True)
            pl.close()
        except ImportError:
            pass
        except Exception as e:
            print(f"    [WARN] PyVista error: {e}")

    print(f"\n[DONE] Check {VISUALS_DIR}")

if __name__ == "__main__":
    train_contingency()
    vis_contingency()
