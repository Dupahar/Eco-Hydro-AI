import torch
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
    global DEVICE
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
        
        # === FORCE CUDA INITIALIZATION ===
        try:
            torch.cuda.init()
            dummy = torch.zeros(1).cuda()
            del dummy
            torch.cuda.empty_cache()
            print("[CUDA] Initialization successful")
        except Exception as e:
            print(f"[ERROR] CUDA init failed: {e}")
            print("[FALLBACK] Switching to CPU mode")
            DEVICE = 'cpu'
            
        print(f"[SYSTEM] CUDA Version: {torch.version.cuda}")
        print(f"[SYSTEM] PyTorch Version: {torch.__version__}")
        print(f"[SYSTEM] DEVICE: {DEVICE} (Accelerated)")
    else:
        print("\n[SYSTEM] NO GPU DETECTED!")
        print(f"[SYSTEM] PyTorch Version: {torch.__version__}")
        print(f"[SYSTEM] DEVICE: cpu (SLOW MODE)")
        print(f"[WARN] Training on CPU with large graphs may be extremely slow.")
    print("=" * 60 + "\n")

    # input("Press Enter to START MISSION (or Ctrl+C to abort)...")
    print("Mission Continuing Automatically...")
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
