import numpy as np
import laspy
import torch
from pathlib import Path
from torch_geometric.data import Data
from scipy.spatial import cKDTree
import traceback
import time
import sys

# === V2 PATHS (AUTO-DETECT: workstation E: or local) ===
# Check multiple possible input dirs
POSSIBLE_INPUTS = [
    Path("Processed_Data/GodTier_V2/Clouds"),   # V2 mission output
    Path("Processed_Data/UP"),                    # Local machine
    Path("Processed_Data/FullScale"),             # Alternate local
]
INPUT_DIR = None
for p in POSSIBLE_INPUTS:
    if p.exists() and list(p.glob("*.laz")):
        INPUT_DIR = p
        break

if INPUT_DIR is None:
    print("[ERROR] No LAZ input directory found! Checked:")
    for p in POSSIBLE_INPUTS:
        print(f"  {p.resolve()} -> exists={p.exists()}")
    sys.exit(1)

OUTPUT_DIR = Path("Processed_Data/GodTier_V2/Graphs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === HIGH-FIDELITY CONFIG ===
VOXEL_SIZE = 0.25   # 25cm voxel grid (4x more points than 0.5m)
K_NEIGHBORS = 16    # k-NN: bounded edges, predictable memory

def voxel_downsample(pos, voxel_size=0.25):
    """Pure Numpy voxel grid filter. High-fidelity mode retains more points."""
    min_bound = np.min(pos, axis=0)
    voxel_indices = ((pos - min_bound) / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
    return pos[unique_idx]

def build_graph_fast(laz_path):
    """cKDTree-accelerated graph construction with voxel downsampling."""
    out_path = OUTPUT_DIR / (laz_path.stem + ".pt")
    if out_path.exists(): 
        print(f"  [SKIP] {out_path.name} exists")
        return True

    print(f"  [GRAPH] Processing: {laz_path.name}")
    start = time.time()
    
    try:
        las = laspy.read(laz_path)
        # CRITICAL: Use float64 — float32 loses precision at large UTM coords
        pos_raw = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float64)
        n_raw = len(pos_raw)
        del las  # Free memory immediately
        
        # HIGH-FIDELITY: Voxel Downsample at 0.25m in float64 (precision-safe)
        pos_f64 = voxel_downsample(pos_raw, VOXEL_SIZE)
        del pos_raw
        n_down = len(pos_f64)
        print(f"    Downsampled: {n_raw:,} -> {n_down:,} nodes ({100*n_down/n_raw:.1f}%)")
        
        # Convert to float32 for all subsequent ops (saves 50% RAM)
        pos = pos_f64.astype(np.float32)
        del pos_f64
        
        # === k-NN graph via cKDTree (bounded memory) ===
        t_tree = time.time()
        tree = cKDTree(pos)
        dists_knn, indices = tree.query(pos, k=K_NEIGHBORS + 1, workers=-1)
        del tree
        # Remove self-connections
        dists_knn = dists_knn[:, 1:].astype(np.float32)
        indices = indices[:, 1:]
        t_tree_done = time.time()
        
        # Build bidirectional edge index
        N = len(pos)
        src_fwd = np.repeat(np.arange(N), K_NEIGHBORS)
        tgt_fwd = indices.flatten()
        src = np.concatenate([src_fwd, tgt_fwd])
        tgt = np.concatenate([tgt_fwd, src_fwd])
        del src_fwd, tgt_fwd, indices
        
        # Edge attributes in chunks
        CHUNK = 50_000_000
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
        
        # Node Features: [NormZ, 0, 0]
        min_z = pos[:, 2].min()
        norm_z = pos[:, 2] - min_z
        x = torch.from_numpy(np.vstack((norm_z, np.zeros_like(norm_z), np.zeros_like(norm_z))).T).float()

        data = Data(x=x, 
                    edge_index=torch.from_numpy(edge_index).long(),
                    edge_attr=torch.from_numpy(edge_attr).float(),
                    pos=torch.from_numpy(pos).float())
        
        torch.save(data, out_path)
        elapsed = time.time() - start
        print(f"    [OK] {out_path.name} | Nodes: {n_down:,} | Edges: {edge_index.shape[1]:,} | "
              f"Tree: {t_tree_done-t_tree:.1f}s | Total: {elapsed:.1f}s")
        return True
        
    except Exception as e:
        print(f"    [FAIL] {laz_path.name}: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("  GOD-TIER V2 GRAPH BUILDER (HIGH-FIDELITY + k-NN)")
    print("=" * 60)
    print(f"  Voxel: {VOXEL_SIZE}m | k-NN: {K_NEIGHBORS}")
    print(f"  Input:  {INPUT_DIR.resolve()}")
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    
    files = sorted(INPUT_DIR.glob("*.laz"))
    total = len(files)
    print(f"  Found {total} LAZ files\n")
    
    done = 0
    skipped = 0
    failed = 0
    t_start = time.time()
    
    for i, f in enumerate(files, 1):
        elapsed_total = time.time() - t_start
        if done > 0:
            avg_time = elapsed_total / done
            remaining = avg_time * (total - i + 1)
            eta_min = remaining / 60
            print(f"[{i}/{total}] (ETA: ~{eta_min:.0f} min remaining)")
        else:
            print(f"[{i}/{total}]")
        
        result = build_graph_fast(f)
        if result is True:
            out_check = OUTPUT_DIR / (f.stem + ".pt")
            done += 1
        elif result is False:
            failed += 1
        else:
            done += 1
    
    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  COMPLETE | Built: {done} | Failed: {failed} | Time: {total_time/60:.1f} min")
    print(f"{'=' * 60}")

