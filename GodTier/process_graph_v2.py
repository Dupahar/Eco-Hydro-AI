import numpy as np
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
    # Remove Skip logic to force rebuild and see logs
    # if out_path.exists():
    #    print(f"  [SKIP] {out_path.name} already exists")
    #    return True

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
            print(f"    [DETECTED DEGREES] Ranges X:{x_range:.6f} Y:{y_range:.6f}")
            
            # *** FIX: Translate to local origin FIRST to avoid precision loss ***
            x_min = pos_raw[:, 0].min()
            y_min = pos_raw[:, 1].min()
            pos_raw[:, 0] -= x_min
            pos_raw[:, 1] -= y_min
            print(f"    Centered Data: X/Y shifted by -{x_min:.4f}/-{y_min:.4f}")

            # Approximate conversion: 1 deg lat ~ 111,111 m
            # 1 deg lon ~ 111,111 * cos(lat) m
            mean_lat_rad = np.radians(np.mean(pos_raw[:, 1] + y_min)) # Use original latitude for scaling factor
            scale_x = 111_111 * np.cos(mean_lat_rad)
            scale_y = 111_111
            
            pos_raw[:, 0] *= scale_x
            pos_raw[:, 1] *= scale_y
            
            print(f"    Applied Scale: X*{scale_x:.1f} Y*{scale_y:.1f}")
            print(f"    Scaled X: {pos_raw[:,0].min():.1f} .. {pos_raw[:,0].max():.1f}")
            print(f"    Scaled Y: {pos_raw[:,1].min():.1f} .. {pos_raw[:,1].max():.1f}")
            
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
            # print(f"  [SKIP] {f.name} -> graph exists")
            # skipped += 1
            # continue
            pass

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
