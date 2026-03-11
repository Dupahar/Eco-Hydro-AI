import numpy as np
import laspy
import torch
from pathlib import Path
from torch_geometric.data import Data
from scipy.spatial import cKDTree
import traceback
try: import rasterio
except: rasterio = None

INPUT_DIR = Path("Processed_Data/FullScale")
OUTPUT_DIR = Path("Processed_Data/Graphs_Full")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def build_giant_graph(laz_path):
    out_path = OUTPUT_DIR / (laz_path.stem + ".pt")
    if out_path.exists(): return

    print(f"Graphing: {laz_path.name}")
    try:
        las = laspy.read(laz_path)
        pos = np.vstack((las.x, las.y, las.z)).transpose()
        
        tree = cKDTree(pos)
        dist, idx = tree.query(pos, k=9)
        
        src = np.repeat(np.arange(len(pos)), 9)
        dst = idx.flatten()
        edge_dist = dist.flatten()
        
        mask = (dst < len(pos)) & (edge_dist > 0.01)
        src, dst, edge_dist = src[mask], dst[mask], edge_dist[mask]
        
        edge_index = np.vstack((src, dst))
        edge_attr = edge_dist[:, np.newaxis] 
        edge_attr = np.hstack([edge_attr, edge_attr, edge_attr]) # Pad to 3 dims
        
        x = torch.from_numpy(pos[:, 2:3]).float()
        if x.shape[1] < 3:
            padding = torch.zeros((x.shape[0], 3 - x.shape[1]))
            x = torch.cat([x, padding], dim=1)

        data = Data(x=x, edge_index=torch.from_numpy(edge_index).long(),
                    edge_attr=torch.from_numpy(edge_attr).float(),
                    pos=torch.from_numpy(pos).float())
        
        torch.save(data, out_path)
        print(f"✅ Saved: {out_path.name}")
    except Exception as e:
        print(f"❌ Failed {laz_path.name}:")
        traceback.print_exc()

if __name__ == "__main__":
    files = list(INPUT_DIR.glob("*.laz"))
    for f in files: build_giant_graph(f)
