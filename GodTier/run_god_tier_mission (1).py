import subprocess
import sys
import os
import io

# Force UTF-8 for stdout/stderr to prevent console crashes on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import time
import logging
import traceback
from pathlib import Path

# --- CONFIGURATION ---
PYTHON_EXE = sys.executable 
LOG_FILE = "mission_control.log"
BASE_DIR = Path.cwd()

# Dependency List for Fresh Install (Auto-Check)
DEPENDENCIES = [
    "torch", "torch_geometric", "numpy", "pandas", "laspy", "lazrs", 
    "pyvista", "imageio", "scipy", "tqdm", "geopandas", "shapely", "rasterio"
]

# --- LOGGING SETUP ---
# We config logging to file, but we will mostly write manually to capture subprocess output cleanly
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def log(msg):
    logging.info(msg)

def log_output(line):
    # Log raw output line to file without re-timestamping everything redundantly
    # but ensuring it's saved. Using the root logger's handler 0 (FileHandler)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except: pass

# --- AUTO-GENERATORS ---

def generate_model_file():
    code = r'''import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class DepthMP(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        num_nodes = x.size(0)
        edge_dim = edge_attr.size(1)
        loop_attr = torch.zeros((num_nodes, edge_dim), device=edge_attr.device, dtype=edge_attr.dtype)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return self.mlp(torch.cat([x_j, edge_attr], dim=1))

class VelocityMP(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean') 
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, x_i, edge_attr):
        delta = x_j - x_i
        return self.mlp(torch.cat([delta, edge_attr], dim=1))

class DUALFloodGNN(nn.Module):
    def __init__(self, node_dim=3, edge_dim=3, hidden_dim=64):
        super().__init__()
        self.node_enc = nn.Linear(node_dim, hidden_dim)
        self.depth_update = DepthMP(hidden_dim + edge_dim, hidden_dim)
        self.velocity_update = VelocityMP(hidden_dim + edge_dim, hidden_dim)
        self.depth_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.ReLU())
        self.velocity_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, edge_index, edge_attr, building_mask=None):
        h = F.relu(self.node_enc(x))
        h = h + self.depth_update(h, edge_index, edge_attr)
        h = h + self.velocity_update(h, edge_index, edge_attr)
        next_depth = self.depth_head(h)
        next_velocity = self.velocity_head(h)
        
        if building_mask is not None:
            next_depth = next_depth * (1 - building_mask)
            next_velocity = next_velocity * (1 - building_mask)
            
        return next_depth, next_velocity

def get_gnn_model(device='cuda'):
    model = DUALFloodGNN(node_dim=3) 
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
    return model
'''
    if Path("flood_gnn_model.py").exists(): return
    with open("flood_gnn_model.py", "w", encoding="utf-8") as f:
        f.write(code)
    log("[OK] Generated Fixed Model: flood_gnn_model.py")

def generate_ingest_script():
    code = r'''import laspy
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
            print(f"[SKIP] {las_path.name} (Exists)")
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
        
        print(f"[OK] Ingested {las_path.stem} -> {len(points)} to {np.sum(mask)} pts")
            
    except Exception as e:
        print(f"[ERROR] {las_path.name}: {e}")
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
            print(f"[ERROR] Zip Error {file_path}:")
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
'''
    if Path("ingest_data_full_scale.py").exists(): return
    with open("ingest_data_full_scale.py", "w", encoding="utf-8") as f:
        f.write(code)
    log("[OK] Generated Ingest Script: ingest_data_full_scale.py")

def generate_graph_script():
    code = r'''import numpy as np
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
        print(f"[OK] Saved: {out_path.name}")
    except Exception as e:
        print(f"[FAIL] Failed {laz_path.name}:")
        traceback.print_exc()

if __name__ == "__main__":
    files = list(INPUT_DIR.glob("*.laz"))
    for f in files: build_giant_graph(f)
'''
    if Path("process_graph_full_scale.py").exists(): return
    with open("process_graph_full_scale.py", "w", encoding="utf-8") as f:
        f.write(code)
    log("[OK] Generated Graph Script: process_graph_full_scale.py")

def generate_training_script():
    code = r'''import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from flood_gnn_model import get_gnn_model
import glob
from pathlib import Path
import traceback

EPOCHS = 50
LR = 0.005
GRAPH_DIR = Path("Processed_Data/Graphs_Full")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_god_tier():
    print(f"[START] Training on {DEVICE}...")
    try:
        files = list(GRAPH_DIR.glob("*.pt"))
        dataset = []
        for f in files:
            try:
                d = torch.load(f)
                if not hasattr(d, 'y'):
                    d.y = (d.pos[:, 2].max() - d.pos[:, 2]) / d.pos[:, 2].max()
                    d.y = d.y.unsqueeze(1)
                dataset.append(d)
            except: pass
            
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        model = get_gnn_model(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = torch.nn.MSELoss()
        
        for epoch in range(1, EPOCHS+1):
            model.train()
            total_loss = 0
            for data in loader:
                data = data.to(DEVICE)
                optimizer.zero_grad()
                pred, _ = model(data.x, data.edge_index, data.edge_attr)
                loss = criterion(pred, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f}")
            
        torch.save(model.state_dict(), "flood_gnn_godtier_final.pth")
        print("[OK] Training Complete.")
    except Exception as e:
        print("[FAIL] Training Crash:")
        traceback.print_exc()

if __name__ == "__main__":
    train_god_tier()
'''
    if Path("train_flood_gnn_full.py").exists(): return
    with open("train_flood_gnn_full.py", "w", encoding="utf-8") as f:
        f.write(code)
    log("[OK] Generated Training Script: train_flood_gnn_full.py")

def generate_export_script():
    code = r'''import torch
import pandas as pd
import numpy as np
from pathlib import Path
from flood_gnn_model import get_gnn_model
import traceback

GRAPH_DIR = Path("Processed_Data/Graphs_Full")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def export_predictions():
    print("[START] Exporting Inference Results...")
    try:
        model = get_gnn_model(DEVICE)
        model.load_state_dict(torch.load("flood_gnn_godtier_final.pth"))
        model.eval()
        
        all_pos = []
        all_depth = []
        
        for f in list(GRAPH_DIR.glob("*.pt")):
            try:
                data = torch.load(f).to(DEVICE)
                with torch.no_grad():
                    pred, _ = model(data.x, data.edge_index, data.edge_attr)
                
                all_pos.append(data.pos.cpu().numpy())
                all_depth.append(pred.cpu().numpy())
                print(f"Processed {f.name}")
            except Exception as e: 
                print(f"Error file {f}: {e}")

        if not all_pos: return

        full_pos = np.vstack(all_pos)
        full_depth = np.vstack(all_depth)
        
        df = pd.DataFrame(full_pos, columns=['X', 'Y', 'Z'])
        df['Depth'] = full_depth
        df.to_csv("Niagara_Manifest.csv", index=False)
        print("[OK] Exported Niagara_Manifest.csv")
    except Exception as e:
        print("[FAIL] Export Fail:")
        traceback.print_exc()

if __name__ == "__main__":
    export_predictions()
'''
    if Path("export_predictions_full.py").exists(): return
    with open("export_predictions_full.py", "w", encoding="utf-8") as f:
        f.write(code)
    log("[OK] Generated Export Script: export_predictions_full.py")

def generate_vis_script():
    code = r'''import pyvista as pv
import pandas as pd
import numpy as np
from pathlib import Path
import imageio
import traceback

OUTPUT_IMAGE = "DigitalTwin_GodTier_Heatmap.png"
OUTPUT_GIF = "DigitalTwin_GodTier_RisingFlood.gif"
OUTPUT_SPLIT = "DigitalTwin_GodTier_SplitScreen.gif"

def run_vis():
    print("[START] Rendering Hollywood-Grade Digital Twin...")
    try:
        # Load Data
        df = pd.read_csv("Niagara_Manifest.csv")

        # 1. Create Terrain Mesh (The Village)
        las_files = list(Path("Processed_Data/FullScale").glob("*.laz"))
        terrain_mesh = None
        if las_files:
            try:
                reader = pv.get_reader(str(las_files[0]))
                terrain_mesh = reader.read()
            except: pass

        # 2. Create Flood Mesh (The Water)
        water_df = df[df['Depth'] > 0.1]
        water_points = water_df[['X', 'Y', 'Z']].values
        water_cloud = pv.PolyData(water_points)
        water_cloud['Depth'] = water_df['Depth'].values

        # --- SCENE 1: The High-Res Screenshot (4K) ---
        pl = pv.Plotter(off_screen=True, window_size=[3840, 2160])
        pl.set_background('black', top='midnightblue')
        pl.enable_eye_dome_lighting() 
        
        if terrain_mesh is not None:
            if 'Red' in terrain_mesh.array_names:
                pl.add_mesh(terrain_mesh, rgb=True, point_size=2)
            else:
                pl.add_mesh(terrain_mesh, color='#8B4513', point_size=2)

        pl.add_mesh(water_cloud, scalars='Depth', cmap='turbo', 
                    point_size=5, opacity=0.8, render_points_as_spheres=True, clim=[0, 5])
        
        pl.view_isometric()
        pl.camera.zoom(1.3)
        pl.screenshot(OUTPUT_IMAGE)
        print(f"[OK] Saved 4K Still: {OUTPUT_IMAGE}")

        # --- SCENE 2: The Rising Flood Animation ---
        print("[VIS] Rendering Rising Flood Animation...")
        pl = pv.Plotter(off_screen=True, window_size=[1920, 1080])
        pl.set_background('black', top='#1a1a2e')
        pl.enable_eye_dome_lighting()
        
        if terrain_mesh is not None:
             if 'Red' in terrain_mesh.array_names:
                pl.add_mesh(terrain_mesh, rgb=True, point_size=2)
             else:
                pl.add_mesh(terrain_mesh, color='#555555', point_size=2)
        
        pl.add_mesh(water_cloud, scalars='Depth', cmap='turbo', point_size=3, opacity=0.8, clim=[0,5])

        pl.view_isometric()
        pl.camera.zoom(1.2)
        
        pl.open_gif(OUTPUT_GIF)
        path = pl.generate_orbital_path(n_points=60, shift=100)
        for pos in path.points:
            pl.camera.position = pos
            pl.write_frame()
        pl.close()
        print(f"[OK] Saved Animation: {OUTPUT_GIF}")

        # --- SCENE 3: The Split-Screen Comparison (Before vs After) ---
        print("[VIS] Rendering Split-Screen Comparison...")
        pl = pv.Plotter(off_screen=True, result_layout='1x2', window_size=[3840, 1080])
        
        # Left Viewport (Before - Dry)
        pl.subplot(0, 0)
        pl.add_text("BEFORE: Current State", position='upper_left', font_size=20, color='white')
        pl.set_background('black')
        pl.enable_eye_dome_lighting()
        if terrain_mesh is not None:
             if 'Red' in terrain_mesh.array_names:
                pl.add_mesh(terrain_mesh, rgb=True, point_size=2)
             else:
                pl.add_mesh(terrain_mesh, color='#8B4513', point_size=2)
        pl.view_isometric()
        pl.camera.zoom(1.2)

        # Right Viewport (After - Flooded)
        pl.subplot(0, 1)
        pl.add_text("AFTER: AI Prediction (God Tier)", position='upper_left', font_size=20, color='red')
        pl.set_background('black', top='midnightblue')
        pl.enable_eye_dome_lighting()
        if terrain_mesh is not None:
             if 'Red' in terrain_mesh.array_names:
                pl.add_mesh(terrain_mesh, rgb=True, point_size=2)
             else:
                pl.add_mesh(terrain_mesh, color='#333333', point_size=2) # Darker terrain to make water pop
        pl.add_mesh(water_cloud, scalars='Depth', cmap='turbo', point_size=4, opacity=0.9, clim=[0, 5])
        pl.view_isometric()
        pl.camera.zoom(1.2)

        pl.link_views()
        pl.open_gif(OUTPUT_SPLIT)
        # But moving camera of subplot(0,1) should move (0,0) if linked.
        
        path = pl.generate_orbital_path(n_points=60, shift=100)
        for i, pos in enumerate(path.points):
            # For linked views, we only need to move one camera if correctly linked, 
            # but PyVista sometimes needs explicit updates.
            pl.subplot(0, 0)
            pl.camera.position = pos
            pl.subplot(0, 1)
            pl.camera.position = pos
            
            pl.write_frame()
        pl.close()
        print(f"[OK] Saved Split-Screen: {OUTPUT_SPLIT}")

    except Exception as e:
        print("[FAIL] Visualization Crash:")
        traceback.print_exc()

if __name__ == "__main__":
    run_vis()
'''
    if Path("vis_cinematic_heatmap.py").exists(): return
    with open("vis_cinematic_heatmap.py", "w", encoding="utf-8") as f:
        f.write(code)
    log("[OK] Generated Vis Script: vis_cinematic_heatmap.py")


# --- EXECUTION ENGINE ---

PIPELINE = [
    ("Ingestion", "ingest_data_full_scale.py"),
    ("Graph Build", "process_graph_full_scale.py"), 
    ("Training", "train_flood_gnn_full.py"),
    ("Export", "export_predictions_full.py"),
    ("Visualization", "vis_cinematic_heatmap.py")
]

def run_step(name, script):
    log(f"\n[START] {name}")
    
    # Generate on demand
    if script == "ingest_data_full_scale.py": generate_ingest_script()
    if script == "process_graph_full_scale.py": generate_graph_script()
    if script == "train_flood_gnn_full.py": generate_training_script()
    if script == "export_predictions_full.py": generate_export_script()
    if script == "vis_cinematic_heatmap.py": generate_vis_script()
    generate_model_file() 

    if not Path(script).exists():
        log(f"[WARN] Script {script} missing. Skipping.")
        return

    try:
        # Force UTF-8 encoding for the subprocess to handle emojis safely
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        # bufsize=1 means line-buffered, ensuring we get output instantly
        proc = subprocess.Popen([PYTHON_EXE, script], stdout=subprocess.PIPE, 
                                stderr=subprocess.STDOUT, text=True, bufsize=1, 
                                encoding='utf-8', env=env)
        
        # Capture and Log Output Line-by-Line
        for line in proc.stdout: 
            print(f"[{name}] {line}", end='') # Console
            log_output(f"[{name}] {line}")    # File (Persistent)

        proc.wait()
        
        if proc.returncode == 0: log(f"[OK] {name} SUCCESS")
        else: log(f"[FAIL] {name} FAILED (Code {proc.returncode})")
        
    except Exception as e:
        log(f"Fatal Error running {name}: {e}")
        traceback.print_exc()

def main():
    log("[INIT] GOD-TIER STATION INITIALIZED")
    log("Logging active: mission_control.log")
    
    for name, script in PIPELINE:
        run_step(name, script)
        
    log("\n[DONE] MISSION COMPLETE.")

if __name__ == "__main__":
    main()
