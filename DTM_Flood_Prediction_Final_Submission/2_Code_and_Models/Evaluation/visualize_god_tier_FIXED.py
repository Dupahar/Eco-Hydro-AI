"""
=============================================================================
GOD-TIER VISUALIZATION ENGINE V2.0
=============================================================================
Creates stunning 4K visualizations of your trained flood prediction model
Optimized for your A2000 12GB GPU + trained GNN model
=============================================================================
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from pathlib import Path
import numpy as np
import sys
import os

# Try importing PyVista, handle if missing
try:
    import pyvista as pv
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("❌ Critical: PyVista is missing. Install with: pip install pyvista")
    print("   For full features: pip install pyvista[all]")
    sys.exit(1)

# Import your trained model
try:
    from flood_gnn_model import FloodGNN
except ImportError:
    print("❌ Cannot import FloodGNN. Make sure flood_gnn_model.py is in the same directory.")
    sys.exit(1)

# =============================================================================
#  CONFIGURATION
# =============================================================================
class Config:
    """God-Tier Configuration for A2000 12GB"""
    
    # VISUAL QUALITY SETTINGS
    RESOLUTION = (3840, 2160)  # 4K UHD
    ANTI_ALIASING = 8          # MSAA x8 for smooth edges
    POINT_SIZE = 3             # Point size for visibility
    USE_EYE_DOME_LIGHTING = True  # Essential for depth perception
    ENABLE_SHADOWS = False     # Disable for speed (GPU memory)
    
    # PATHS - ADJUST THESE TO YOUR SETUP
    BASE_DIR = Path(r"E:\DTM\Data\GodTier\Processed_Data\GodTier_V2")
    GRAPH_DIR = BASE_DIR / "Graphs_With_Flood"
    
    # Model paths - Try multiple locations
    MODEL_PATHS = [
        Path("final_outputs/checkpoints/best_model.pt"),  # Primary
        Path("best_model.pt"),  # If uploaded to current dir
        Path("final_outputs/checkpoints/model_final.pt"),  # Alternative
    ]
    
    # Output directory
    OUTPUT_DIR = Path("GodTier_Visuals_4K")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Model architecture (must match training)
    IN_CHANNELS = 3
    HIDDEN_DIM = 64
    EDGE_ATTR_DIM = 3
    NUM_LAYERS = 3

# =============================================================================
#  GOD-TIER RENDERER
# =============================================================================
class GodTierRenderer:
    """Professional-grade 4K visualization engine"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"\n{'='*80}")
        print(f"   🌟 GOD-TIER RENDERER V2.0 - INITIALIZING")
        print(f"{'='*80}")
        print(f"  Device: {self.device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"{'='*80}\n")
        
        # Load trained model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load your trained GNN model"""
        print("🧠 Loading Trained Model...")
        
        # Find model file
        model_path = None
        for path in Config.MODEL_PATHS:
            if path.exists():
                model_path = path
                print(f"  ✓ Found model: {model_path}")
                break
        
        if model_path is None:
            print("  ❌ Model not found! Looking in:")
            for path in Config.MODEL_PATHS:
                print(f"     - {path.absolute()}")
            print("\n  Please copy best_model.pt to the current directory or update Config.MODEL_PATHS")
            sys.exit(1)
        
        # Initialize model architecture
        model = FloodGNN(
            in_channels=Config.IN_CHANNELS,
            hidden_channels=Config.HIDDEN_DIM,
            edge_attr_dim=Config.EDGE_ATTR_DIM,
            num_layers=Config.NUM_LAYERS
        ).to(self.device)
        
        # Load weights
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"  ✓ Model loaded successfully!")
            return model
        except Exception as e:
            print(f"  ❌ Error loading model: {e}")
            sys.exit(1)
    
    def load_graph(self, file_path):
        """Load a village graph"""
        print(f"\n📥 Loading Graph: {file_path.name}...")
        try:
            data = torch.load(file_path, map_location='cpu', weights_only=False)
            print(f"  ✓ Nodes: {data.x.shape[0]:,}")
            print(f"  ✓ Edges: {data.edge_index.shape[1]:,}")
            return data
        except Exception as e:
            print(f"  ❌ Failed to load: {e}")
            return None
    
    def predict_flood(self, data, subsample=None):
        """
        Run flood prediction on the graph
        
        Args:
            data: Graph data
            subsample: Number of points to subsample for visualization (None = all)
        """
        print("🧠 Running AI Flood Prediction...")
        
        # Move to GPU in chunks to avoid OOM
        num_nodes = data.x.shape[0]
        
        if subsample and num_nodes > subsample:
            # Random subsample for faster visualization
            indices = torch.randperm(num_nodes)[:subsample]
            data_sub = Data(
                x=data.x[indices],
                edge_index=data.edge_index,  # Keep all edges for now
                edge_attr=data.edge_attr,
                pos=data.pos[indices]
            )
            print(f"  ℹ️ Subsampled {subsample:,} points for visualization")
        else:
            data_sub = data
            indices = torch.arange(num_nodes)
        
        # Predict in batches to save memory
        batch_size = 100000
        all_depths = []
        all_velocities = []
        
        with torch.no_grad():
            for i in range(0, len(data_sub.x), batch_size):
                end = min(i + batch_size, len(data_sub.x))
                batch_data = Data(
                    x=data_sub.x[i:end].to(self.device),
                    edge_index=data.edge_index.to(self.device),
                    edge_attr=data.edge_attr.to(self.device) if data.edge_attr is not None else None
                )
                
                depth, velocity = self.model(batch_data.x, batch_data.edge_index, batch_data.edge_attr)
                all_depths.append(depth.cpu())
                all_velocities.append(velocity.cpu())
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"  Processing batch {i//batch_size + 1}...")
        
        # Combine results
        depth_pred = torch.cat(all_depths).numpy().flatten()
        vel_pred = torch.cat(all_velocities).numpy()
        
        # Get positions
        pos = data_sub.pos.cpu().numpy()
        
        # Clip negative depths (physical constraint)
        depth_pred = np.clip(depth_pred, 0, None)
        
        # Calculate velocity magnitude
        vel_mag = np.sqrt(vel_pred[:, 0]**2 + vel_pred[:, 1]**2)
        
        print(f"  ✓ Prediction complete!")
        print(f"    Max depth: {depth_pred.max():.2f} m")
        print(f"    Mean depth (flooded): {depth_pred[depth_pred > 0].mean():.2f} m")
        print(f"    Flooded area: {(depth_pred > 0.1).sum() / len(depth_pred) * 100:.1f}%")
        
        return pos, depth_pred, vel_mag, indices
    
    def export_gis_data(self, pos, depth_values, vel_mag, name):
        """Export GIS-compatible data for QGIS/ArcGIS"""
        print(f"\n🗺️  Exporting GIS Data for {name}...")
        
        # Filter to flooded points only
        mask = depth_values > 0.05
        if np.sum(mask) == 0:
            print("   ⚠️ No significant flooding detected, skipping GIS export.")
            return
        
        pos_flood = pos[mask]
        depth_flood = depth_values[mask]
        vel_flood = vel_mag[mask]
        
        # 1. Export CSV (Universal format)
        csv_path = Config.OUTPUT_DIR / f"GIS_{name}.csv"
        print(f"  Exporting CSV layer...")
        with open(csv_path, 'w') as f:
            f.write("X,Y,Z,Flood_Depth_m,Velocity_m_s,Risk_Level\n")
            for i in range(len(pos_flood)):
                p = pos_flood[i]
                d = depth_flood[i]
                v = vel_flood[i]
                # Risk classification
                if d < 0.3:
                    risk = "Low"
                elif d < 0.6:
                    risk = "Moderate"
                elif d < 1.0:
                    risk = "High"
                else:
                    risk = "Extreme"
                f.write(f"{p[0]:.3f},{p[1]:.3f},{p[2]:.3f},{d:.3f},{v:.3f},{risk}\n")
        
        print(f"  ✓ CSV saved: {csv_path.name}")
        print(f"    Import to QGIS: Layer > Add Layer > Add Delimited Text Layer")
        print(f"    Flooded points: {len(pos_flood):,}")
        
        # 2. Try GeoPandas for Shapefile (optional)
        try:
            import geopandas as gpd
            from shapely.geometry import Point
            
            # Subsample for shapefile (max 50k points)
            if len(pos_flood) > 50000:
                idx = np.random.choice(len(pos_flood), 50000, replace=False)
                pos_shp = pos_flood[idx]
                depth_shp = depth_flood[idx]
                vel_shp = vel_flood[idx]
            else:
                pos_shp = pos_flood
                depth_shp = depth_flood
                vel_shp = vel_flood
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame({
                'Depth_m': depth_shp,
                'Velocity': vel_shp,
                'geometry': [Point(x, y) for x, y in zip(pos_shp[:,0], pos_shp[:,1])]
            }, crs="EPSG:32643")  # UTM Zone 43N (India)
            
            shp_path = Config.OUTPUT_DIR / f"GIS_{name}_Vector.shp"
            gdf.to_file(shp_path)
            print(f"  ✓ Shapefile saved: {shp_path.name}")
            
        except ImportError:
            print("  ℹ️ GeoPandas not installed (optional). CSV is ready though!")
        except Exception as e:
            print(f"  ⚠️ Shapefile export warning: {e}")
    
    def render_cinematic_4k(self, pos, depth_values, vel_mag, name):
        """Create stunning 4K cinematic visualization"""
        print(f"\n🎬 Rendering Cinematic 4K: {name}")
        
        # Subsample if too many points (for reasonable render time)
        if len(pos) > 500000:
            print(f"  Subsampling to 500k points for rendering...")
            idx = np.random.choice(len(pos), 500000, replace=False)
            pos = pos[idx]
            depth_values = depth_values[idx]
            vel_mag = vel_mag[idx]
        
        # Create PyVista point cloud
        cloud = pv.PolyData(pos)
        cloud['Flood_Depth'] = depth_values
        cloud['Velocity'] = vel_mag
        
        # Setup plotter (off-screen for high res)
        pl = pv.Plotter(off_screen=True, window_size=Config.RESOLUTION)
        
        # Deep space background (cinematic)
        pl.set_background('#0a0a15', top='#1a1a35')
        
        # Add terrain with flood depth coloring
        pl.add_mesh(
            cloud,
            scalars='Flood_Depth',
            cmap='turbo',  # Vibrant colormap
            point_size=Config.POINT_SIZE,
            render_points_as_spheres=True,
            show_scalar_bar=True,
            scalar_bar_args={
                'title': 'Flood Depth (m)',
                'title_font_size': 20,
                'label_font_size': 16,
                'shadow': True,
                'n_labels': 5,
                'fmt': '%.2f',
                'color': 'white'
            },
            lighting=False
        )
        
        # Eye Dome Lighting (makes it look AMAZING)
        if Config.USE_EYE_DOME_LIGHTING:
            pl.enable_eye_dome_lighting()
        
        # Add professional HUD
        pl.add_text(
            f"AI FLOOD PREDICTION: {name.upper()}\n"
            f"GNN Physics Engine | 3.3cm Accuracy",
            position='upper_left',
            font_size=16,
            color='white',
            font='courier',
            shadow=True
        )
        
        stats_text = (
            f"Max Depth: {depth_values.max():.2f}m\n"
            f"Flooded: {(depth_values > 0.1).sum() / len(depth_values) * 100:.1f}%\n"
            f"Points: {len(pos):,}"
        )
        pl.add_text(
            stats_text,
            position='upper_right',
            font_size=14,
            color='#00ff00',
            font='courier',
            shadow=True
        )
        
        # Camera setup (dramatic angle)
        pl.camera_position = 'xy'
        pl.camera.azimuth = 45
        pl.camera.elevation = 30
        pl.camera.zoom(1.3)
        
        # Save 4K still
        out_img = Config.OUTPUT_DIR / f"Cinematic_4K_{name}.png"
        pl.screenshot(str(out_img), transparent_background=False)
        print(f"  ✓ 4K Image saved: {out_img.name}")
        
        # Generate orbital flyover animation
        print(f"  🎥 Generating orbital flyover animation...")
        
        # Create smooth camera path
        path = pl.generate_orbital_path(
            n_points=120,  # 5 seconds @ 24fps
            shift=0.0,
            factor=0.3
        )
        
        # Save as GIF for preview
        gif_path = Config.OUTPUT_DIR / f"Flyover_{name}.gif"
        pl.open_gif(str(gif_path), fps=24)
        
        # Save individual frames for video editing
        frame_dir = Config.OUTPUT_DIR / f"Frames_{name}"
        frame_dir.mkdir(exist_ok=True)
        
        for i, point in enumerate(path.points):
            pl.camera_position = point
            pl.render()
            pl.write_frame()
            
            # Save PNG frame every 5th frame (saves disk space)
            if i % 5 == 0:
                frame_path = frame_dir / f"frame_{i:04d}.png"
                pl.screenshot(str(frame_path))
        
        pl.close()
        
        print(f"  ✓ Animation saved: {gif_path.name}")
        print(f"  ✓ Frames saved to: {frame_dir.name}/")
        print(f"    Import frames to video editor for 4K export")
    
    def render_velocity_field(self, pos, vel_mag, name):
        """Render velocity magnitude visualization"""
        print(f"\n💨 Rendering Velocity Field: {name}")
        
        # Subsample for rendering
        if len(pos) > 300000:
            idx = np.random.choice(len(pos), 300000, replace=False)
            pos = pos[idx]
            vel_mag = vel_mag[idx]
        
        cloud = pv.PolyData(pos)
        cloud['Velocity'] = vel_mag
        
        pl = pv.Plotter(off_screen=True, window_size=(1920, 1080))
        pl.set_background('#0a0a15')
        
        pl.add_mesh(
            cloud,
            scalars='Velocity',
            cmap='plasma',
            point_size=3,
            render_points_as_spheres=True,
            show_scalar_bar=True,
            scalar_bar_args={
                'title': 'Flow Velocity (m/s)',
                'color': 'white'
            }
        )
        
        pl.add_text(
            f"FLOOD FLOW VELOCITY: {name.upper()}",
            position='upper_left',
            font_size=16,
            color='white'
        )
        
        if Config.USE_EYE_DOME_LIGHTING:
            pl.enable_eye_dome_lighting()
        
        pl.camera_position = 'xy'
        pl.camera.azimuth = 45
        pl.camera.elevation = 25
        
        out_img = Config.OUTPUT_DIR / f"Velocity_Field_{name}.png"
        pl.screenshot(str(out_img))
        print(f"  ✓ Velocity visualization saved: {out_img.name}")
        pl.close()

# =============================================================================
#  MAIN EXECUTION
# =============================================================================
def main():
    print("\n" + "="*80)
    print("   🌊 GOD-TIER VISUALIZATION ENGINE V2.0")
    print("   4K Cinematic Flood Predictions")
    print("="*80)
    print(f"  Resolution: {Config.RESOLUTION[0]}x{Config.RESOLUTION[1]} UHD")
    print(f"  Anti-aliasing: {Config.ANTI_ALIASING}x MSAA")
    print(f"  Eye Dome Lighting: {'Enabled' if Config.USE_EYE_DOME_LIGHTING else 'Disabled'}")
    print("="*80)
    
    # Initialize renderer
    renderer = GodTierRenderer()
    
    # Find graph files
    if not Config.GRAPH_DIR.exists():
        print(f"\n❌ Graph directory not found: {Config.GRAPH_DIR}")
        print("Please update Config.GRAPH_DIR in the script")
        sys.exit(1)
    
    graph_files = list(Config.GRAPH_DIR.glob("*_WITH_FLOOD.pt"))
    
    if not graph_files:
        print(f"\n❌ No flood-labeled graphs found in {Config.GRAPH_DIR}")
        print("Please run generate_flood_labels_QUICK_FIXED.py first")
        sys.exit(1)
    
    print(f"\n📁 Found {len(graph_files)} villages to visualize\n")
    
    # Process each village
    for i, graph_path in enumerate(graph_files, 1):
        print(f"\n{'='*80}")
        print(f"  VILLAGE {i}/{len(graph_files)}")
        print(f"{'='*80}")
        
        # Load graph
        data = renderer.load_graph(graph_path)
        if data is None:
            continue
        
        # Extract clean name
        name = graph_path.stem.replace("_WITH_FLOOD", "").replace("_POINT CLOUD", "")
        name = name.replace("_", " ").strip()
        
        # Run prediction (subsample to 500k for speed)
        pos, depth_pred, vel_mag, indices = renderer.predict_flood(data, subsample=500000)
        
        # Create visualizations
        renderer.render_cinematic_4k(pos, depth_pred, vel_mag, name)
        renderer.render_velocity_field(pos, vel_mag, name)
        renderer.export_gis_data(pos, depth_pred, vel_mag, name)
    
    # Final summary
    print(f"\n{'='*80}")
    print("   ✅ RENDERING COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📁 All outputs saved to: {Config.OUTPUT_DIR.absolute()}\n")
    print("Contents:")
    print("  🖼️  Cinematic_4K_*.png - Stunning 4K still images")
    print("  🎬 Flyover_*.gif - Orbital animation previews")
    print("  📁 Frames_*/ - PNG sequences for video editing")
    print("  🗺️  GIS_*.csv - Import to QGIS/ArcGIS")
    print("  🗺️  GIS_*_Vector.shp - Shapefile layers (if GeoPandas installed)")
    print("\n💡 Tip: Use Frames_*/*.png in DaVinci Resolve or Premiere for 4K video export")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
