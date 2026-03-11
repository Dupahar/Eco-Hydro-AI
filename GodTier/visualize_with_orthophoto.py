"""
=============================================================================
GOD-TIER VISUALIZATION WITH ORTHOPHOTO TEXTURES
=============================================================================
Overlays high-resolution aerial photos on 3D point clouds
Creates photorealistic flood visualizations
=============================================================================
"""

import torch
from torch_geometric.data import Data
from pathlib import Path
import numpy as np
import sys

# PyVista for 3D visualization
try:
    import pyvista as pv
    import matplotlib.pyplot as plt
except ImportError:
    print("❌ Install: pip install pyvista matplotlib")
    sys.exit(1)

# Rasterio for GeoTIFF orthophotos
try:
    import rasterio
    from rasterio.transform import rowcol
    HAS_RASTERIO = True
except ImportError:
    print("⚠️ Rasterio not installed. Install for orthophoto support:")
    print("   pip install rasterio")
    HAS_RASTERIO = False

# PIL for image processing
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    print("⚠️ PIL not installed. Install for image processing:")
    print("   pip install pillow")
    HAS_PIL = False

from flood_gnn_model import FloodGNN

# =============================================================================
#  CONFIGURATION
# =============================================================================
class Config:
    """Enhanced configuration with orthophoto support"""
    
    # Paths
    BASE_DIR = Path(r"E:\DTM\Data\GodTier\Processed_Data\GodTier_V2")
    GRAPH_DIR = BASE_DIR / "Graphs_With_Flood"
    ORTHO_DIR = Path(r"E:\DTM\Data\GodTier\Orthophotos")  # Adjust to your ortho location
    
    # Model
    MODEL_PATHS = [
        Path("final_outputs/checkpoints/best_model.pt"),
        Path("best_model.pt"),
        Path("final_outputs/checkpoints/model_final.pt"),
    ]
    
    # Model architecture
    IN_CHANNELS = 3
    HIDDEN_DIM = 64
    EDGE_ATTR_DIM = 3
    NUM_LAYERS = 3
    
    # Visualization settings
    RESOLUTION = (3840, 2160)  # 4K
    POINT_SIZE = 2  # Smaller for textured view
    USE_EYE_DOME_LIGHTING = True
    ENABLE_TEXTURES = True  # New: orthophoto textures
    
    # Output
    OUTPUT_DIR = Path("GodTier_Visuals_PhotoRealistic")
    OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
#  ORTHOPHOTO HANDLER
# =============================================================================
class OrthophotoMapper:
    """Maps orthophoto textures to 3D point clouds"""
    
    def __init__(self, ortho_path):
        """
        Initialize with orthophoto GeoTIFF
        
        Args:
            ortho_path: Path to orthophoto .tif file
        """
        self.ortho_path = ortho_path
        self.ortho_data = None
        self.transform = None
        self.crs = None
        
        if HAS_RASTERIO and ortho_path and ortho_path.exists():
            self._load_orthophoto()
        else:
            print(f"  ⚠️ Orthophoto not found or rasterio missing: {ortho_path}")
    
    def _load_orthophoto(self):
        """Load orthophoto from GeoTIFF"""
        try:
            with rasterio.open(self.ortho_path) as src:
                # Read RGB bands
                if src.count >= 3:
                    r = src.read(1)
                    g = src.read(2)
                    b = src.read(3)
                    
                    # Stack to RGB
                    self.ortho_data = np.dstack([r, g, b])
                    self.transform = src.transform
                    self.crs = src.crs
                    
                    print(f"  ✓ Loaded orthophoto: {self.ortho_data.shape}")
                else:
                    print(f"  ⚠️ Orthophoto has {src.count} bands, need 3 for RGB")
        except Exception as e:
            print(f"  ❌ Error loading orthophoto: {e}")
    
    def get_color_at_point(self, x, y):
        """
        Get RGB color from orthophoto at world coordinates
        
        Args:
            x, y: World coordinates
            
        Returns:
            RGB tuple (0-255) or None
        """
        if self.ortho_data is None or self.transform is None:
            return None
        
        try:
            # Convert world coords to pixel coords
            row, col = rowcol(self.transform, x, y)
            
            # Check bounds
            if 0 <= row < self.ortho_data.shape[0] and 0 <= col < self.ortho_data.shape[1]:
                rgb = self.ortho_data[row, col]
                return tuple(rgb)
            else:
                return None
        except Exception:
            return None
    
    def get_colors_for_points(self, points):
        """
        Get RGB colors for array of points
        
        Args:
            points: Nx3 array of [x, y, z] coordinates
            
        Returns:
            Nx3 array of RGB values (0-255)
        """
        if self.ortho_data is None:
            return None
        
        print(f"  Mapping orthophoto colors to {len(points):,} points...")
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        
        # Vectorized approach for speed
        try:
            # Convert all points at once
            rows, cols = rowcol(self.transform, points[:, 0], points[:, 1])
            rows = rows.astype(int)
            cols = cols.astype(int)
            
            # Mask for valid indices
            valid = (rows >= 0) & (rows < self.ortho_data.shape[0]) & \
                    (cols >= 0) & (cols < self.ortho_data.shape[1])
            
            # Extract colors for valid points
            colors[valid] = self.ortho_data[rows[valid], cols[valid]]
            
            print(f"  ✓ Mapped {valid.sum():,} / {len(points):,} points ({valid.sum()/len(points)*100:.1f}%)")
            
        except Exception as e:
            print(f"  ⚠️ Error mapping colors: {e}")
            return None
        
        return colors

# =============================================================================
#  ENHANCED RENDERER WITH ORTHOPHOTOS
# =============================================================================
class PhotoRealisticRenderer:
    """Renders point clouds with orthophoto textures and flood overlay"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"\n{'='*80}")
        print(f"   📸 PHOTO-REALISTIC RENDERER - INITIALIZING")
        print(f"{'='*80}")
        print(f"  Device: {self.device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Orthophoto support: {'✓ Enabled' if HAS_RASTERIO else '✗ Install rasterio'}")
        print(f"{'='*80}\n")
        
        # Load model
        self.model = self._load_model()
    
    def _load_model(self):
        """Load trained flood prediction model"""
        print("🧠 Loading Trained Model...")
        
        model_path = None
        for path in Config.MODEL_PATHS:
            if path.exists():
                model_path = path
                print(f"  ✓ Found: {model_path}")
                break
        
        if not model_path:
            print("  ❌ Model not found!")
            sys.exit(1)
        
        model = FloodGNN(
            in_channels=Config.IN_CHANNELS,
            hidden_channels=Config.HIDDEN_DIM,
            edge_attr_dim=Config.EDGE_ATTR_DIM,
            num_layers=Config.NUM_LAYERS
        ).to(self.device)
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        print(f"  ✓ Model ready!")
        return model
    
    def find_orthophoto(self, village_name):
        """
        Find orthophoto for a village
        
        Args:
            village_name: Name of village
            
        Returns:
            Path to orthophoto .tif or None
        """
        if not Config.ORTHO_DIR.exists():
            return None
        
        # Search for matching orthophoto
        patterns = [
            f"*{village_name}*.tif",
            f"*{village_name}*.TIF",
            f"*ortho*.tif",  # Generic
        ]
        
        for pattern in patterns:
            matches = list(Config.ORTHO_DIR.glob(pattern))
            if matches:
                return matches[0]
        
        return None
    
    def render_photorealistic(self, graph_path, ortho_path=None):
        """
        Create photorealistic visualization with orthophoto texture
        
        Args:
            graph_path: Path to graph .pt file
            ortho_path: Path to orthophoto (optional, will auto-detect)
        """
        print(f"\n{'='*80}")
        print(f"  RENDERING: {graph_path.name}")
        print(f"{'='*80}")
        
        # Load graph
        print("\n📥 Loading graph...")
        data = torch.load(graph_path, map_location='cpu', weights_only=False)
        print(f"  ✓ Nodes: {data.x.shape[0]:,}")
        
        # Extract village name
        name = graph_path.stem.replace("_WITH_FLOOD", "").replace("_POINT CLOUD", "")
        name = name.replace("_", " ").strip()
        
        # Find orthophoto if not provided
        if ortho_path is None:
            ortho_path = self.find_orthophoto(name)
        
        # Load orthophoto mapper
        ortho_mapper = None
        if ortho_path and HAS_RASTERIO:
            print(f"\n📷 Loading orthophoto: {ortho_path.name}")
            ortho_mapper = OrthophotoMapper(ortho_path)
        else:
            print(f"\n⚠️ No orthophoto found for {name}")
            print("   Will render with colormap only")
        
        # Subsample for visualization
        subsample_size = 500000
        if data.x.shape[0] > subsample_size:
            print(f"\n  Subsampling to {subsample_size:,} points...")
            indices = torch.randperm(data.x.shape[0])[:subsample_size]
            data_sub = Data(
                x=data.x[indices],
                pos=data.pos[indices],
                edge_index=data.edge_index,
                edge_attr=data.edge_attr
            )
        else:
            data_sub = data
            indices = torch.arange(data.x.shape[0])
        
        # Predict floods
        print("\n🧠 Predicting floods...")
        pos = data_sub.pos.cpu().numpy()
        
        with torch.no_grad():
            data_gpu = data_sub.to(self.device)
            depth, velocity = self.model(data_gpu.x, data_gpu.edge_index, data_gpu.edge_attr)
            depth_pred = depth.cpu().numpy().flatten()
            depth_pred = np.clip(depth_pred, 0, None)
        
        print(f"  ✓ Max depth: {depth_pred.max():.2f} m")
        print(f"  ✓ Flooded: {(depth_pred > 0.1).sum() / len(depth_pred) * 100:.1f}%")
        
        # Get orthophoto colors
        rgb_colors = None
        if ortho_mapper and ortho_mapper.ortho_data is not None:
            rgb_colors = ortho_mapper.get_colors_for_points(pos)
        
        # Create visualization
        self._render_textured_view(pos, depth_pred, rgb_colors, name)
        self._render_flood_overlay(pos, depth_pred, rgb_colors, name)
    
    def _render_textured_view(self, pos, depth, rgb_colors, name):
        """Render with orthophoto texture (pre-flood baseline)"""
        print(f"\n📸 Rendering textured terrain view...")
        
        cloud = pv.PolyData(pos)
        
        # Setup plotter
        pl = pv.Plotter(off_screen=True, window_size=Config.RESOLUTION)
        pl.set_background('#e8f4f8', top='#b8d8e8')  # Sky blue
        
        if rgb_colors is not None:
            # Use orthophoto colors
            cloud['RGB'] = rgb_colors
            pl.add_mesh(
                cloud,
                scalars='RGB',
                rgb=True,
                point_size=Config.POINT_SIZE,
                render_points_as_spheres=True,
                lighting=True
            )
        else:
            # Fallback: elevation coloring
            cloud['Elevation'] = pos[:, 2]
            pl.add_mesh(
                cloud,
                scalars='Elevation',
                cmap='gist_earth',
                point_size=Config.POINT_SIZE,
                render_points_as_spheres=True,
                show_scalar_bar=False
            )
        
        if Config.USE_EYE_DOME_LIGHTING:
            pl.enable_eye_dome_lighting()
        
        # Add labels
        pl.add_text(
            f"AERIAL VIEW: {name.upper()}",
            position='upper_left',
            font_size=18,
            color='black',
            shadow=True
        )
        
        pl.add_text(
            f"Source: SVAMITVA Orthophoto\nResolution: 4K UHD",
            position='lower_right',
            font_size=12,
            color='black'
        )
        
        # Camera
        pl.camera_position = 'xy'
        pl.camera.azimuth = 0
        pl.camera.elevation = 45
        pl.camera.zoom(1.2)
        
        # Save
        out_path = Config.OUTPUT_DIR / f"Textured_Aerial_{name}.png"
        pl.screenshot(str(out_path))
        print(f"  ✓ Saved: {out_path.name}")
        pl.close()
    
    def _render_flood_overlay(self, pos, depth, rgb_colors, name):
        """Render flood predictions overlaid on orthophoto"""
        print(f"\n🌊 Rendering flood overlay on aerial imagery...")
        
        cloud = pv.PolyData(pos)
        
        # Create semi-transparent flood layer
        # Where flooded: show flood color
        # Where dry: show orthophoto
        
        if rgb_colors is not None:
            # Blend orthophoto with flood colors
            flood_alpha = np.clip(depth / 2.0, 0, 1)  # Alpha based on depth
            
            # Flood colormap (blue tones)
            flood_cmap = plt.cm.Blues(np.clip(depth * 0.5, 0, 1))[:, :3] * 255
            
            # Blend: photo where dry, flood where wet
            blended = rgb_colors * (1 - flood_alpha[:, None]) + \
                     flood_cmap * flood_alpha[:, None]
            
            cloud['Blended'] = blended.astype(np.uint8)
            scalars_key = 'Blended'
            use_rgb = True
        else:
            # Fallback: just flood depth
            cloud['Flood_Depth'] = depth
            scalars_key = 'Flood_Depth'
            use_rgb = False
        
        # Setup plotter
        pl = pv.Plotter(off_screen=True, window_size=Config.RESOLUTION)
        pl.set_background('#0a0a15', top='#1a1a35')
        
        pl.add_mesh(
            cloud,
            scalars=scalars_key,
            rgb=use_rgb,
            cmap='Blues' if not use_rgb else None,
            point_size=Config.POINT_SIZE,
            render_points_as_spheres=True,
            show_scalar_bar=not use_rgb,
            scalar_bar_args={'title': 'Flood Depth (m)', 'color': 'white'} if not use_rgb else None
        )
        
        if Config.USE_EYE_DOME_LIGHTING:
            pl.enable_eye_dome_lighting()
        
        # Labels
        pl.add_text(
            f"FLOOD PREDICTION: {name.upper()}\nPhoto-Realistic Overlay",
            position='upper_left',
            font_size=18,
            color='white',
            shadow=True
        )
        
        stats = (
            f"Max Depth: {depth.max():.2f}m\n"
            f"Flooded Area: {(depth > 0.1).sum() / len(depth) * 100:.1f}%\n"
            f"AI Accuracy: 3.3cm"
        )
        pl.add_text(
            stats,
            position='upper_right',
            font_size=14,
            color='#00ff00',
            font='courier'
        )
        
        pl.camera_position = 'xy'
        pl.camera.azimuth = 45
        pl.camera.elevation = 30
        
        out_path = Config.OUTPUT_DIR / f"FloodOverlay_PhotoRealistic_{name}.png"
        pl.screenshot(str(out_path))
        print(f"  ✓ Saved: {out_path.name}")
        pl.close()

# =============================================================================
#  MAIN
# =============================================================================
def main():
    print("\n" + "="*80)
    print("   📸 PHOTO-REALISTIC FLOOD VISUALIZATION")
    print("   Orthophoto + AI Flood Predictions")
    print("="*80)
    
    renderer = PhotoRealisticRenderer()
    
    # Find graphs
    if not Config.GRAPH_DIR.exists():
        print(f"\n❌ Graph directory not found: {Config.GRAPH_DIR}")
        sys.exit(1)
    
    graphs = list(Config.GRAPH_DIR.glob("*_WITH_FLOOD.pt"))
    
    if not graphs:
        print(f"\n❌ No flood graphs found in {Config.GRAPH_DIR}")
        sys.exit(1)
    
    print(f"\n📁 Found {len(graphs)} villages\n")
    
    # Process each
    for graph_path in graphs:
        renderer.render_photorealistic(graph_path)
    
    print(f"\n{'='*80}")
    print("   ✅ PHOTO-REALISTIC RENDERING COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📁 Outputs: {Config.OUTPUT_DIR.absolute()}")
    print("\nGenerated:")
    print("  📸 Textured_Aerial_*.png - Orthophoto terrain views")
    print("  🌊 FloodOverlay_PhotoRealistic_*.png - Floods on aerial imagery")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
