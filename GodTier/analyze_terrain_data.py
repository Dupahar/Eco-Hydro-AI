"""
TERRAIN POINT CLOUD ANALYZER
Understand your SVAMITVA terrain data and prepare it for flood modeling
"""

import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ========================================================================
#  CONFIGURATION
# ========================================================================
BASE_DIR = Path(r"E:\DTM\Data\GodTier\Processed_Data\GodTier_V2\Graphs")
OUTPUT_DIR = Path("./terrain_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# ========================================================================
#  ANALYZE POINT CLOUD DATA
# ========================================================================
def analyze_terrain_point_cloud(file_path):
    """Analyze a single terrain point cloud graph"""
    
    print("\n" + "="*70)
    print(f"Analyzing: {file_path.name}")
    print("="*70)
    
    # Load data
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    
    # Basic info
    print(f"\n📊 BASIC STATISTICS:")
    print(f"  Total nodes: {data.x.shape[0]:,}")
    print(f"  Total edges: {data.edge_index.shape[1]:,}")
    print(f"  Node features: {data.x.shape[1]}")
    
    # What attributes exist?
    print(f"\n📋 AVAILABLE ATTRIBUTES:")
    for key in data.keys():
        attr = getattr(data, key)
        if torch.is_tensor(attr):
            print(f"  ✓ {key}: shape={attr.shape}, dtype={attr.dtype}")
            if key == 'pos' or key == 'x':
                print(f"    └─ Min: {attr.min(dim=0).values.tolist()}")
                print(f"    └─ Max: {attr.max(dim=0).values.tolist()}")
                print(f"    └─ Mean: {attr.mean(dim=0).tolist()}")
        else:
            print(f"  ✓ {key}: {type(attr)}")
    
    # Terrain analysis
    print(f"\n🗺️  TERRAIN CHARACTERISTICS:")
    
    if hasattr(data, 'pos'):
        pos = data.pos.numpy()
    elif hasattr(data, 'x'):
        pos = data.x.numpy()
    else:
        print("  ❌ No position data found!")
        return data
    
    # Assume columns are [X, Y, Z] or [X, Y, elevation]
    x_coords = pos[:, 0]
    y_coords = pos[:, 1]
    z_coords = pos[:, 2] if pos.shape[1] > 2 else None
    
    print(f"  X range: {x_coords.min():.2f} to {x_coords.max():.2f} ({np.ptp(x_coords):.2f} span)")
    print(f"  Y range: {y_coords.min():.2f} to {y_coords.max():.2f} ({np.ptp(y_coords):.2f} span)")
    
    if z_coords is not None:
        print(f"  Z range (elevation): {z_coords.min():.2f} to {z_coords.max():.2f}")
        print(f"  Elevation span: {np.ptp(z_coords):.2f} meters")
        print(f"  Mean elevation: {z_coords.mean():.2f} meters")
        print(f"  Std elevation: {z_coords.std():.2f} meters")
        
        # Terrain slope analysis
        elevation_diff = np.ptp(z_coords)
        area_diagonal = np.sqrt(np.ptp(x_coords)**2 + np.ptp(y_coords)**2)
        avg_slope = np.degrees(np.arctan(elevation_diff / area_diagonal))
        print(f"  Average terrain slope: {avg_slope:.2f} degrees")
    
    # Point density
    area = np.ptp(x_coords) * np.ptp(y_coords)
    density = len(pos) / area if area > 0 else 0
    print(f"\n📍 POINT DENSITY:")
    print(f"  Total area: {area:.2f} square units")
    print(f"  Point density: {density:.2f} points/unit²")
    
    # Memory usage
    memory_mb = sum([attr.element_size() * attr.nelement() for attr in [data.x, data.edge_index, data.edge_attr] if attr is not None]) / 1024 / 1024
    print(f"\n💾 MEMORY USAGE:")
    print(f"  Approximate: {memory_mb:.1f} MB")
    
    return data

# ========================================================================
#  VISUALIZE TERRAIN
# ========================================================================
def visualize_terrain(data, save_path, title="Terrain Point Cloud"):
    """Create 3D visualization of terrain"""
    
    print(f"\n📊 Creating visualization...")
    
    # Get position data
    if hasattr(data, 'pos'):
        pos = data.pos.numpy()
    else:
        pos = data.x.numpy()
    
    # Sample points for visualization (to avoid memory issues)
    n_samples = min(50000, len(pos))
    indices = np.random.choice(len(pos), n_samples, replace=False)
    pos_sample = pos[indices]
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(20, 6))
    
    # 3D view
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(pos_sample[:, 0], pos_sample[:, 1], pos_sample[:, 2], 
                         c=pos_sample[:, 2], cmap='terrain', s=0.5, alpha=0.6)
    ax1.set_xlabel('X (meters)', fontsize=10)
    ax1.set_ylabel('Y (meters)', fontsize=10)
    ax1.set_zlabel('Elevation (m)', fontsize=10)
    ax1.set_title('3D Terrain View', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax1, label='Elevation (m)', shrink=0.6)
    
    # Top-down view (elevation map)
    ax2 = fig.add_subplot(132)
    scatter2 = ax2.scatter(pos_sample[:, 0], pos_sample[:, 1], 
                          c=pos_sample[:, 2], cmap='terrain', s=1, alpha=0.8)
    ax2.set_xlabel('X (meters)', fontsize=10)
    ax2.set_ylabel('Y (meters)', fontsize=10)
    ax2.set_title('Top-Down Elevation Map', fontsize=12, fontweight='bold')
    ax2.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax2, label='Elevation (m)')
    
    # Elevation histogram
    ax3 = fig.add_subplot(133)
    ax3.hist(pos[:, 2], bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    ax3.set_xlabel('Elevation (m)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Elevation Distribution', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.axvline(pos[:, 2].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax3.legend()
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved to: {save_path}")

# ========================================================================
#  FLOOD SIMULATION GUIDANCE
# ========================================================================
def print_flood_modeling_guidance():
    """Print guidance on how to generate flood simulation data"""
    
    print("\n" + "="*70)
    print("🌊 NEXT STEPS: FLOOD SIMULATION")
    print("="*70)
    
    print("""
Your terrain point clouds are ready, but you need to generate flood simulation
data to train the GNN model. Here are your options:

┌─────────────────────────────────────────────────────────────────────┐
│ OPTION 1: Use Existing Flood Simulation Software                   │
└─────────────────────────────────────────────────────────────────────┘

Use hydrodynamic models to simulate flooding on your terrain:

1. HEC-RAS (Free, widely used)
   - Import your DEM/point cloud
   - Define flood scenarios (100-year, 500-year, etc.)
   - Run 2D hydraulic simulation
   - Export depth and velocity grids
   
2. LISFLOOD-FP (Academic, open-source)
   - Good for large-scale flood modeling
   - Can handle DEMs
   - Outputs depth/velocity rasters
   
3. MIKE FLOOD / TUFLOW (Commercial)
   - Professional-grade
   - High accuracy
   - Expensive but comprehensive

┌─────────────────────────────────────────────────────────────────────┐
│ OPTION 2: Simple Synthetic Flood Simulation                        │
└─────────────────────────────────────────────────────────────────────┘

Create synthetic flood scenarios based on terrain:

1. Height Above Nearest Drainage (HAND)
   - Calculate distance to drainage
   - Simulate different water levels
   - Simple but physically-based
   
2. Bathtub Model
   - Set water surface elevation
   - Flood everything below that level
   - Quick but ignores flow dynamics

┌─────────────────────────────────────────────────────────────────────┐
│ OPTION 3: Data-Driven Approach                                     │
└─────────────────────────────────────────────────────────────────────┘

If you have historical flood events:
- Use satellite imagery (Sentinel-1 SAR)
- Extract flood extents
- Match to terrain elevations
- Create training labels

┌─────────────────────────────────────────────────────────────────────┐
│ WHAT YOUR GNN MODEL NEEDS                                          │
└─────────────────────────────────────────────────────────────────────┘

For each node (point) in your terrain:
  ✓ depth (float): Water depth in meters
  ✓ velocity (2D vector): [vx, vy] flow velocity in m/s
  
Optional but helpful:
  • inundation (bool): Is this point flooded?
  • arrival_time (float): When does water arrive?
  • hazard_level (int): Risk classification (0-4)

┌─────────────────────────────────────────────────────────────────────┐
│ RECOMMENDED WORKFLOW                                                │
└─────────────────────────────────────────────────────────────────────┘

1. Convert point cloud to DEM (Digital Elevation Model)
   → Use QGIS or GDAL to create raster DEM
   
2. Run flood simulation (HEC-RAS recommended)
   → Input: DEM
   → Output: Depth and velocity grids
   
3. Sample simulation results back to point cloud
   → Match each point to nearest grid cell
   → Extract depth and velocity
   
4. Save as PyTorch Geometric Data object
   → data.depth = torch.tensor(depths)
   → data.velocity = torch.tensor(velocities)
   
5. Train GNN model
   → Now you have real flood labels!
   → Model learns terrain → flood mapping

┌─────────────────────────────────────────────────────────────────────┐
│ CODE TEMPLATE FOR ADDING FLOOD DATA                                │
└─────────────────────────────────────────────────────────────────────┘

# After running flood simulation, add labels to your graph:

import torch
import numpy as np

# Load your existing point cloud
data = torch.load('118118_Dariyapur_POINT CLOUD.pt', weights_only=False)

# Load flood simulation results (example)
# depths = load_flood_depths_from_simulation()
# velocities = load_flood_velocities_from_simulation()

# Match simulation grid to points (simple nearest neighbor)
from scipy.spatial import cKDTree

# Build KD-Tree for simulation grid
sim_coords = np.array([simulation_x, simulation_y]).T
tree = cKDTree(sim_coords)

# Find nearest simulation point for each terrain point
point_coords = data.pos[:, :2].numpy()  # X, Y only
distances, indices = tree.query(point_coords)

# Assign flood values
data.depth = torch.tensor(sim_depths[indices]).float()
data.velocity = torch.tensor(sim_velocities[indices]).float()

# Save updated graph
torch.save(data, '118118_Dariyapur_WITH_FLOOD_DATA.pt')

""")

# ========================================================================
#  MAIN ANALYSIS
# ========================================================================
def main():
    print("\n" + "="*70)
    print("  🗺️  TERRAIN POINT CLOUD ANALYZER")
    print("  Understanding SVAMITVA terrain data for flood modeling")
    print("="*70)
    
    if not BASE_DIR.exists():
        print(f"\n❌ ERROR: Directory not found: {BASE_DIR}")
        print("Update BASE_DIR in the script")
        return
    
    # Find all point cloud files
    pt_files = list(BASE_DIR.glob("*.pt"))
    village_files = [f for f in pt_files if any(vid in f.name for vid in ["118118", "118125", "118129"])]
    
    if not village_files:
        print(f"\n❌ No village files found in {BASE_DIR}")
        return
    
    print(f"\nFound {len(village_files)} village point cloud files")
    
    # Analyze each file
    all_data = {}
    for file_path in village_files:
        data = analyze_terrain_point_cloud(file_path)
        all_data[file_path.stem] = data
        
        # Create visualization
        vis_path = OUTPUT_DIR / f"{file_path.stem}_terrain_viz.png"
        visualize_terrain(data, vis_path, title=f"Terrain: {file_path.stem}")
    
    # Summary comparison
    print("\n" + "="*70)
    print("📊 SUMMARY COMPARISON")
    print("="*70)
    
    summary_data = []
    for name, data in all_data.items():
        pos = data.pos if hasattr(data, 'pos') else data.x
        summary_data.append({
            'Name': name,
            'Nodes': f"{data.x.shape[0]:,}",
            'Edges': f"{data.edge_index.shape[1]:,}",
            'Elevation Range': f"{pos[:, 2].min():.1f} - {pos[:, 2].max():.1f} m",
            'Mean Elev': f"{pos[:, 2].mean():.1f} m"
        })
    
    for item in summary_data:
        print(f"\n{item['Name']}:")
        for key, val in item.items():
            if key != 'Name':
                print(f"  {key:20s}: {val}")
    
    # Print guidance
    print_flood_modeling_guidance()
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print(f"📁 Visualizations saved to: {OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
