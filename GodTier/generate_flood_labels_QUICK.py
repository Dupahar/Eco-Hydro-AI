"""
QUICK FLOOD LABEL GENERATOR (PATH 2)
Generates physics-inspired flood labels using Height Above Nearest Drainage (HAND)
This gets you started quickly so you can test your GNN pipeline
"""

import torch
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# ========================================================================
#  CONFIGURATION
# ========================================================================
BASE_DIR = Path(r"E:\DTM\Data\GodTier\Processed_Data\GodTier_V2\Graphs")
OUTPUT_DIR = Path(r"E:\DTM\Data\GodTier\Processed_Data\GodTier_V2\Graphs_With_Flood")
OUTPUT_DIR.mkdir(exist_ok=True)

# Flood scenario parameters
WATER_LEVELS = [2.0, 5.0, 10.0]  # Different flood intensities in meters
DEFAULT_WATER_LEVEL = 5.0  # Moderate flood

# ========================================================================
#  HAND-BASED FLOOD MODEL
# ========================================================================
def compute_flow_direction(points, k_neighbors=8):
    """
    Compute approximate flow direction based on local terrain slope
    Water flows from high to low elevation
    """
    # Build KD-Tree for neighbor search
    tree = cKDTree(points[:, :2])  # X, Y only
    
    # Find k nearest neighbors
    distances, indices = tree.query(points[:, :2], k=k_neighbors+1)
    
    flow_vectors = np.zeros((len(points), 2))
    
    print("  Computing flow directions...")
    # Process in batches to show progress
    batch_size = 100000
    for i in range(0, len(points), batch_size):
        end = min(i + batch_size, len(points))
        
        for j in range(i, end):
            # Get neighbors (excluding self)
            neighbor_idx = indices[j, 1:]
            neighbor_points = points[neighbor_idx]
            
            # Compute elevation differences
            current_z = points[j, 2]
            neighbor_z = neighbor_points[:, 2]
            dz = current_z - neighbor_z  # Positive if current is higher
            
            # Compute horizontal distances
            dx = neighbor_points[:, 0] - points[j, 0]
            dy = neighbor_points[:, 1] - points[j, 1]
            
            # Flow direction: weighted by elevation difference
            # Water flows toward lower neighbors
            weights = np.maximum(dz, 0)  # Only downhill
            if weights.sum() > 0:
                weights = weights / weights.sum()
                flow_vectors[j, 0] = (dx * weights).sum()
                flow_vectors[j, 1] = (dy * weights).sum()
        
        if (end - i) >= batch_size:
            print(f"    Processed {end:,}/{len(points):,} points...")
    
    # Normalize
    magnitudes = np.sqrt(flow_vectors[:, 0]**2 + flow_vectors[:, 1]**2)
    magnitudes[magnitudes == 0] = 1  # Avoid division by zero
    flow_vectors[:, 0] /= magnitudes
    flow_vectors[:, 1] /= magnitudes
    
    return flow_vectors

def generate_hand_flood(points, water_level=5.0, use_flow=True):
    """
    Generate flood depth and velocity using HAND approach
    
    Args:
        points: Nx3 array [x, y, elevation]
        water_level: Height of water surface above lowest point (meters)
        use_flow: Whether to compute flow directions (slower but more realistic)
    
    Returns:
        depth: N array of flood depths
        velocity: Nx2 array of flow velocities [vx, vy]
    """
    print(f"\n  Generating flood scenario (water level = {water_level}m)...")
    
    # Extract elevation
    z = points[:, 2]
    z_min = z.min()
    z_max = z.max()
    
    print(f"    Elevation range: {z_min:.2f} to {z_max:.2f} m")
    print(f"    Water surface: {z_min + water_level:.2f} m")
    
    # Flood depth = water surface - ground elevation
    water_surface = z_min + water_level
    depth = np.maximum(water_surface - z, 0)
    
    flooded_fraction = (depth > 0).sum() / len(depth)
    print(f"    Flooded area: {flooded_fraction*100:.1f}% of points")
    print(f"    Max depth: {depth.max():.2f} m")
    if (depth > 0).any():
        print(f"    Mean depth (where flooded): {depth[depth > 0].mean():.2f} m")
    
    # Velocity estimation
    if use_flow:
        print("  Computing flow velocities...")
        flow_directions = compute_flow_direction(points, k_neighbors=8)
    else:
        print("  Using simplified velocity model...")
        flow_directions = np.zeros((len(points), 2))
        flow_directions[:, 0] = 1.0  # Simple eastward flow
    
    # Velocity magnitude based on Manning's equation (simplified)
    # V ~ depth^(2/3) * slope^(1/2)
    # Simplified: V ~ sqrt(depth) with some scaling
    velocity_magnitude = np.sqrt(depth) * 0.5  # m/s, reasonable for floods
    
    # Apply flow direction
    velocity = np.zeros((len(points), 2))
    velocity[:, 0] = flow_directions[:, 0] * velocity_magnitude
    velocity[:, 1] = flow_directions[:, 1] * velocity_magnitude
    
    # Only flooded areas have velocity
    velocity[depth == 0] = 0
    
    avg_vel = np.sqrt(velocity[:, 0]**2 + velocity[:, 1]**2)
    print(f"    Max velocity: {avg_vel.max():.2f} m/s")
    print(f"    Mean velocity (where flooded): {avg_vel[depth > 0].mean():.2f} m/s")
    
    return depth, velocity

# ========================================================================
#  PROCESS GRAPHS
# ========================================================================
def add_flood_labels_to_graph(input_path, output_path, water_level=DEFAULT_WATER_LEVEL, use_flow=False):
    """
    Load graph, add flood labels, save updated version
    
    Args:
        use_flow: Set to False for faster processing, True for more realistic flow
    """
    print("\n" + "="*70)
    print(f"Processing: {input_path.name}")
    print("="*70)
    
    # Load graph
    print("  Loading graph...")
    data = torch.load(input_path, map_location='cpu', weights_only=False)
    
    print(f"  Nodes: {data.x.shape[0]:,}")
    print(f"  Edges: {data.edge_index.shape[1]:,}")
    
    # Get point coordinates
    if hasattr(data, 'pos'):
        points = data.pos.numpy()
    else:
        points = data.x.numpy()
    
    # Generate flood labels
    depth, velocity = generate_hand_flood(points, water_level, use_flow=use_flow)
    
    # Add to graph
    data.depth = torch.tensor(depth, dtype=torch.float32).unsqueeze(-1)
    data.velocity = torch.tensor(velocity, dtype=torch.float32)
    
    # Also add some useful metadata
    data.flood_scenario = water_level
    data.method = 'HAND'
    
    # Save
    print(f"\n  Saving to: {output_path}")
    torch.save(data, output_path)
    
    print("  ✓ Done!")
    
    return data

def visualize_flood_scenario(data, save_path, title="Flood Scenario"):
    """Quick visualization of the flood scenario"""
    
    print(f"\n  Creating visualization...")
    
    if hasattr(data, 'pos'):
        pos = data.pos.numpy()
    else:
        pos = data.x.numpy()
    
    depth = data.depth.numpy().flatten()
    velocity = data.velocity.numpy()
    
    # Sample for visualization
    n_samples = min(50000, len(pos))
    indices = np.random.choice(len(pos), n_samples, replace=False)
    
    pos_sample = pos[indices]
    depth_sample = depth[indices]
    vel_sample = velocity[indices]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Elevation map
    ax = axes[0, 0]
    scatter = ax.scatter(pos_sample[:, 0], pos_sample[:, 1], 
                        c=pos_sample[:, 2], cmap='terrain', s=1, alpha=0.6)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Terrain Elevation', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Elevation (m)')
    
    # 2. Flood depth
    ax = axes[0, 1]
    # Only show flooded areas
    flooded = depth_sample > 0.01
    scatter = ax.scatter(pos_sample[flooded, 0], pos_sample[flooded, 1], 
                        c=depth_sample[flooded], cmap='Blues', s=2, alpha=0.8)
    ax.scatter(pos_sample[~flooded, 0], pos_sample[~flooded, 1], 
              c='lightgray', s=0.5, alpha=0.3, label='Dry')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Flood Depth', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Depth (m)')
    ax.legend()
    
    # 3. Velocity magnitude
    ax = axes[1, 0]
    vel_mag = np.sqrt(vel_sample[:, 0]**2 + vel_sample[:, 1]**2)
    flooded = vel_mag > 0.01
    scatter = ax.scatter(pos_sample[flooded, 0], pos_sample[flooded, 1], 
                        c=vel_mag[flooded], cmap='Reds', s=2, alpha=0.8)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Flow Velocity', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Velocity (m/s)')
    
    # 4. Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    FLOOD SCENARIO STATISTICS
    {'='*40}
    
    Water Level: {data.flood_scenario:.1f} m above base
    Method: {data.method}
    
    Terrain:
      • Min elevation: {pos[:, 2].min():.2f} m
      • Max elevation: {pos[:, 2].max():.2f} m
      • Elevation range: {pos[:, 2].ptp():.2f} m
    
    Flooding:
      • Flooded area: {(depth > 0).sum() / len(depth) * 100:.1f}%
      • Max depth: {depth.max():.2f} m
      • Mean depth (flooded): {depth[depth > 0].mean():.2f} m
      • Median depth (flooded): {np.median(depth[depth > 0]):.2f} m
    
    Flow:
      • Max velocity: {vel_mag.max():.2f} m/s
      • Mean velocity (flooded): {vel_mag[depth > 0].mean():.2f} m/s
    
    Hazard Assessment:
      • Low risk (<0.5m): {(depth < 0.5).sum() / len(depth) * 100:.1f}%
      • Moderate (0.5-1m): {((depth >= 0.5) & (depth < 1.0)).sum() / len(depth) * 100:.1f}%
      • High (1-2m): {((depth >= 1.0) & (depth < 2.0)).sum() / len(depth) * 100:.1f}%
      • Extreme (>2m): {(depth >= 2.0).sum() / len(depth) * 100:.1f}%
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved visualization to: {save_path}")

# ========================================================================
#  MAIN PROCESSING
# ========================================================================
def main():
    print("\n" + "="*70)
    print("  🌊 QUICK FLOOD LABEL GENERATOR")
    print("  Generating HAND-based flood scenarios for GNN training")
    print("="*70)
    
    # Find village files
    pt_files = list(BASE_DIR.glob("*.pt"))
    village_files = [f for f in pt_files if any(vid in f.name for vid in ["118118", "118125", "118129"])]
    
    if not village_files:
        print(f"\n❌ No village files found in {BASE_DIR}")
        return
    
    print(f"\nFound {len(village_files)} village files")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Ask user about flow computation
    print("\n" + "="*70)
    print("FLOW VELOCITY OPTIONS:")
    print("  1. Simple (FAST): ~2-3 minutes per village")
    print("  2. Realistic (SLOW): ~10-20 minutes per village")
    print("="*70)
    
    use_realistic_flow = False  # Default to fast mode
    
    print("\nUsing SIMPLE mode for quick results")
    print("(You can edit the script to enable realistic flow later)")
    
    # Process each file
    processed_files = []
    
    for file_path in village_files:
        # Create output filename
        output_name = file_path.stem + "_WITH_FLOOD.pt"
        output_path = OUTPUT_DIR / output_name
        
        # Process
        data = add_flood_labels_to_graph(
            file_path, 
            output_path, 
            water_level=DEFAULT_WATER_LEVEL,
            use_flow=use_realistic_flow
        )
        
        # Visualize
        vis_path = OUTPUT_DIR / f"{file_path.stem}_flood_viz.png"
        visualize_flood_scenario(data, vis_path, title=f"Flood Scenario: {file_path.stem}")
        
        processed_files.append(output_path)
    
    # Summary
    print("\n" + "="*70)
    print("✅ PROCESSING COMPLETE!")
    print("="*70)
    print(f"\nGenerated {len(processed_files)} flood scenarios:")
    for fp in processed_files:
        print(f"  ✓ {fp.name}")
    
    print(f"\n📁 All files saved to: {OUTPUT_DIR}")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("""
1. Check the visualizations in the output folder
2. Update your training script to use these new files:
   
   BASE_DIR = Path(r"E:\\DTM\\Data\\GodTier\\Processed_Data\\GodTier_V2\\Graphs_With_Flood")
   
3. Run training:
   python train_SIMPLIFIED_NO_TORCH_SPARSE.py
   
4. The model will now train on physics-based flood labels!

NOTE: These are proxy labels for quick testing. For production use,
consider running proper flood simulations (HEC-RAS, LISFLOOD-FP, etc.)
    """)

if __name__ == "__main__":
    main()
