"""
CRITICAL FIX FOR visualize_with_orthophoto.py

Replace lines 295-321 with this fixed version:
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Add at top of file after imports

# Also add this validation function near the top (after imports):
def validate_and_fix_data(data, expected_node_dim=3, expected_edge_dim=3):
    """Validate and fix data before model inference"""
    # Clean NaN/Inf
    if torch.isnan(data.x).any() or torch.isinf(data.x).any():
        data.x = torch.nan_to_num(data.x, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        if torch.isnan(data.edge_attr).any() or torch.isinf(data.edge_attr).any():
            data.edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Fix dimensions
    if data.x.shape[1] != expected_node_dim:
        if data.x.shape[1] < expected_node_dim:
            padding = torch.zeros(data.x.shape[0], expected_node_dim - data.x.shape[1], device=data.x.device)
            data.x = torch.cat([data.x, padding], dim=1)
        else:
            data.x = data.x[:, :expected_node_dim]
    
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        if data.edge_attr.shape[1] != expected_edge_dim:
            if data.edge_attr.shape[1] < expected_edge_dim:
                padding = torch.zeros(data.edge_attr.shape[0], expected_edge_dim - data.edge_attr.shape[1], device=data.edge_attr.device)
                data.edge_attr = torch.cat([data.edge_attr, padding], dim=1)
            else:
                data.edge_attr = data.edge_attr[:, :expected_edge_dim]
    
    # Validate edge indices
    if data.edge_index.max() >= data.num_nodes:
        raise ValueError(f"Edge index {data.edge_index.max()} >= num_nodes {data.num_nodes}. Graph corrupted!")
    
    return data


# Replace the render_photorealistic method (starting at line 261) with this:

def render_photorealistic(self, graph_path, ortho_path=None):
    """
    Create photorealistic visualization with orthophoto texture - FIXED VERSION
    
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
        
        # FIXED: Proper subgraph sampling
        num_nodes = data.x.shape[0]
        indices = torch.randperm(num_nodes)[:subsample_size].sort()[0]
        
        # Create node mask
        node_mask = torch.zeros(num_nodes, dtype=torch.bool)
        node_mask[indices] = True
        
        # Filter edges to only include those between sampled nodes
        edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
        
        # Create mapping from old indices to new
        node_mapping = torch.zeros(num_nodes, dtype=torch.long)
        node_mapping[indices] = torch.arange(len(indices))
        
        # Create properly subsampled graph
        data_sub = Data(
            x=data.x[indices],
            pos=data.pos[indices] if hasattr(data, 'pos') else data.x[indices, :3],
            edge_index=node_mapping[data.edge_index[:, edge_mask]]
        )
        
        # Add edge attributes if present
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data_sub.edge_attr = data.edge_attr[edge_mask]
    else:
        data_sub = data
        indices = torch.arange(data.x.shape[0])
    
    # CRITICAL: Validate and fix data BEFORE GPU transfer
    print("\n🔍 Validating data...")
    data_sub = validate_and_fix_data(data_sub, expected_node_dim=Config.IN_CHANNELS, expected_edge_dim=Config.EDGE_ATTR_DIM)
    print("  ✓ Data validated")
    
    # Get positions (before moving to GPU)
    if hasattr(data_sub, 'pos'):
        pos = data_sub.pos.cpu().numpy()
    else:
        pos = data_sub.x[:, :3].cpu().numpy()
    
    # Predict floods
    print("\n🧠 Predicting floods...")
    
    try:
        with torch.no_grad():
            # Move to GPU after validation
            data_gpu = data_sub.to(self.device)
            
            # Run model
            depth, velocity = self.model(data_gpu.x, data_gpu.edge_index, data_gpu.edge_attr)
            
            # Move results back to CPU
            depth_pred = depth.cpu().numpy().flatten()
            depth_pred = np.clip(depth_pred, 0, None)
        
        print(f"  ✓ Max depth: {depth_pred.max():.2f} m")
        print(f"  ✓ Flooded: {(depth_pred > 0.1).sum() / len(depth_pred) * 100:.1f}%")
        
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        print(f"\nDEBUG INFO:")
        print(f"  • Node features shape: {data_gpu.x.shape}")
        print(f"  • Edge index shape: {data_gpu.edge_index.shape}")
        if hasattr(data_gpu, 'edge_attr') and data_gpu.edge_attr is not None:
            print(f"  • Edge features shape: {data_gpu.edge_attr.shape}")
        raise
    
    # Get orthophoto colors
    rgb_colors = None
    if ortho_mapper and ortho_mapper.ortho_data is not None:
        rgb_colors = ortho_mapper.get_colors_for_points(pos)
    
    # Create visualization
    self._render_textured_view(pos, depth_pred, rgb_colors, name)
    self._render_flood_overlay(pos, depth_pred, rgb_colors, name)
