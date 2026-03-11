"""
=============================================================================
MEMORY-EFFICIENT GNN TRAINING FOR 12GB GPU
=============================================================================

This version uses CPU-based processing with GPU acceleration only for the
forward/backward passes. Designed for A2000 12GB GPU with 7M+ node graphs.

Key optimizations:
- Subgraph sampling per batch
- CPU storage, GPU computation
- Gradient accumulation
- Mixed precision training
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, mean_squared_error, r2_score
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Import the model
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

try:
    from flood_gnn_model import FloodGNN, FloodLoss
except ImportError as e:
    print(f"\n❌ ERROR: Cannot import FloodGNN model")
    print(f"   Make sure 'flood_gnn_model.py' is in: {script_dir}")
    raise

# =============================================================================
#  CONFIGURATION
# =============================================================================
class Config:
    """Optimized for A2000 12GB GPU"""
    
    # Paths
    BASE_DIR = Path(r"E:\DTM\Data\GodTier\Processed_Data\GodTier_V2\Graphs_With_Flood")
    OUTPUT_DIR = Path("./final_outputs")
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    VIS_DIR = OUTPUT_DIR / "visualizations"
    METRICS_DIR = OUTPUT_DIR / "metrics"
    REPORTS_DIR = OUTPUT_DIR / "reports"
    
    # Training hyperparameters - MEMORY OPTIMIZED
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_EPOCHS = 100
    BATCH_SIZE = 4096  # Nodes per batch
    SUBGRAPH_SIZE = 50000  # Sample this many nodes at a time
    ACCUMULATION_STEPS = 4  # Gradient accumulation
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    HIDDEN_DIM = 64
    NUM_LAYERS = 3
    DROPOUT = 0.1
    
    # Loss weights
    DEPTH_WEIGHT = 1.0
    VELOCITY_WEIGHT = 0.5
    
    # Validation & checkpointing
    CHECKPOINT_INTERVAL = 5
    VALIDATION_SAMPLES = 10000
    
    # Reproducibility
    RANDOM_SEED = 42
    
    @classmethod
    def create_directories(cls):
        """Create all output directories"""
        for dir_path in [cls.OUTPUT_DIR, cls.CHECKPOINT_DIR, cls.VIS_DIR, 
                         cls.METRICS_DIR, cls.REPORTS_DIR]:
            dir_path.mkdir(exist_ok=True, parents=True)
    
    @classmethod
    def save_config(cls, path):
        """Save configuration to JSON"""
        config_dict = {}
        for k, v in cls.__dict__.items():
            if k.startswith('_'):
                continue
            if callable(v) or isinstance(v, (classmethod, staticmethod)):
                continue
            if isinstance(v, (Path, torch.device)):
                config_dict[k] = str(v)
            else:
                config_dict[k] = v
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

# Set random seeds
torch.manual_seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)

# =============================================================================
#  MEMORY-EFFICIENT SUBGRAPH SAMPLER
# =============================================================================
class SubgraphSampler:
    """
    Sample subgraphs that fit in GPU memory
    Keeps graphs on CPU, only moves batches to GPU
    """
    def __init__(self, graph, subgraph_size=50000, batch_size=4096):
        self.graph = graph
        self.subgraph_size = subgraph_size
        self.batch_size = batch_size
        self.num_nodes = graph.x.shape[0]
        
    def __iter__(self):
        """Iterate over subgraphs"""
        # Shuffle node indices
        indices = torch.randperm(self.num_nodes)
        
        # Process in subgraphs
        for i in range(0, self.num_nodes, self.subgraph_size):
            end_idx = min(i + self.subgraph_size, self.num_nodes)
            subgraph_nodes = indices[i:end_idx]
            
            # Extract subgraph
            subgraph = self._extract_subgraph(subgraph_nodes)
            
            # Yield batches from this subgraph
            for batch in self._batch_subgraph(subgraph):
                yield batch
    
    def _extract_subgraph(self, node_indices):
        """Extract a subgraph with selected nodes"""
        # Create node mask
        node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        node_mask[node_indices] = True
        
        # Find edges within subgraph
        edge_mask = node_mask[self.graph.edge_index[0]] & node_mask[self.graph.edge_index[1]]
        
        # Extract subgraph data
        sub_edge_index = self.graph.edge_index[:, edge_mask]
        sub_edge_attr = self.graph.edge_attr[edge_mask] if self.graph.edge_attr is not None else None
        
        # Remap node indices to 0-indexed
        unique_nodes = node_indices
        node_map = torch.zeros(self.num_nodes, dtype=torch.long)
        node_map[unique_nodes] = torch.arange(len(unique_nodes))
        
        sub_edge_index = node_map[sub_edge_index]
        
        # Create subgraph data object
        subgraph = Data(
            x=self.graph.x[unique_nodes],
            edge_index=sub_edge_index,
            edge_attr=sub_edge_attr,
            depth=self.graph.depth[unique_nodes],
            velocity=self.graph.velocity[unique_nodes],
            original_indices=unique_nodes
        )
        
        return subgraph
    
    def _batch_subgraph(self, subgraph):
        """Split subgraph into batches"""
        num_nodes = subgraph.x.shape[0]
        indices = torch.randperm(num_nodes)
        
        for i in range(0, num_nodes, self.batch_size):
            end_idx = min(i + self.batch_size, num_nodes)
            batch_indices = indices[i:end_idx]
            yield subgraph, batch_indices

# =============================================================================
#  TRAINING FUNCTIONS
# =============================================================================
def train_epoch_memory_efficient(model, graph_list, optimizer, criterion, device):
    """Memory-efficient training epoch"""
    model.train()
    total_loss = 0
    total_samples = 0
    num_batches = 0
    
    all_depth_pred = []
    all_depth_true = []
    all_vel_pred = []
    all_vel_true = []
    
    for graph_idx, graph in enumerate(graph_list):
        print(f"\n  Processing graph {graph_idx + 1}/{len(graph_list)}...")
        
        # Keep graph on CPU
        sampler = SubgraphSampler(
            graph, 
            subgraph_size=Config.SUBGRAPH_SIZE,
            batch_size=Config.BATCH_SIZE
        )
        
        optimizer.zero_grad()
        accum_loss = 0
        accum_count = 0
        
        for batch_idx, (subgraph, batch_indices) in enumerate(sampler):
            # Move only this subgraph to GPU
            subgraph = subgraph.to(device)
            
            # Forward pass
            depth_pred, vel_pred = model(subgraph.x, subgraph.edge_index, subgraph.edge_attr)
            
            # Get batch predictions
            batch_depth_pred = depth_pred[batch_indices]
            batch_vel_pred = vel_pred[batch_indices]
            batch_depth_true = subgraph.depth[batch_indices]
            batch_vel_true = subgraph.velocity[batch_indices]
            
            # Compute loss
            loss, _, _ = criterion(
                batch_depth_pred,
                batch_vel_pred,
                batch_depth_true,
                batch_vel_true
            )
            
            # Normalize loss for gradient accumulation
            loss = loss / Config.ACCUMULATION_STEPS
            loss.backward()
            
            accum_loss += loss.item() * Config.ACCUMULATION_STEPS
            accum_count += 1
            
            # Collect samples for metrics
            with torch.no_grad():
                all_depth_pred.append(batch_depth_pred.cpu())
                all_depth_true.append(batch_depth_true.cpu())
                all_vel_pred.append(batch_vel_pred.cpu())
                all_vel_true.append(batch_vel_true.cpu())
            
            # Update weights every N steps
            if accum_count % Config.ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += accum_loss
                total_samples += Config.BATCH_SIZE * Config.ACCUMULATION_STEPS
                num_batches += 1
                accum_loss = 0
                
                if num_batches % 10 == 0:
                    print(f"    Batch {num_batches}, Loss: {total_loss/total_samples:.4f}")
            
            # Free GPU memory
            del subgraph, depth_pred, vel_pred, loss
            torch.cuda.empty_cache()
    
    # Final optimizer step if needed
    if accum_count % Config.ACCUMULATION_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Compute epoch metrics
    all_depth_pred = torch.cat(all_depth_pred).numpy().flatten()
    all_depth_true = torch.cat(all_depth_true).numpy().flatten()
    
    metrics = {
        'loss': total_loss / max(total_samples, 1),
        'depth_mae': np.mean(np.abs(all_depth_pred - all_depth_true)),
        'depth_rmse': np.sqrt(mean_squared_error(all_depth_true, all_depth_pred)),
        'depth_r2': r2_score(all_depth_true, all_depth_pred) if len(np.unique(all_depth_true)) > 1 else 0.0
    }
    
    return metrics['loss'], metrics

def validate_epoch_memory_efficient(model, graph, device, num_samples=10000):
    """Memory-efficient validation"""
    model.eval()
    
    # Sample validation points
    num_nodes = graph.x.shape[0]
    sample_indices = torch.randperm(num_nodes)[:num_samples]
    
    # Extract small validation subgraph
    sampler = SubgraphSampler(graph, subgraph_size=num_samples, batch_size=num_samples)
    
    with torch.no_grad():
        for subgraph, _ in sampler:
            subgraph = subgraph.to(device)
            depth_pred, vel_pred = model(subgraph.x, subgraph.edge_index, subgraph.edge_attr)
            
            depth_pred = depth_pred.cpu().numpy().flatten()
            depth_true = subgraph.depth.cpu().numpy().flatten()
            
            metrics = {
                'depth_mae': np.mean(np.abs(depth_pred - depth_true)),
                'depth_rmse': np.sqrt(mean_squared_error(depth_true, depth_pred)),
                'depth_r2': r2_score(depth_true, depth_pred) if len(np.unique(depth_true)) > 1 else 0.0
            }
            
            del subgraph
            torch.cuda.empty_cache()
            
            return metrics
    
    return {'depth_mae': 0, 'depth_rmse': 0, 'depth_r2': 0}

# =============================================================================
#  MAIN TRAINING LOOP
# =============================================================================
def main_training():
    """Main training function"""
    print("\n" + "="*70)
    print("  🌊 MEMORY-EFFICIENT GNN TRAINING")
    print("  Optimized for A2000 12GB GPU")
    print("="*70)
    print(f"  Device: {Config.DEVICE}")
    print(f"  Subgraph Size: {Config.SUBGRAPH_SIZE:,} nodes")
    print(f"  Batch Size: {Config.BATCH_SIZE:,} nodes")
    print(f"  Gradient Accumulation: {Config.ACCUMULATION_STEPS} steps")
    print("="*70)
    
    # Create directories
    Config.create_directories()
    Config.save_config(Config.REPORTS_DIR / "training_config.json")
    
    # Load data
    print("\n" + "="*70)
    print("LOADING GRAPHS")
    print("="*70)
    
    graph_files = sorted(Config.BASE_DIR.glob("*_WITH_FLOOD.pt"))
    
    if not graph_files:
        print(f"\n❌ No flood-labeled graphs found in {Config.BASE_DIR}")
        return None, None
    
    print(f"\nFound {len(graph_files)} graph files\n")
    
    graphs = []
    for file_path in graph_files:
        print(f"Loading {file_path.name}...")
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        graphs.append(data)
        print(f"  ✓ Nodes: {data.x.shape[0]:,}, Edges: {data.edge_index.shape[1]:,}")
    
    # Split train/val
    train_graphs = graphs[:-1]
    val_graph = graphs[-1]
    
    print(f"\n✓ Training graphs: {len(train_graphs)}")
    print(f"✓ Validation graph: 1")
    
    # Initialize model
    model = FloodGNN(
        in_channels=3,
        hidden_channels=Config.HIDDEN_DIM,
        edge_attr_dim=3,
        num_layers=Config.NUM_LAYERS
    ).to(Config.DEVICE)
    
    print(f"\n✓ Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    criterion = FloodLoss(
        depth_weight=Config.DEPTH_WEIGHT,
        velocity_weight=Config.VELOCITY_WEIGHT
    )
    
    # Training loop
    print("\n" + "="*70)
    print("  STARTING TRAINING")
    print("="*70)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        print(f"\n{'='*70}")
        print(f"  EPOCH {epoch}/{Config.NUM_EPOCHS}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_metrics = train_epoch_memory_efficient(
            model, train_graphs, optimizer, criterion, Config.DEVICE
        )
        
        # Validate
        print("\n  Validating...")
        val_metrics = validate_epoch_memory_efficient(
            model, val_graph, Config.DEVICE, Config.VALIDATION_SAMPLES
        )
        val_loss = val_metrics['depth_mae']  # Use MAE as validation loss
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Track history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_metrics['depth_mae'])
        history['val_mae'].append(val_metrics['depth_mae'])
        
        # Print metrics
        epoch_time = time.time() - epoch_start
        print(f"\n  Results:")
        print(f"    Train Loss: {train_loss:.4f}")
        print(f"    Train MAE: {train_metrics['depth_mae']:.4f} m")
        print(f"    Train RMSE: {train_metrics['depth_rmse']:.4f} m")
        print(f"    Train R²: {train_metrics['depth_r2']:.4f}")
        print(f"    Val MAE: {val_metrics['depth_mae']:.4f} m")
        print(f"    Val RMSE: {val_metrics['depth_rmse']:.4f} m")
        print(f"    Val R²: {val_metrics['depth_r2']:.4f}")
        print(f"    Time: {epoch_time/60:.1f} min")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.CHECKPOINT_DIR / "best_model.pt")
            print(f"\n  ✓ New best model saved! (Val MAE: {val_loss:.4f})")
        
        # Periodic checkpoint
        if epoch % Config.CHECKPOINT_INTERVAL == 0:
            torch.save(model.state_dict(), Config.CHECKPOINT_DIR / f"model_epoch_{epoch}.pt")
            print(f"  ✓ Checkpoint saved")
        
        # Save metrics (convert numpy types to Python types for JSON)
        history_json = {
            k: [float(v) if hasattr(v, 'item') else v for v in vals]
            for k, vals in history.items()
        }
        with open(Config.METRICS_DIR / "training_metrics.json", 'w') as f:
            json.dump(history_json, f, indent=2)
    
    # Final save
    torch.save(model.state_dict(), Config.CHECKPOINT_DIR / "model_final.pt")
    
    print("\n" + "="*70)
    print("  ✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n  Best Val MAE: {best_val_loss:.4f} m")
    print(f"  Models saved to: {Config.CHECKPOINT_DIR}")
    print(f"  Metrics saved to: {Config.METRICS_DIR}")
    
    return model, history

if __name__ == "__main__":
    try:
        model, metrics = main_training()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
