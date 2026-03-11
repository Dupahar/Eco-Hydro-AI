import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_recall_fscore_support
import os
import sys
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import the model (assumes flood_gnn_model.py is in the same directory)
# If not, add the directory to the path
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

try:
    from flood_gnn_model import FloodGNN, FloodLoss
except ImportError as e:
    print(f"\n❌ ERROR: Cannot import FloodGNN model")
    print(f"   Make sure 'flood_gnn_model.py' is in the same directory as this script")
    print(f"   Script directory: {script_dir}")
    raise

# ========================================================================
#  CONFIGURATION
# ========================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 100
CHECKPOINT_INTERVAL = 5  # Save every 5 epochs
BATCH_SIZE = 1024  # Nodes per batch
NUM_NEIGHBORS = [10, 5]  # 2-hop neighbors: 10 from first hop, 5 from second hop
LEARNING_RATE = 0.001
HIDDEN_DIM = 64

# Update these paths to match your directory structure
BASE_DIR = Path(r"E:\DTM\Data\GodTier\Processed_Data\GodTier_V2\Graphs")
OUTPUT_DIR = Path("./outputs")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VIS_DIR = OUTPUT_DIR / "visualizations"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
VIS_DIR.mkdir(exist_ok=True)

# ========================================================================
#  TRAINING HISTORY TRACKER
# ========================================================================
class TrainingHistory:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_f1_scores = []
        self.val_f1_scores = []
        self.epochs = []
        
    def update(self, epoch, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.train_f1_scores.append(train_f1)
        self.val_f1_scores.append(val_f1)

# ========================================================================
#  DATA LOADING - WITH FLEXIBLE PATH HANDLING
# ========================================================================
def load_graph_data():
    """Load the three village point cloud graphs with flexible path handling"""
    
    # Try different filename variations
    graph_patterns = [
        "118118_Dariyapur_POINT CLOUD.pt",
        "118125_devipura urf Dhanna Nagla_POINT CLOUD.pt", 
        "118129_Manjhola Khurd_POINT CLOUD.pt",
        # Alternative patterns if files were renamed
        "118118_Dariyapur_POINT_CLOUD.pt",
        "118125_devipura_urf_Dhanna_Nagla_POINT_CLOUD.pt",
        "118129_Manjhola_Khurd_POINT_CLOUD.pt",
    ]
    
    graphs = []
    loaded_files = []
    
    print("\nSearching for graph files...")
    print(f"Base directory: {BASE_DIR}")
    
    # First, list all .pt files in the directory
    if BASE_DIR.exists():
        all_pt_files = list(BASE_DIR.glob("*.pt"))
        print(f"\nFound {len(all_pt_files)} .pt files in directory:")
        for f in all_pt_files:
            print(f"  - {f.name}")
    else:
        print(f"\n❌ ERROR: Base directory does not exist: {BASE_DIR}")
        print("\nPlease update BASE_DIR in the script to point to your graph files directory")
        sys.exit(1)
    
    # Try to load the three village files
    village_ids = ["118118", "118125", "118129"]
    
    for village_id in village_ids:
        # Find file matching this village ID
        matching_files = [f for f in all_pt_files if village_id in f.name]
        
        if not matching_files:
            print(f"\n❌ ERROR: No file found for village {village_id}")
            continue
        
        # Use the first matching file
        file_path = matching_files[0]
        
        try:
            print(f"\nLoading {file_path.name}...")
            # PyTorch 2.6 compatibility: use weights_only=False for PyTorch Geometric data
            # This is safe for your own trusted graph files
            data = torch.load(file_path, map_location='cpu', weights_only=False)
            
            # Validate the loaded data
            if not hasattr(data, 'x') or not hasattr(data, 'edge_index'):
                print(f"  ⚠️  Warning: File may not be a valid PyTorch Geometric graph")
                continue
                
            print(f"  ✓ Loaded: {data.x.shape[0]:,} nodes | {data.edge_index.shape[1]:,} edges")
            
            # Check for required attributes
            required_attrs = ['depth', 'velocity', 'elevation']
            missing_attrs = [attr for attr in required_attrs if not hasattr(data, attr)]
            if missing_attrs:
                print(f"  ⚠️  Warning: Missing attributes: {missing_attrs}")
            
            graphs.append(data)
            loaded_files.append(file_path.name)
            
        except Exception as e:
            print(f"  ❌ Error loading {file_path.name}: {e}")
            continue
    
    if len(graphs) == 0:
        print("\n❌ ERROR: No valid graph files could be loaded!")
        print("\nTroubleshooting:")
        print("1. Check that the .pt files are valid PyTorch Geometric Data objects")
        print("2. Verify the BASE_DIR path is correct")
        print("3. Ensure files have 'x' and 'edge_index' attributes")
        sys.exit(1)
    
    print(f"\n✓ Successfully loaded {len(graphs)} graphs:")
    for fname in loaded_files:
        print(f"  - {fname}")
    
    return graphs

# ========================================================================
#  CREATE NEIGHBOR LOADERS FOR MEMORY-EFFICIENT TRAINING
# ========================================================================
def create_data_loaders(graphs, train_idx, val_idx):
    """Create mini-batch loaders for training and validation"""
    train_loaders = []
    val_loaders = []
    
    for idx in train_idx:
        data = graphs[idx]
        # Create masks for all nodes (we'll sample from all nodes)
        train_mask = torch.ones(data.x.shape[0], dtype=torch.bool)
        
        train_loader = NeighborLoader(
            data,
            num_neighbors=NUM_NEIGHBORS,
            batch_size=BATCH_SIZE,
            input_nodes=train_mask,
            shuffle=True,
        )
        train_loaders.append(train_loader)
    
    for idx in val_idx:
        data = graphs[idx]
        val_mask = torch.ones(data.x.shape[0], dtype=torch.bool)
        
        val_loader = NeighborLoader(
            data,
            num_neighbors=NUM_NEIGHBORS,
            batch_size=BATCH_SIZE,
            input_nodes=val_mask,
            shuffle=False,
        )
        val_loaders.append(val_loader)
    
    return train_loaders, val_loaders

# ========================================================================
#  DEPTH CLASSIFICATION (FOR METRICS)
# ========================================================================
def classify_depth(depth_tensor, thresholds=[0.1, 0.5, 1.0]):
    """
    Classify depth into categories:
    0: dry (< 0.1m)
    1: shallow (0.1-0.5m)
    2: moderate (0.5-1.0m)
    3: deep (> 1.0m)
    """
    classes = torch.zeros_like(depth_tensor, dtype=torch.long)
    classes[depth_tensor >= thresholds[0]] = 1
    classes[depth_tensor >= thresholds[1]] = 2
    classes[depth_tensor >= thresholds[2]] = 3
    return classes.squeeze()

# ========================================================================
#  TRAINING STEP
# ========================================================================
def train_epoch(model, train_loaders, optimizer, criterion, device):
    """Train for one epoch across all training graphs"""
    model.train()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    for loader in train_loaders:
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            depth_pred, vel_pred = model(batch.x, batch.edge_index, batch.edge_attr)
            
            # Get true values
            depth_true = batch.depth.unsqueeze(-1) if batch.depth.dim() == 1 else batch.depth
            vel_true = batch.velocity
            
            # Compute loss
            loss, depth_loss, vel_loss = criterion(depth_pred, vel_pred, depth_true, vel_true)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * batch.x.shape[0]
            total_samples += batch.x.shape[0]
            
            # For classification metrics
            pred_classes = classify_depth(depth_pred.detach().cpu())
            true_classes = classify_depth(depth_true.detach().cpu())
            all_preds.append(pred_classes)
            all_labels.append(true_classes)
    
    # Calculate metrics
    avg_loss = total_loss / total_samples
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    accuracy = (all_preds == all_labels).float().mean().item()
    f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='weighted')
    
    return avg_loss, accuracy, f1

# ========================================================================
#  VALIDATION STEP
# ========================================================================
@torch.no_grad()
def validate(model, val_loaders, criterion, device):
    """Validate across all validation graphs"""
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    for loader in val_loaders:
        for batch in loader:
            batch = batch.to(device)
            
            # Forward pass
            depth_pred, vel_pred = model(batch.x, batch.edge_index, batch.edge_attr)
            
            # Get true values
            depth_true = batch.depth.unsqueeze(-1) if batch.depth.dim() == 1 else batch.depth
            vel_true = batch.velocity
            
            # Compute loss
            loss, _, _ = criterion(depth_pred, vel_pred, depth_true, vel_true)
            
            # Track metrics
            total_loss += loss.item() * batch.x.shape[0]
            total_samples += batch.x.shape[0]
            
            # For classification metrics
            pred_classes = classify_depth(depth_pred.cpu())
            true_classes = classify_depth(depth_true.cpu())
            all_preds.append(pred_classes)
            all_labels.append(true_classes)
    
    # Calculate metrics
    avg_loss = total_loss / total_samples
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    accuracy = (all_preds == all_labels).float().mean().item()
    f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='weighted')
    
    return avg_loss, accuracy, f1

# ========================================================================
#  VISUALIZATION FUNCTIONS
# ========================================================================
def plot_training_curves(history, save_path):
    """Plot loss, accuracy, and F1 curves"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(history.epochs, history.train_losses, label='Train', linewidth=2)
    axes[0].plot(history.epochs, history.val_losses, label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.epochs, history.train_accuracies, label='Train', linewidth=2)
    axes[1].plot(history.epochs, history.val_accuracies, label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # F1 Score
    axes[2].plot(history.epochs, history.train_f1_scores, label='Train', linewidth=2)
    axes[2].plot(history.epochs, history.val_f1_scores, label='Val', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('F1 Score', fontsize=12)
    axes[2].set_title('Training & Validation F1', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_predictions(model, graph, device, save_path, graph_name="Village"):
    """Visualize model predictions vs ground truth"""
    model.eval()
    
    with torch.no_grad():
        # Get predictions
        data = graph.to(device)
        depth_pred, vel_pred = model(data.x, data.edge_index, data.edge_attr)
        
        # Move to CPU for visualization
        depth_pred = depth_pred.cpu().numpy().flatten()
        depth_true = data.depth.cpu().numpy()
        
        # Sample points for visualization (to avoid memory issues)
        n_samples = min(10000, len(depth_pred))
        indices = np.random.choice(len(depth_pred), n_samples, replace=False)
        
        depth_pred_sample = depth_pred[indices]
        depth_true_sample = depth_true[indices]
        
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    axes[0].scatter(depth_true_sample, depth_pred_sample, alpha=0.3, s=1)
    axes[0].plot([0, depth_true_sample.max()], [0, depth_true_sample.max()], 
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('True Depth (m)', fontsize=12)
    axes[0].set_ylabel('Predicted Depth (m)', fontsize=12)
    axes[0].set_title(f'{graph_name}: Prediction vs Truth', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Histogram of errors
    errors = depth_pred_sample - depth_true_sample
    axes[1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1].set_xlabel('Prediction Error (m)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Error Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ========================================================================
#  MAIN TRAINING FUNCTION
# ========================================================================
def train_contingency():
    """Main training loop for 3-village contingency training"""
    
    print("=" * 70)
    print("  GOD-TIER CONTINGENCY TRAINING (3 VILLAGES) — 100 EPOCHS")
    print(f"  Device: {DEVICE}")
    print(f"  Batch Size: {BATCH_SIZE} nodes")
    print(f"  Neighbors: {NUM_NEIGHBORS}")
    print("=" * 70)
    
    # Load graphs
    graphs = load_graph_data()
    
    if len(graphs) < 2:
        print("\n❌ ERROR: Need at least 2 graphs for train/val split")
        sys.exit(1)
    
    # Split: use first N-1 for training, last for validation
    train_idx = list(range(len(graphs) - 1))
    val_idx = [len(graphs) - 1]
    
    print(f"\nTraining on {len(train_idx)} graph(s), validating on {len(val_idx)} graph(s)")
    
    # Create data loaders
    train_loaders, val_loaders = create_data_loaders(graphs, train_idx, val_idx)
    
    # Initialize model
    in_channels = graphs[0].x.shape[1]
    edge_attr_dim = graphs[0].edge_attr.shape[1]
    
    model = FloodGNN(
        in_channels=in_channels,
        hidden_channels=HIDDEN_DIM,
        edge_attr_dim=edge_attr_dim,
        num_layers=3
    ).to(DEVICE)
    
    print(f"\n✓ Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss and optimizer
    criterion = FloodLoss(depth_weight=1.0, velocity_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training history
    history = TrainingHistory()
    
    # Training loop
    print("\n" + "=" * 70)
    print("  STARTING TRAINING")
    print("=" * 70)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loaders, optimizer, criterion, DEVICE
        )
        
        # Validate
        val_loss, val_acc, val_f1 = validate(
            model, val_loaders, criterion, DEVICE
        )
        
        # Update history
        history.update(epoch, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1)
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
            print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        # Save checkpoint
        if epoch % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = CHECKPOINT_DIR / f"model_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
    
    # Save final model
    final_model_path = CHECKPOINT_DIR / "model_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✓ Final model saved to {final_model_path}")
    
    # Plot training curves
    curves_path = VIS_DIR / "training_curves.png"
    plot_training_curves(history, curves_path)
    print(f"✓ Training curves saved to {curves_path}")
    
    # Visualize predictions on validation graph
    for i, idx in enumerate(val_idx):
        pred_path = VIS_DIR / f"predictions_val_graph_{i}.png"
        visualize_predictions(model, graphs[idx], DEVICE, pred_path, f"Validation Graph {i}")
        print(f"✓ Predictions saved to {pred_path}")
    
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE!")
    print("=" * 70)
    
    return model, history

# ========================================================================
#  ENTRY POINT
# ========================================================================
if __name__ == "__main__":
    try:
        model, history = train_contingency()
        print("\n✅ All done! Check the outputs/ directory for results.")
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
