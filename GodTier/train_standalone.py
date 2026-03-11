"""
GOD-TIER Flood GNN Training - All-in-One Standalone Version
No separate imports needed - everything in one file!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from torch.utils.checkpoint import checkpoint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import os
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ========================================================================
#  MODEL DEFINITION (EMBEDDED)
# ========================================================================
class DepthMP(MessagePassing):
    """Memory-efficient message passing for depth/flow propagation"""
    def __init__(self, in_channels, edge_attr_dim):
        super().__init__(aggr='mean')
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + edge_attr_dim, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels)
        )
        
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        return self.mlp(torch.cat([x_j, edge_attr], dim=1))


class FloodGNN(nn.Module):
    """Graph Neural Network for flood prediction"""
    def __init__(self, in_channels, hidden_channels=64, edge_attr_dim=3, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.mp_layers = nn.ModuleList([
            DepthMP(hidden_channels, edge_attr_dim) 
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) 
            for _ in range(num_layers)
        ])
        
        self.depth_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels // 2, 2)
        )
    
    def forward(self, x, edge_index, edge_attr):
        h = self.encoder(x)
        
        for i, (mp_layer, norm) in enumerate(zip(self.mp_layers, self.layer_norms)):
            h_new = checkpoint(mp_layer, h, edge_index, edge_attr, use_reentrant=False)
            h = h + h_new
            h = norm(h)
            if i < self.num_layers - 1:
                h = F.dropout(h, p=0.1, training=self.training)
        
        depth = self.depth_head(h)
        velocity = self.velocity_head(h)
        
        return depth, velocity


# ========================================================================
#  CONFIGURATION
# ========================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 100
CHECKPOINT_INTERVAL = 5
BATCH_SIZE = 1024
NUM_NEIGHBORS = [10, 5]
LEARNING_RATE = 0.001
HIDDEN_DIM = 64
OUTPUT_DIR = Path("./outputs")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VIS_DIR = OUTPUT_DIR / "visualizations"

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
#  DATA LOADING
# ========================================================================
def load_graph_data():
    """Load the three village point cloud graphs"""
    graph_files = [
        "118118_Dariyapur_POINT CLOUD.pt",
        "118125_devipura urf Dhanna Nagla_POINT CLOUD.pt",
        "118129_Manjhola Khurd_POINT CLOUD.pt"
    ]
    
    graphs = []
    print("\nLoading Graphs...")
    for gf in graph_files:
        if not os.path.exists(gf):
            raise FileNotFoundError(f"Graph file not found: {gf}")
        
        data = torch.load(gf, map_location='cpu')
        print(f"  Loaded {gf}: {data.x.shape[0]:,} nodes | {data.edge_index.shape[1]:,} edges")
        graphs.append(data)
    
    return graphs


def create_data_loaders(graphs, train_idx, val_idx):
    """Create mini-batch loaders"""
    train_loaders = []
    val_loaders = []
    
    for idx in train_idx:
        data = graphs[idx]
        train_mask = torch.ones(data.x.shape[0], dtype=torch.bool)
        
        train_loader = NeighborLoader(
            data,
            num_neighbors=NUM_NEIGHBORS,
            batch_size=BATCH_SIZE,
            input_nodes=train_mask,
            shuffle=True,
            num_workers=0
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
            num_workers=0
        )
        val_loaders.append(val_loader)
    
    return train_loaders, val_loaders


# ========================================================================
#  TRAINING & VALIDATION
# ========================================================================
def train_epoch(model, train_loaders, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    for loader in train_loaders:
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            depth_pred, vel_pred = model(batch.x, batch.edge_index, batch.edge_attr)
            
            if hasattr(batch, 'y') and batch.y is not None:
                labels = (batch.y[:, 0] > 0.1).long()
                flood_logits = depth_pred.squeeze()
                flood_probs = torch.sigmoid(flood_logits)
                preds = (flood_probs > 0.5).long()
                
                loss = F.binary_cross_entropy_with_logits(flood_logits, labels.float())
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item() * batch.num_nodes
                total_correct += (preds == labels).sum().item()
                total_samples += batch.num_nodes
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0) if len(all_labels) > 0 else 0
    
    return avg_loss, accuracy, f1, all_preds, all_labels


def validate(model, val_loaders, device):
    """Validate"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for loader in val_loaders:
            for batch in loader:
                batch = batch.to(device)
                
                depth_pred, vel_pred = model(batch.x, batch.edge_index, batch.edge_attr)
                
                if hasattr(batch, 'y') and batch.y is not None:
                    labels = (batch.y[:, 0] > 0.1).long()
                    flood_logits = depth_pred.squeeze()
                    flood_probs = torch.sigmoid(flood_logits)
                    preds = (flood_probs > 0.5).long()
                    
                    loss = F.binary_cross_entropy_with_logits(flood_logits, labels.float())
                    
                    total_loss += loss.item() * batch.num_nodes
                    total_correct += (preds == labels).sum().item()
                    total_samples += batch.num_nodes
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0) if len(all_labels) > 0 else 0
    
    return avg_loss, accuracy, f1, all_preds, all_labels


# ========================================================================
#  CHECKPOINT MANAGEMENT
# ========================================================================
def save_checkpoint(model, optimizer, epoch, history, filename):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': {
            'epochs': history.epochs,
            'train_losses': history.train_losses,
            'val_losses': history.val_losses,
            'train_accuracies': history.train_accuracies,
            'val_accuracies': history.val_accuracies,
            'train_f1_scores': history.train_f1_scores,
            'val_f1_scores': history.val_f1_scores,
        }
    }
    torch.save(checkpoint, filename)
    print(f"  💾 Saved: {filename}")


# ========================================================================
#  VISUALIZATION
# ========================================================================
def plot_training_curves(history, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('GOD-TIER Training Analysis - 3 Villages (100 Epochs)', 
                 fontsize=16, fontweight='bold')
    
    epochs = history.epochs
    
    # Loss
    axes[0, 0].plot(epochs, history.train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history.val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Loss Curves', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history.train_accuracies, 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history.val_accuracies, 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1
    axes[1, 0].plot(epochs, history.train_f1_scores, 'b-', label='Train F1', linewidth=2)
    axes[1, 0].plot(epochs, history.val_f1_scores, 'r-', label='Val F1', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('F1 Score', fontsize=12)
    axes[1, 0].set_title('F1 Score Curves', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss gap
    loss_gap = [abs(t - v) for t, v in zip(history.train_losses, history.val_losses)]
    axes[1, 1].plot(epochs, loss_gap, 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('|Train Loss - Val Loss|', fontsize=12)
    axes[1, 1].set_title('Overfitting Monitor', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].fill_between(epochs, loss_gap, alpha=0.3, color='green')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  📊 Saved: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=['Not Flooded', 'Flooded'],
                yticklabels=['Not Flooded', 'Flooded'],
                cbar_kws={'label': 'Count'}, ax=ax,
                annot_kws={'size': 16, 'weight': 'bold'})
    
    ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = (cm[i, j] / total) * 100
            ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                   ha='center', va='center', fontsize=11, color='gray')
    
    # Metrics
    if len(cm) == 2:
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / total
        
        metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}'
        ax.text(1.5, 0.5, metrics_text, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  📊 Saved: {save_path}")
    plt.close()


# ========================================================================
#  MAIN TRAINING
# ========================================================================
def train_contingency():
    """Main training function"""
    
    print("=" * 70)
    print("  GOD-TIER FLOOD GNN TRAINING - 100 EPOCHS")
    print(f"  Device: {DEVICE}")
    print(f"  Batch Size: {BATCH_SIZE} nodes")
    print(f"  Neighbors: {NUM_NEIGHBORS}")
    print("=" * 70)
    
    # Load data
    graphs = load_graph_data()
    train_idx = [0, 1]
    val_idx = [2]
    print(f"Split: {len(train_idx)} Train / {len(val_idx)} Val")
    
    # Create loaders
    print("\nCreating mini-batch loaders...")
    train_loaders, val_loaders = create_data_loaders(graphs, train_idx, val_idx)
    
    # Model
    input_dim = graphs[0].x.shape[1]
    edge_attr_dim = graphs[0].edge_attr.shape[1] if graphs[0].edge_attr is not None else 0
    
    print(f"\nModel: FloodGNN")
    print(f"  Input: {input_dim}, Hidden: {HIDDEN_DIM}, Edge: {edge_attr_dim}")
    
    model = FloodGNN(
        in_channels=input_dim,
        hidden_channels=HIDDEN_DIM,
        edge_attr_dim=edge_attr_dim
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    history = TrainingHistory()
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    
    print(f"\nStarting {NUM_EPOCHS} Epochs...")
    print("-" * 70)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        train_loss, train_acc, train_f1, _, _ = train_epoch(
            model, train_loaders, optimizer, DEVICE
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loaders, DEVICE
        )
        
        # Update
        history.update(epoch, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1)
        scheduler.step(val_loss)
        
        # Print
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"TrL: {train_loss:.4f} | VaL: {val_loss:.4f} | "
              f"TrA: {train_acc:.4f} | VaA: {val_acc:.4f} | "
              f"TrF1: {train_f1:.4f} | VaF1: {val_f1:.4f}")
        
        # Checkpoints
        if epoch % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(model, optimizer, epoch, history, 
                          CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, history,
                          CHECKPOINT_DIR / "best_model_loss.pt")
            print(f"  🏆 New best loss: {val_loss:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_checkpoint(model, optimizer, epoch, history,
                          CHECKPOINT_DIR / "best_model_f1.pt")
            print(f"  🏆 New best F1: {val_f1:.4f}")
    
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE!")
    print("=" * 70)
    
    # Final viz
    print("\nGenerating visualizations...")
    _, _, _, final_preds, final_labels = validate(model, val_loaders, DEVICE)
    
    plot_training_curves(history, VIS_DIR / "training_curves.png")
    
    if len(final_labels) > 0:
        plot_confusion_matrix(final_labels, final_preds, 
                            VIS_DIR / "confusion_matrix.png")
    
    save_checkpoint(model, optimizer, NUM_EPOCHS, history,
                   CHECKPOINT_DIR / "final_model.pt")
    
    print("\n" + "=" * 70)
    print("  ALL OUTPUTS SAVED!")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"  Visualizations: {VIS_DIR}")
    print("=" * 70)
    
    return model, history


if __name__ == "__main__":
    model, history = train_contingency()
