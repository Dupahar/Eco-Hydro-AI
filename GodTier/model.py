import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from pathlib import Path
import os
import warnings

# Suppress DLL warnings if they persist during runtime
warnings.filterwarnings('ignore')

# Import your local model
from flood_gnn_model_fixed import FloodGNN

# ========================================================================
#  CONFIGURATION
# ========================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 100
BATCH_SIZE = 1024
NUM_NEIGHBORS = [10, 5]
LEARNING_RATE = 0.001
HIDDEN_DIM = 64

# Updated Absolute Path for your environment
GRAPH_BASE_PATH = Path(r"E:\DTM\Data\GodTier\Processed_Data\GodTier_V2\Graphs")
OUTPUT_DIR = Path("./outputs")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VIS_DIR = OUTPUT_DIR / "visualizations"

for d in [OUTPUT_DIR, CHECKPOINT_DIR, VIS_DIR]: d.mkdir(exist_ok=True, parents=True)

class TrainingHistory:
    def __init__(self):
        self.train_losses, self.val_losses = [], []
        self.train_accuracies, self.val_accuracies = [], []
        self.train_f1_scores, self.val_f1_scores = [], []
        self.epochs = []
        
    def update(self, epoch, tl, vl, ta, va, tf, vf):
        self.epochs.append(epoch)
        self.train_losses.append(tl); self.val_losses.append(vl)
        self.train_accuracies.append(ta); self.val_accuracies.append(va)
        self.train_f1_scores.append(tf); self.val_f1_scores.append(vf)

# ========================================================================
#  LOAD DATA WITH CORRECT PATHS
# ========================================================================
def load_graph_data():
    graph_files = [
        "118118_Dariyapur_POINT CLOUD.pt",
        "118125_devipura urf Dhanna Nagla_POINT CLOUD.pt",
        "118129_Manjhola Khurd_POINT CLOUD.pt"
    ]
    graphs = []
    print(f"\nScanning for Graphs in: {GRAPH_BASE_PATH}")
    for gf in graph_files:
        full_path = GRAPH_BASE_PATH / gf
        if not full_path.exists():
            raise FileNotFoundError(f"CRITICAL: Graph file not found at {full_path}")
        
        data = torch.load(full_path, map_location='cpu')
        print(f"  Successfully loaded {gf} ({data.x.shape[0]} nodes)")
        graphs.append(data)
    return graphs

def create_data_loaders(graphs, train_idx, val_idx):
    train_loaders, val_loaders = [], []
    for idx in train_idx:
        train_loaders.append(NeighborLoader(graphs[idx], num_neighbors=NUM_NEIGHBORS, 
                                          batch_size=BATCH_SIZE, shuffle=True))
    for idx in val_idx:
        val_loaders.append(NeighborLoader(graphs[idx], num_neighbors=NUM_NEIGHBORS, 
                                        batch_size=BATCH_SIZE, shuffle=False))
    return train_loaders, val_loaders

# ========================================================================
#  TRAIN & VALIDATE
# ========================================================================
def train_epoch(model, loaders, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_preds, all_labels = [], []
    
    for loader in loaders:
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Predict
            depth_pred, _ = model(batch.x, batch.edge_index, batch.edge_attr)
            
            # Target logic: depth > 0.1m is flooded
            labels = (batch.y[:, 0] > 0.1).long()
            logits = depth_pred.squeeze()
            
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            loss.backward()
            optimizer.step()
            
            preds = (torch.sigmoid(logits) > 0.5).long()
            total_loss += loss.item() * batch.num_nodes
            total_correct += (preds == labels).sum().item()
            total_samples += batch.num_nodes
            all_preds.extend(preds.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
            
    return total_loss/total_samples, total_correct/total_samples, f1_score(all_labels, all_preds, zero_division=0)

def validate(model, loaders, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for loader in loaders:
            for batch in loader:
                batch = batch.to(device)
                depth_pred, _ = model(batch.x, batch.edge_index, batch.edge_attr)
                labels = (batch.y[:, 0] > 0.1).long()
                logits = depth_pred.squeeze()
                
                loss = F.binary_cross_entropy_with_logits(logits, labels.float())
                preds = (torch.sigmoid(logits) > 0.5).long()
                
                total_loss += loss.item() * batch.num_nodes
                total_correct += (preds == labels).sum().item()
                total_samples += batch.num_nodes
                all_preds.extend(preds.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
                
    return total_loss/total_samples, total_correct/total_samples, f1_score(all_labels, all_preds, zero_division=0), all_preds, all_labels

# ========================================================================
#  MAIN EXECUTION
# ========================================================================
def run_training():
    print("="*60)
    print(" GOD-TIER FLOOD GNN TRAINING")
    print("="*60)
    
    graphs = load_graph_data()
    train_loaders, val_loaders = create_data_loaders(graphs, [0, 1], [2])
    
    model = FloodGNN(
        in_channels=graphs[0].x.shape[1],
        hidden_channels=HIDDEN_DIM,
        edge_attr_dim=graphs[0].edge_attr.shape[1] if graphs[0].edge_attr is not None else 0
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = TrainingHistory()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        tl, ta, tf = train_epoch(model, train_loaders, optimizer, DEVICE)
        vl, va, vf, _, _ = validate(model, val_loaders, DEVICE)
        
        history.update(epoch, tl, vl, ta, va, tf, vf)
        print(f"Epoch {epoch:03d} | Train Loss: {tl:.4f} | Val F1: {vf:.4f} | Val Acc: {va:.4f}")
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"epoch_{epoch}.pt")

    print("\nTraining complete. Saving final model and generating plots...")
    torch.save(model.state_dict(), CHECKPOINT_DIR / "final_model.pt")

if __name__ == "__main__":
    run_training()