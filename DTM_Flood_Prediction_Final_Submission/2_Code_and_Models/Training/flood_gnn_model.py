import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.utils.checkpoint import checkpoint

# ========================================================================
#  MEMORY-EFFICIENT MESSAGE PASSING LAYER
# ========================================================================
class DepthMP(MessagePassing):
    """
    Memory-efficient message passing for depth/flow propagation
    Uses gradient checkpointing to reduce memory usage
    """
    def __init__(self, in_channels, edge_attr_dim):
        super().__init__(aggr='mean')  # Mean aggregation
        
        # Smaller MLP to reduce memory
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + edge_attr_dim, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels)
        )
        
    def forward(self, x, edge_index, edge_attr):
        # Use gradient checkpointing to save memory during backward pass
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        """Message function: combine neighbor features with edge attributes"""
        return self.mlp(torch.cat([x_j, edge_attr], dim=1))


# ========================================================================
#  MAIN FLOOD GNN MODEL - MEMORY OPTIMIZED
# ========================================================================
class FloodGNN(nn.Module):
    """
    Graph Neural Network for flood depth and velocity prediction
    
    Optimizations:
    - Smaller hidden dimensions
    - Gradient checkpointing
    - Mixed precision training support
    - Efficient message passing
    """
    def __init__(self, in_channels, hidden_channels=64, edge_attr_dim=3, num_layers=3):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Initial encoding (smaller to save memory)
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Message passing layers
        self.mp_layers = nn.ModuleList([
            DepthMP(hidden_channels, edge_attr_dim) 
            for _ in range(num_layers)
        ])
        
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) 
            for _ in range(num_layers)
        ])
        
        # Output heads (depth and velocity)
        self.depth_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels // 2, 1)  # Single output for depth
        )
        
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels // 2, 2)  # 2D velocity (vx, vy)
        )
    
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass with gradient checkpointing for memory efficiency
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_attr_dim]
        
        Returns:
            depth: Predicted flood depth [num_nodes, 1]
            velocity: Predicted velocity [num_nodes, 2]
        """
        # Initial encoding
        h = self.encoder(x)
        
        # Message passing with residual connections
        for i, (mp_layer, norm) in enumerate(zip(self.mp_layers, self.layer_norms)):
            # Use gradient checkpointing to save memory
            h_new = checkpoint(mp_layer, h, edge_index, edge_attr, use_reentrant=False)
            h = h + h_new  # Residual connection
            h = norm(h)  # Normalize
            
            # Optional: add dropout between layers
            if i < self.num_layers - 1:
                h = F.dropout(h, p=0.1, training=self.training)
        
        # Predict depth and velocity
        depth = self.depth_head(h)
        velocity = self.velocity_head(h)
        
        return depth, velocity
    
    def get_embeddings(self, x, edge_index, edge_attr):
        """
        Get node embeddings for visualization/analysis
        
        Returns:
            embeddings: Node embeddings [num_nodes, hidden_channels]
        """
        h = self.encoder(x)
        
        for mp_layer, norm in zip(self.mp_layers, self.layer_norms):
            h_new = mp_layer(h, edge_index, edge_attr)
            h = h + h_new
            h = norm(h)
        
        return h


# ========================================================================
#  LOSS FUNCTIONS
# ========================================================================
class FloodLoss(nn.Module):
    """
    Combined loss for depth and velocity prediction
    """
    def __init__(self, depth_weight=1.0, velocity_weight=0.5):
        super().__init__()
        self.depth_weight = depth_weight
        self.velocity_weight = velocity_weight
    
    def forward(self, depth_pred, vel_pred, depth_true, vel_true, mask=None):
        """
        Compute weighted loss
        
        Args:
            depth_pred: Predicted depths [num_nodes, 1]
            vel_pred: Predicted velocities [num_nodes, 2]
            depth_true: True depths [num_nodes, 1]
            vel_true: True velocities [num_nodes, 2]
            mask: Optional mask for valid nodes [num_nodes]
        """
        if mask is not None:
            depth_pred = depth_pred[mask]
            vel_pred = vel_pred[mask]
            depth_true = depth_true[mask]
            vel_true = vel_true[mask]
        
        # MSE for depth
        depth_loss = F.mse_loss(depth_pred, depth_true)
        
        # MSE for velocity
        vel_loss = F.mse_loss(vel_pred, vel_true)
        
        # Combined loss
        total_loss = self.depth_weight * depth_loss + self.velocity_weight * vel_loss
        
        return total_loss, depth_loss, vel_loss


# ========================================================================
#  UTILITY FUNCTIONS
# ========================================================================
def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model):
    """Print model architecture summary"""
    print("\n" + "=" * 70)
    print("  MODEL SUMMARY")
    print("=" * 70)
    print(f"  Total parameters: {count_parameters(model):,}")
    print("\n  Architecture:")
    print(model)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Test the model
    print("Testing FloodGNN model...")
    
    # Dummy data
    num_nodes = 1000
    num_edges = 5000
    in_channels = 10
    edge_attr_dim = 3
    
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_attr_dim)
    
    # Create model
    model = FloodGNN(in_channels=in_channels, hidden_channels=64, edge_attr_dim=edge_attr_dim)
    print_model_summary(model)
    
    # Forward pass
    depth, velocity = model(x, edge_index, edge_attr)
    
    print(f"Input shape: {x.shape}")
    print(f"Output depth shape: {depth.shape}")
    print(f"Output velocity shape: {velocity.shape}")
    print("\n✅ Model test passed!")
