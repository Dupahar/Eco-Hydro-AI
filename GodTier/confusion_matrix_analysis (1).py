"""
=============================================================================
CONFUSION MATRIX & CLASSIFICATION ANALYSIS
=============================================================================
Evaluates your trained flood prediction model with:
- Multi-class confusion matrices
- Per-class metrics (precision, recall, F1)
- Beautiful visualizations
- Comprehensive performance report
=============================================================================
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score
)
import pandas as pd
import json

# Import your model
try:
    from flood_gnn_model import FloodGNN
except ImportError:
    print("❌ Cannot import FloodGNN. Make sure flood_gnn_model.py is in the same directory.")
    import sys
    sys.exit(1)

# =============================================================================
#  CONFIGURATION
# =============================================================================
class Config:
    """Configuration for confusion matrix analysis"""
    
    # Paths
    BASE_DIR = Path(r"E:\DTM\Data\GodTier\Processed_Data\GodTier_V2")
    GRAPH_DIR = BASE_DIR / "Graphs_With_Flood"
    
    MODEL_PATHS = [
        Path("final_outputs/checkpoints/best_model.pt"),
        Path("best_model.pt"),
        Path("final_outputs/checkpoints/model_final.pt"),
    ]
    
    OUTPUT_DIR = Path("confusion_matrix_analysis")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Model architecture
    IN_CHANNELS = 3
    HIDDEN_DIM = 64
    EDGE_ATTR_DIM = 3
    NUM_LAYERS = 3
    
    # Flood classification thresholds (meters)
    DEPTH_THRESHOLDS = [0.0, 0.3, 0.6, 1.0, 2.0]
    CLASS_NAMES = ['Dry', 'Low Risk', 'Moderate', 'High', 'Extreme']
    CLASS_COLORS = ['#90EE90', '#FFFF00', '#FFA500', '#FF4500', '#8B0000']
    
    # Alternative: Binary classification
    BINARY_THRESHOLD = 0.1  # meters (flooded vs not flooded)

# =============================================================================
#  FLOOD CLASSIFIER
# =============================================================================
class FloodClassifier:
    """Classifies flood depths into risk categories"""
    
    @staticmethod
    def classify_depth(depth, thresholds=None):
        """
        Classify flood depth into risk categories
        
        Args:
            depth: Array of depths in meters
            thresholds: Custom thresholds (default uses Config)
            
        Returns:
            Array of class labels (0 to num_classes-1)
        """
        if thresholds is None:
            thresholds = Config.DEPTH_THRESHOLDS
        
        depth = np.asarray(depth)
        classes = np.zeros_like(depth, dtype=int)
        
        for i, threshold in enumerate(thresholds[1:], start=1):
            classes[depth >= threshold] = i
        
        return classes
    
    @staticmethod
    def classify_binary(depth, threshold=None):
        """Binary classification: flooded vs not flooded"""
        if threshold is None:
            threshold = Config.BINARY_THRESHOLD
        
        return (np.asarray(depth) >= threshold).astype(int)
    
    @staticmethod
    def get_class_name(class_idx):
        """Get human-readable class name"""
        if 0 <= class_idx < len(Config.CLASS_NAMES):
            return Config.CLASS_NAMES[class_idx]
        return f"Class_{class_idx}"

# =============================================================================
#  MODEL EVALUATOR
# =============================================================================
class ConfusionMatrixAnalyzer:
    """Evaluates model and creates confusion matrices"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"\n{'='*80}")
        print(f"   📊 CONFUSION MATRIX ANALYZER")
        print(f"{'='*80}")
        print(f"  Device: {self.device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"{'='*80}\n")
        
        self.model = self._load_model()
        self.classifier = FloodClassifier()
    
    def _load_model(self):
        """Load trained model"""
        print("🧠 Loading Trained Model...")
        
        model_path = None
        for path in Config.MODEL_PATHS:
            if path.exists():
                model_path = path
                print(f"  ✓ Found: {model_path}")
                break
        
        if not model_path:
            print("  ❌ Model not found!")
            import sys
            sys.exit(1)
        
        model = FloodGNN(
            in_channels=Config.IN_CHANNELS,
            hidden_channels=Config.HIDDEN_DIM,
            edge_attr_dim=Config.EDGE_ATTR_DIM,
            num_layers=Config.NUM_LAYERS
        ).to(self.device)
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        print(f"  ✓ Model loaded successfully!\n")
        return model
    
    def evaluate_graph(self, graph_path, subsample=None):
        """
        Evaluate model on a graph and return predictions + ground truth
        
        Args:
            graph_path: Path to graph .pt file
            subsample: Number of points to sample (None = all)
            
        Returns:
            y_true, y_pred: Ground truth and predicted classes
            depth_true, depth_pred: Actual depth values
        """
        print(f"\n📊 Evaluating: {graph_path.name}")
        
        # Load graph
        data = torch.load(graph_path, map_location='cpu', weights_only=False)
        print(f"  Nodes: {data.x.shape[0]:,}")
        
        # Check if ground truth exists
        if not hasattr(data, 'depth'):
            print("  ⚠️ No ground truth depth labels found!")
            return None, None, None, None
        
        # Subsample if needed
        if subsample and data.x.shape[0] > subsample:
            print(f"  Subsampling to {subsample:,} points...")
            indices = torch.randperm(data.x.shape[0])[:subsample]
            
            # Create a mapping from old indices to new indices
            node_mapping = torch.full((data.x.shape[0],), -1, dtype=torch.long)
            node_mapping[indices] = torch.arange(len(indices))
            
            # Filter edges to only include edges between sampled nodes
            edge_mask = (node_mapping[data.edge_index[0]] != -1) & (node_mapping[data.edge_index[1]] != -1)
            new_edge_index = data.edge_index[:, edge_mask]
            
            # Remap edge indices to new node numbering
            new_edge_index = torch.stack([
                node_mapping[new_edge_index[0]],
                node_mapping[new_edge_index[1]]
            ])
            
            print(f"  Edges retained: {new_edge_index.shape[1]:,} / {data.edge_index.shape[1]:,}")
            
            # Validate edge indices
            assert new_edge_index.min() >= 0, "Negative edge index!"
            assert new_edge_index.max() < len(indices), f"Edge index {new_edge_index.max()} >= num nodes {len(indices)}"
            
            data_sub = Data(
                x=data.x[indices],
                pos=data.pos[indices] if hasattr(data, 'pos') else None,
                edge_index=new_edge_index,
                edge_attr=data.edge_attr[edge_mask] if data.edge_attr is not None else None,
                depth=data.depth[indices],
                velocity=data.velocity[indices] if hasattr(data, 'velocity') else None
            )
        else:
            data_sub = data
        
        # Predict on full subsampled graph
        print("  Running predictions...")
        
        with torch.no_grad():
            try:
                # Try GPU first
                data_sub = data_sub.to(self.device)
                
                # Verify data before model forward pass
                print(f"  Graph on device: {data_sub.x.shape[0]} nodes, {data_sub.edge_index.shape[1]} edges")
                
                depth, _ = self.model(data_sub.x, data_sub.edge_index, data_sub.edge_attr)
                depth_pred = depth.cpu().numpy().flatten()
                depth_pred = np.clip(depth_pred, 0, None)  # Physical constraint
                
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e):
                    print(f"  ⚠️ GPU error: {e}")
                    print(f"  Falling back to CPU...")
                    
                    # Clear GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Move model to CPU
                    self.model = self.model.cpu()
                    self.device = torch.device('cpu')
                    
                    # Run on CPU
                    data_sub = data_sub.cpu()
                    depth, _ = self.model(data_sub.x, data_sub.edge_index, data_sub.edge_attr)
                    depth_pred = depth.numpy().flatten()
                    depth_pred = np.clip(depth_pred, 0, None)
                else:
                    raise
        
        # Get ground truth
        depth_true = data_sub.depth.cpu().numpy().flatten()
        
        # Classify into risk categories
        y_true = self.classifier.classify_depth(depth_true)
        y_pred = self.classifier.classify_depth(depth_pred)
        
        print(f"  ✓ Predictions complete")
        print(f"    True flooded: {(depth_true > 0.1).sum():,} points ({(depth_true > 0.1).mean()*100:.1f}%)")
        print(f"    Pred flooded: {(depth_pred > 0.1).sum():,} points ({(depth_pred > 0.1).mean()*100:.1f}%)")
        
        return y_true, y_pred, depth_true, depth_pred
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=False, title='Confusion Matrix'):
        """
        Create beautiful confusion matrix visualization
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize by row (recall)
            title: Plot title
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            # Avoid division by zero for classes with no samples
            row_sums = cm.sum(axis=1)[:, np.newaxis]
            row_sums[row_sums == 0] = 1  # Replace zeros with 1 to avoid NaN
            cm = cm.astype('float') / row_sums
            fmt = '.2%'
        else:
            fmt = 'd'
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=Config.CLASS_NAMES,
            yticklabels=Config.CLASS_NAMES,
            cbar_kws={'label': 'Percentage' if normalize else 'Count'},
            ax=ax,
            square=True,
            linewidths=1,
            linecolor='gray'
        )
        
        ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Rotate labels for readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix_with_metrics(self, y_true, y_pred, title='Confusion Matrix with Metrics'):
        """
        Create confusion matrix with accuracy, precision, recall, F1 displayed
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Calculate overall accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1], 
                             hspace=0.3, wspace=0.3)
        
        # Main confusion matrix
        ax_cm = fig.add_subplot(gs[0, 0])
        
        # Normalize for percentages (avoid division by zero)
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        row_sums[row_sums == 0] = 1
        cm_norm = cm.astype('float') / row_sums
        
        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=Config.CLASS_NAMES,
            yticklabels=Config.CLASS_NAMES,
            cbar_kws={'label': 'Count'},
            ax=ax_cm,
            square=True,
            linewidths=1,
            linecolor='gray'
        )
        
        ax_cm.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
        ax_cm.set_ylabel('True Class', fontsize=12, fontweight='bold')
        ax_cm.set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
        plt.setp(ax_cm.get_xticklabels(), rotation=45, ha='right')
        
        # Per-class metrics table (right side)
        ax_metrics = fig.add_subplot(gs[0, 1])
        ax_metrics.axis('off')
        
        # Create metrics table
        metrics_text = "PER-CLASS METRICS\n" + "="*30 + "\n\n"
        for i, class_name in enumerate(Config.CLASS_NAMES):
            if i < len(precision):
                metrics_text += f"{class_name}\n"
                metrics_text += f"  Precision: {precision[i]:>6.2%}\n"
                metrics_text += f"  Recall:    {recall[i]:>6.2%}\n"
                metrics_text += f"  F1-Score:  {f1[i]:>6.2%}\n"
                metrics_text += f"  Support:   {support[i]:>6,}\n\n"
        
        ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes,
                       fontsize=10, verticalalignment='top', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # Overall metrics (bottom)
        ax_overall = fig.add_subplot(gs[1, :])
        ax_overall.axis('off')
        
        # Calculate weighted averages
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        overall_text = f"""
OVERALL PERFORMANCE METRICS
{'='*80}

Overall Accuracy:          {accuracy*100:>6.2f}%
Weighted Precision:        {precision_avg*100:>6.2f}%
Weighted Recall:           {recall_avg*100:>6.2f}%
Weighted F1-Score:         {f1_avg*100:>6.2f}%
Cohen's Kappa:             {cohen_kappa_score(y_true, y_pred):>6.4f}

Total Samples:             {len(y_true):>8,}
"""
        
        ax_overall.text(0.5, 0.5, overall_text, transform=ax_overall.transAxes,
                       fontsize=11, verticalalignment='center', horizontalalignment='center',
                       family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        return fig
    
    def plot_binary_confusion_matrix(self, y_true, y_pred, title='Binary Confusion Matrix'):
        """Binary confusion matrix: Flooded vs Not Flooded"""
        
        # Binary classification
        y_true_bin = self.classifier.classify_binary(y_true)
        y_pred_bin = self.classifier.classify_binary(y_pred)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true_bin, y_pred_bin)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Normalize for percentages (avoid division by zero)
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        row_sums[row_sums == 0] = 1
        cm_norm = cm.astype('float') / row_sums
        
        # Plot
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2%',
            cmap='RdYlGn_r',
            xticklabels=['Not Flooded', 'Flooded'],
            yticklabels=['Not Flooded', 'Flooded'],
            cbar_kws={'label': 'Percentage'},
            ax=ax,
            square=True,
            linewidths=2,
            linecolor='black',
            vmin=0,
            vmax=1
        )
        
        # Add counts as secondary annotation
        for i in range(2):
            for j in range(2):
                text = ax.text(j + 0.5, i + 0.7, f'n={cm[i, j]:,}',
                             ha='center', va='center', color='gray', fontsize=9)
        
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def generate_classification_report(self, y_true, y_pred, depth_true, depth_pred):
        """Generate comprehensive classification metrics"""
        
        print("\n" + "="*80)
        print("  CLASSIFICATION REPORT")
        print("="*80)
        
        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
        
        # Cohen's Kappa (agreement accounting for chance)
        kappa = cohen_kappa_score(y_true, y_pred)
        print(f"Cohen's Kappa: {kappa:.4f}")
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Create detailed report
        report_data = []
        print("\n" + "-"*80)
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
        print("-"*80)
        
        for i, class_name in enumerate(Config.CLASS_NAMES):
            if i < len(precision):
                print(f"{class_name:<15} {precision[i]:>10.2%}  {recall[i]:>10.2%}  {f1[i]:>10.2%}  {support[i]:>10,}")
                report_data.append({
                    'Class': class_name,
                    'Precision': float(precision[i]),
                    'Recall': float(recall[i]),
                    'F1-Score': float(f1[i]),
                    'Support': int(support[i])
                })
        
        # Weighted averages
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        print("-"*80)
        print(f"{'Weighted Avg':<15} {precision_avg:>10.2%}  {recall_avg:>10.2%}  {f1_avg:>10.2%}")
        print("="*80)
        
        # Regression metrics
        print("\nREGRESSION METRICS (Depth Prediction):")
        print("-"*80)
        
        mae = np.mean(np.abs(depth_true - depth_pred))
        rmse = np.sqrt(np.mean((depth_true - depth_pred)**2))
        
        # R² score
        ss_res = np.sum((depth_true - depth_pred)**2)
        ss_tot = np.sum((depth_true - np.mean(depth_true))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"MAE (Mean Absolute Error):  {mae:.4f} m ({mae*100:.2f} cm)")
        print(f"RMSE (Root Mean Squared):   {rmse:.4f} m ({rmse*100:.2f} cm)")
        print(f"R² Score:                   {r2:.4f}")
        print("="*80)
        
        # Binary metrics
        y_true_bin = self.classifier.classify_binary(depth_true)
        y_pred_bin = self.classifier.classify_binary(depth_pred)
        
        bin_accuracy = accuracy_score(y_true_bin, y_pred_bin)
        bin_precision, bin_recall, bin_f1, _ = precision_recall_fscore_support(
            y_true_bin, y_pred_bin, average='binary', zero_division=0
        )
        
        print("\nBINARY CLASSIFICATION (Flooded vs Not Flooded):")
        print("-"*80)
        print(f"Accuracy:   {bin_accuracy*100:.2f}%")
        print(f"Precision:  {bin_precision*100:.2f}%")
        print(f"Recall:     {bin_recall*100:.2f}%")
        print(f"F1-Score:   {bin_f1*100:.2f}%")
        print("="*80)
        
        # Save to JSON
        report = {
            'multiclass': {
                'accuracy': float(accuracy),
                'cohen_kappa': float(kappa),
                'precision_weighted': float(precision_avg),
                'recall_weighted': float(recall_avg),
                'f1_weighted': float(f1_avg),
                'per_class': report_data
            },
            'regression': {
                'mae_meters': float(mae),
                'mae_cm': float(mae * 100),
                'rmse_meters': float(rmse),
                'r2_score': float(r2)
            },
            'binary': {
                'accuracy': float(bin_accuracy),
                'precision': float(bin_precision),
                'recall': float(bin_recall),
                'f1_score': float(bin_f1)
            }
        }
        
        return report
    
    def create_comprehensive_report(self, y_true, y_pred, depth_true, depth_pred, village_name):
        """Create all visualizations and reports for a village"""
        
        print(f"\n{'='*80}")
        print(f"  GENERATING COMPREHENSIVE REPORT: {village_name}")
        print(f"{'='*80}")
        
        # 1. Multi-class confusion matrix
        print("\n1. Creating multi-class confusion matrix...")
        fig1 = self.plot_confusion_matrix(
            y_true, y_pred, 
            normalize=False,
            title=f'Confusion Matrix - {village_name}\n(Absolute Counts)'
        )
        fig1.savefig(
            Config.OUTPUT_DIR / f'confusion_matrix_{village_name}.png',
            dpi=300,
            bbox_inches='tight'
        )
        print(f"  ✓ Saved: confusion_matrix_{village_name}.png")
        plt.close(fig1)
        
        # 2. Normalized confusion matrix
        print("\n2. Creating normalized confusion matrix...")
        fig2 = self.plot_confusion_matrix(
            y_true, y_pred,
            normalize=True,
            title=f'Normalized Confusion Matrix - {village_name}\n(Recall by Row)'
        )
        fig2.savefig(
            Config.OUTPUT_DIR / f'confusion_matrix_normalized_{village_name}.png',
            dpi=300,
            bbox_inches='tight'
        )
        print(f"  ✓ Saved: confusion_matrix_normalized_{village_name}.png")
        plt.close(fig2)
        
        # 3. Confusion matrix with metrics
        print("\n3. Creating confusion matrix with metrics...")
        fig3 = self.plot_confusion_matrix_with_metrics(
            y_true, y_pred,
            title=f'Confusion Matrix with Performance Metrics - {village_name}'
        )
        fig3.savefig(
            Config.OUTPUT_DIR / f'confusion_matrix_with_metrics_{village_name}.png',
            dpi=300,
            bbox_inches='tight'
        )
        print(f"  ✓ Saved: confusion_matrix_with_metrics_{village_name}.png")
        plt.close(fig3)
        
        # 4. Binary confusion matrix
        print("\n4. Creating binary confusion matrix...")
        fig4 = self.plot_binary_confusion_matrix(
            depth_true, depth_pred,
            title=f'Binary Classification - {village_name}\nFlooded vs Not Flooded'
        )
        fig4.savefig(
            Config.OUTPUT_DIR / f'confusion_matrix_binary_{village_name}.png',
            dpi=300,
            bbox_inches='tight'
        )
        print(f"  ✓ Saved: confusion_matrix_binary_{village_name}.png")
        plt.close(fig4)
        
        # 5. Classification report
        print("\n5. Generating classification metrics...")
        report = self.generate_classification_report(y_true, y_pred, depth_true, depth_pred)
        
        # Save report
        report_path = Config.OUTPUT_DIR / f'classification_report_{village_name}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n  ✓ Saved: classification_report_{village_name}.json")
        
        # 6. Create summary visualization
        print("\n6. Creating summary visualization...")
        fig5 = self.create_summary_figure(y_true, y_pred, depth_true, depth_pred, village_name)
        fig5.savefig(
            Config.OUTPUT_DIR / f'summary_{village_name}.png',
            dpi=300,
            bbox_inches='tight'
        )
        print(f"  ✓ Saved: summary_{village_name}.png")
        plt.close(fig5)
        
        return report
    
    def create_summary_figure(self, y_true, y_pred, depth_true, depth_pred, village_name):
        """Create a comprehensive 4-panel summary figure"""
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Panel 1: Multi-class confusion matrix (normalized)
        ax1 = fig.add_subplot(gs[0, 0])
        cm = confusion_matrix(y_true, y_pred)
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm_norm = cm.astype('float') / row_sums
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=Config.CLASS_NAMES, yticklabels=Config.CLASS_NAMES,
                   ax=ax1, cbar_kws={'label': 'Recall'})
        ax1.set_title('Multi-Class Confusion Matrix', fontweight='bold')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Panel 2: Binary confusion matrix
        ax2 = fig.add_subplot(gs[0, 1])
        y_true_bin = self.classifier.classify_binary(depth_true)
        y_pred_bin = self.classifier.classify_binary(depth_pred)
        cm_bin = confusion_matrix(y_true_bin, y_pred_bin)
        cm_bin_norm = cm_bin.astype('float') / cm_bin.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_bin_norm, annot=True, fmt='.2%', cmap='RdYlGn_r',
                   xticklabels=['Dry', 'Flooded'], yticklabels=['Dry', 'Flooded'],
                   ax=ax2, vmin=0, vmax=1)
        ax2.set_title('Binary Classification', fontweight='bold')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        
        # Panel 3: Depth prediction scatter
        ax3 = fig.add_subplot(gs[1, 0])
        # Subsample for plotting
        if len(depth_true) > 10000:
            idx = np.random.choice(len(depth_true), 10000, replace=False)
            depth_true_plot = depth_true[idx]
            depth_pred_plot = depth_pred[idx]
        else:
            depth_true_plot = depth_true
            depth_pred_plot = depth_pred
        
        ax3.scatter(depth_true_plot, depth_pred_plot, alpha=0.3, s=1)
        max_val = max(depth_true.max(), depth_pred.max())
        ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax3.set_xlabel('True Depth (m)', fontweight='bold')
        ax3.set_ylabel('Predicted Depth (m)', fontweight='bold')
        ax3.set_title('Depth Prediction Accuracy', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Metrics summary
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        # Calculate metrics
        mae = np.mean(np.abs(depth_true - depth_pred))
        rmse = np.sqrt(np.mean((depth_true - depth_pred)**2))
        accuracy = accuracy_score(y_true, y_pred)
        bin_accuracy = accuracy_score(y_true_bin, y_pred_bin)
        
        metrics_text = f"""
PERFORMANCE SUMMARY
{'='*40}

CLASSIFICATION METRICS:
  Multi-class Accuracy: {accuracy*100:.2f}%
  Binary Accuracy:      {bin_accuracy*100:.2f}%

REGRESSION METRICS:
  MAE:  {mae:.4f} m ({mae*100:.2f} cm)
  RMSE: {rmse:.4f} m ({rmse*100:.2f} cm)

DATA SUMMARY:
  Total Points:  {len(depth_true):,}
  True Flooded:  {(depth_true > 0.1).sum():,} ({(depth_true > 0.1).mean()*100:.1f}%)
  Pred Flooded:  {(depth_pred > 0.1).sum():,} ({(depth_pred > 0.1).mean()*100:.1f}%)

FLOOD RISK DISTRIBUTION:
"""
        
        # Add class distribution
        for i, class_name in enumerate(Config.CLASS_NAMES):
            true_count = (y_true == i).sum()
            pred_count = (y_pred == i).sum()
            metrics_text += f"  {class_name:<12} True: {true_count:>6,}  Pred: {pred_count:>6,}\n"
        
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        fig.suptitle(f'Confusion Matrix Analysis - {village_name}',
                    fontsize=16, fontweight='bold', y=0.98)
        
        return fig

# =============================================================================
#  MAIN EXECUTION
# =============================================================================
def main():
    print("\n" + "="*80)
    print("   📊 CONFUSION MATRIX & CLASSIFICATION ANALYSIS")
    print("   Comprehensive Model Evaluation")
    print("="*80)
    
    # Initialize analyzer
    analyzer = ConfusionMatrixAnalyzer()
    
    # Find graphs
    if not Config.GRAPH_DIR.exists():
        print(f"\n❌ Graph directory not found: {Config.GRAPH_DIR}")
        import sys
        sys.exit(1)
    
    graph_files = list(Config.GRAPH_DIR.glob("*_WITH_FLOOD.pt"))
    
    if not graph_files:
        print(f"\n❌ No flood graphs found in {Config.GRAPH_DIR}")
        import sys
        sys.exit(1)
    
    print(f"\n📁 Found {len(graph_files)} villages to analyze\n")
    
    # Process each village
    all_reports = {}
    
    for i, graph_path in enumerate(graph_files, 1):
        print(f"\n{'='*80}")
        print(f"  VILLAGE {i}/{len(graph_files)}: {graph_path.name}")
        print(f"{'='*80}")
        
        # Extract name
        village_name = graph_path.stem.replace("_WITH_FLOOD", "").replace("_POINT CLOUD", "")
        village_name = village_name.replace("_", "_").strip()
        
        # Evaluate
        y_true, y_pred, depth_true, depth_pred = analyzer.evaluate_graph(
            graph_path,
            subsample=500000  # Sample 500k points for speed
        )
        
        if y_true is None:
            print("  ⚠️ Skipping (no ground truth)")
            continue
        
        # Generate comprehensive report
        report = analyzer.create_comprehensive_report(
            y_true, y_pred, depth_true, depth_pred, village_name
        )
        
        all_reports[village_name] = report
    
    # Save combined report
    combined_path = Config.OUTPUT_DIR / 'all_villages_report.json'
    with open(combined_path, 'w') as f:
        json.dump(all_reports, f, indent=2)
    
    # Final summary
    print(f"\n{'='*80}")
    print("   ✅ ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📁 All outputs saved to: {Config.OUTPUT_DIR.absolute()}")
    print("\nGenerated files:")
    print("  📊 confusion_matrix_*.png - Multi-class confusion matrices (counts)")
    print("  📊 confusion_matrix_normalized_*.png - Normalized by recall")
    print("  📊 confusion_matrix_with_metrics_*.png - CM with Accuracy, Precision, Recall, F1")
    print("  📊 confusion_matrix_binary_*.png - Binary classification (flooded vs dry)")
    print("  📊 summary_*.png - 4-panel comprehensive view")
    print("  📄 classification_report_*.json - Detailed metrics")
    print("  📄 all_villages_report.json - Combined report")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()