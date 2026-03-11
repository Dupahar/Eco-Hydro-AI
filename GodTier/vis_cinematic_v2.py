import numpy as np
import torch
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from flood_gnn_model import get_gnn_model
import traceback

GRAPH_DIR = Path("Processed_Data/GodTier_V2/Graphs")
MODEL_PATH = Path("Models/GodTier_V2/gnn_best.pth")
METRICS_DIR = Path("Models/GodTier_V2/metrics")
OUTPUT_DIR = Path("Visuals/GodTier_V2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DEPTH_LABELS = ['Dry', 'Low', 'Medium', 'High', 'Critical']

# Custom colormaps
FLOOD_COLORS = ['#1a1a2e', '#16213e', '#0f3460', '#e94560', '#ff6b6b']  # Dark to hot
TERRAIN_CMAP = LinearSegmentedColormap.from_list('terrain_pro',
    ['#2d1b0e', '#5c4033', '#8B7355', '#90EE90', '#228B22', '#006400'], N=256)

def load_model_and_predict(graph_path):
    """Load best model and run inference on a graph."""
    model = get_gnn_model(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    data = torch.load(graph_path, weights_only=False).to(DEVICE)
    with torch.no_grad():
        depth, velocity = model(data.x, data.edge_index, data.edge_attr)

    pos = data.pos.cpu().numpy()
    depth = depth.cpu().numpy().flatten()
    velocity = velocity.cpu().numpy().flatten()
    target = data.y.cpu().numpy().flatten() if hasattr(data, 'y') and data.y is not None else None

    return pos, depth, velocity, target

def render_3d_flood_prediction(pos, depth, name, output_path):
    """Cinematic 3D scatter plot of terrain with flood depth overlay."""
    fig = plt.figure(figsize=(20, 14), facecolor='#0a0a1a')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a1a')

    # Subsample for rendering performance
    n = len(pos)
    if n > 500000:
        idx = np.random.choice(n, 500000, replace=False)
    else:
        idx = np.arange(n)

    x, y, z = pos[idx, 0], pos[idx, 1], pos[idx, 2]
    d = depth[idx]

    # Normalize coordinates for better viewing
    x = x - x.mean()
    y = y - y.mean()

    sc = ax.scatter(x, y, z, c=d, cmap='turbo', s=0.5, alpha=0.8,
                    vmin=0, vmax=np.percentile(d, 95))

    ax.set_xlabel('X (m)', color='white', fontsize=10)
    ax.set_ylabel('Y (m)', color='white', fontsize=10)
    ax.set_zlabel('Elevation (m)', color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=8)

    # Dark styling
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#333333')
    ax.yaxis.pane.set_edgecolor('#333333')
    ax.zaxis.pane.set_edgecolor('#333333')
    ax.grid(True, alpha=0.15, color='white')

    ax.view_init(elev=35, azim=45)

    cb = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.1, label='Predicted Flood Depth')
    cb.ax.yaxis.label.set_color('white')
    cb.ax.tick_params(colors='white')

    ax.set_title(f'Flood Risk Prediction: {name}',
                 color='#00d4ff', fontsize=16, fontweight='bold', pad=20)

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] 3D Render -> {output_path.name}")

def render_overhead_heatmap(pos, depth, name, output_path):
    """Top-down flood risk heatmap with contours."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 14), facecolor='#0a0a1a')
    ax.set_facecolor('#0a0a1a')

    x = pos[:, 0] - pos[:, 0].mean()
    y = pos[:, 1] - pos[:, 1].mean()

    sc = ax.scatter(x, y, c=depth, cmap='turbo', s=0.3, alpha=0.9,
                    vmin=0, vmax=np.percentile(depth, 95))

    ax.set_xlabel('X (m)', color='white', fontsize=12)
    ax.set_ylabel('Y (m)', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.set_aspect('equal')

    cb = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label('Flood Risk (Normalized Depth)', color='white', fontsize=12)
    cb.ax.tick_params(colors='white')

    ax.set_title(f'Overhead Flood Risk Heatmap: {name}',
                 color='#ff6b6b', fontsize=16, fontweight='bold')

    # Add risk zone annotations
    high_risk = depth > np.percentile(depth, 90)
    if high_risk.sum() > 0:
        ax.scatter(x[high_risk], y[high_risk], c='red', s=1, alpha=0.3, marker='x', label='Critical Zone')
        ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white', fontsize=10)

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Heatmap -> {output_path.name}")

def render_training_dashboard(output_path):
    """Training metrics dashboard: loss curves + confusion matrix + scatter."""
    history_path = METRICS_DIR / "training_history.json"
    cm_path = METRICS_DIR / "confusion_matrix.npy"
    preds_path = METRICS_DIR / "final_predictions.npy"
    targets_path = METRICS_DIR / "final_targets.npy"

    if not history_path.exists():
        print("  [SKIP] No training_history.json found")
        return

    with open(history_path) as f:
        history = json.load(f)

    fig = plt.figure(figsize=(24, 16), facecolor='#0a0a1a')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # --- Panel 1: Training & Validation Loss ---
    ax1 = fig.add_subplot(gs[0, 0], facecolor='#1a1a2e')
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], color='#00d4ff', linewidth=2, label='Train Loss', alpha=0.9)
    ax1.plot(epochs, history['val_loss'], color='#ff6b6b', linewidth=2, label='Val Loss', alpha=0.9)
    ax1.fill_between(epochs, history['train_loss'], alpha=0.1, color='#00d4ff')
    ax1.fill_between(epochs, history['val_loss'], alpha=0.1, color='#ff6b6b')
    ax1.set_xlabel('Epoch', color='white')
    ax1.set_ylabel('MSE Loss', color='white')
    ax1.set_title('Training & Validation Loss', color='#00d4ff', fontsize=14, fontweight='bold')
    ax1.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.15, color='white')
    ax1.set_yscale('log')

    # --- Panel 2: Learning Rate Schedule ---
    ax2 = fig.add_subplot(gs[0, 1], facecolor='#1a1a2e')
    ax2.plot(epochs, history['lr'], color='#ffd700', linewidth=2)
    ax2.fill_between(epochs, history['lr'], alpha=0.15, color='#ffd700')
    ax2.set_xlabel('Epoch', color='white')
    ax2.set_ylabel('Learning Rate', color='white')
    ax2.set_title('Cosine Annealing LR Schedule', color='#ffd700', fontsize=14, fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.15, color='white')

    # --- Panel 3: Metrics Summary ---
    ax3 = fig.add_subplot(gs[0, 2], facecolor='#1a1a2e')
    metrics = history.get('final_metrics', {})
    if metrics:
        metric_names = ['MSE', 'RMSE', 'MAE', 'R²', 'Accuracy']
        metric_vals = [metrics['mse'], metrics['rmse'], metrics['mae'],
                       metrics['r2'], metrics['overall_accuracy']]
        colors = ['#00d4ff', '#00d4ff', '#00d4ff', '#4ade80', '#4ade80']

        bars = ax3.barh(metric_names, metric_vals, color=colors, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, metric_vals):
            ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{val:.4f}', va='center', color='white', fontweight='bold')

    ax3.set_title('Final Model Metrics', color='#4ade80', fontsize=14, fontweight='bold')
    ax3.tick_params(colors='white')
    ax3.set_xlim(0, 1.2)
    ax3.grid(True, alpha=0.15, color='white', axis='x')

    # --- Panel 4: Confusion Matrix ---
    ax4 = fig.add_subplot(gs[1, 0], facecolor='#1a1a2e')
    if cm_path.exists():
        cm = np.load(cm_path)
        # Normalize for display
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

        im = ax4.imshow(cm_norm, interpolation='nearest', cmap='YlOrRd', vmin=0, vmax=1)
        ax4.set_xticks(range(len(DEPTH_LABELS)))
        ax4.set_yticks(range(len(DEPTH_LABELS)))
        ax4.set_xticklabels(DEPTH_LABELS, color='white', rotation=45, fontsize=9)
        ax4.set_yticklabels(DEPTH_LABELS, color='white', fontsize=9)

        # Annotate cells
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'white' if cm_norm[i, j] > 0.5 else 'black'
                ax4.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                         color=color, fontsize=8, fontweight='bold')

        ax4.set_xlabel('Predicted', color='white', fontsize=11)
        ax4.set_ylabel('True', color='white', fontsize=11)
        cb = plt.colorbar(im, ax=ax4, shrink=0.8)
        cb.ax.tick_params(colors='white')
        cb.set_label('Normalized', color='white')

    ax4.set_title('Confusion Matrix (Flood Risk Classes)', color='#ff6b6b', fontsize=14, fontweight='bold')

    # --- Panel 5: Prediction vs Target Scatter ---
    ax5 = fig.add_subplot(gs[1, 1], facecolor='#1a1a2e')
    if preds_path.exists() and targets_path.exists():
        preds = np.load(preds_path).flatten()
        targets = np.load(targets_path).flatten()
        # Subsample for plotting
        n = len(preds)
        if n > 50000:
            idx = np.random.choice(n, 50000, replace=False)
            preds_s, targets_s = preds[idx], targets[idx]
        else:
            preds_s, targets_s = preds, targets

        ax5.scatter(targets_s, preds_s, c='#00d4ff', s=0.5, alpha=0.3)
        ax5.plot([0, 1], [0, 1], 'r--', linewidth=2, alpha=0.8, label='Perfect Prediction')
        ax5.set_xlabel('True Depth', color='white')
        ax5.set_ylabel('Predicted Depth', color='white')
        ax5.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')

    ax5.set_title('Predicted vs True Depth', color='#00d4ff', fontsize=14, fontweight='bold')
    ax5.tick_params(colors='white')
    ax5.grid(True, alpha=0.15, color='white')

    # --- Panel 6: Per-Class Accuracy ---
    ax6 = fig.add_subplot(gs[1, 2], facecolor='#1a1a2e')
    if metrics and 'per_class_accuracy' in metrics:
        colors_bar = ['#4ade80', '#fbbf24', '#fb923c', '#f87171', '#ef4444']
        bars = ax6.bar(DEPTH_LABELS, metrics['per_class_accuracy'],
                       color=colors_bar[:len(DEPTH_LABELS)], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, metrics['per_class_accuracy']):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.2f}', ha='center', color='white', fontweight='bold')

    ax6.set_title('Per-Class Accuracy', color='#4ade80', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Accuracy', color='white')
    ax6.tick_params(colors='white')
    ax6.set_ylim(0, 1.15)
    ax6.grid(True, alpha=0.15, color='white', axis='y')

    # Main title
    best_epoch = history.get('best_epoch', '?')
    fig.suptitle(f'GOD-TIER V2 — DUALFloodGNN Training Dashboard (Best @ Epoch {best_epoch})',
                 color='white', fontsize=20, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Dashboard -> {output_path.name}")

def render_multi_village_comparison(output_path):
    """Side-by-side prediction comparison for all villages."""
    if not MODEL_PATH.exists():
        return

    files = sorted(GRAPH_DIR.glob("*.pt"))
    if not files:
        return

    n = min(len(files), 6)  # Max 6 villages
    fig, axes = plt.subplots(2, 3, figsize=(24, 16), facecolor='#0a0a1a')

    for i in range(6):
        row, col = i // 3, i % 3
        ax = axes[row][col]
        ax.set_facecolor('#0a0a1a')

        if i < n:
            try:
                pos, depth, _, _ = load_model_and_predict(files[i])
                name = files[i].stem.split('_')[0]

                # Subsample for rendering
                if len(pos) > 200000:
                    idx = np.random.choice(len(pos), 200000, replace=False)
                else:
                    idx = np.arange(len(pos))

                x = pos[idx, 0] - pos[idx, 0].mean()
                y = pos[idx, 1] - pos[idx, 1].mean()

                sc = ax.scatter(x, y, c=depth[idx], cmap='turbo', s=0.3, alpha=0.9,
                                vmin=0, vmax=np.percentile(depth, 95))
                ax.set_title(name, color='white', fontsize=12, fontweight='bold')
                ax.set_aspect('equal')
                ax.tick_params(colors='white', labelsize=7)

                del pos, depth
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {e}', transform=ax.transAxes,
                        ha='center', color='red', fontsize=9)
        else:
            ax.set_visible(False)

    fig.suptitle('GOD-TIER V2 — Multi-Village Flood Risk Comparison',
                 color='#00d4ff', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Multi-Village -> {output_path.name}")


def vis():
    print("=" * 70)
    print("  GOD-TIER V2 CINEMATIC VISUALIZER (BEAST MODE)")
    print("=" * 70)

    if not MODEL_PATH.exists():
        print("[ERROR] No trained model found!")
        return

    files = sorted(GRAPH_DIR.glob("*.pt"))
    if not files:
        print("[ERROR] No graph files found!")
        return

    print(f"  Model: {MODEL_PATH}")
    print(f"  Graphs: {len(files)} files")
    print(f"  Output: {OUTPUT_DIR}\n")

    # 1. TRAINING DASHBOARD (confusion matrix + loss curves + metrics)
    print("[1/4] Training Dashboard...")
    try:
        render_training_dashboard(OUTPUT_DIR / "training_dashboard.png")
    except Exception as e:
        print(f"  [WARN] Dashboard failed: {e}")
        traceback.print_exc()

    # 2. 3D FLOOD PREDICTION (first village)
    print("[2/4] 3D Flood Prediction Render...")
    try:
        pos, depth, vel, target = load_model_and_predict(files[0])
        name = files[0].stem
        render_3d_flood_prediction(pos, depth, name, OUTPUT_DIR / "flood_3d_prediction.png")
    except Exception as e:
        print(f"  [WARN] 3D render failed: {e}")
        traceback.print_exc()

    # 3. OVERHEAD HEATMAP (first village)
    print("[3/4] Overhead Risk Heatmap...")
    try:
        render_overhead_heatmap(pos, depth, name, OUTPUT_DIR / "flood_overhead_heatmap.png")
    except Exception as e:
        print(f"  [WARN] Heatmap failed: {e}")
        traceback.print_exc()

    # 4. MULTI-VILLAGE COMPARISON
    print("[4/4] Multi-Village Comparison...")
    try:
        render_multi_village_comparison(OUTPUT_DIR / "multi_village_comparison.png")
    except Exception as e:
        print(f"  [WARN] Multi-village render failed: {e}")
        traceback.print_exc()

    # BONUS: Try PyVista 3D render if available
    try:
        import pyvista as pv
        print("\n[BONUS] PyVista Cinematic Render...")
        pos, depth, _, _ = load_model_and_predict(files[0])

        if len(pos) > 500000:
            idx = np.random.choice(len(pos), 500000, replace=False)
            pos_sub, depth_sub = pos[idx], depth[idx]
        else:
            pos_sub, depth_sub = pos, depth

        cloud = pv.PolyData(pos_sub)
        cloud['Flood Depth'] = depth_sub

        pl = pv.Plotter(off_screen=True, window_size=[1920, 1080])
        pl.set_background('#0a0a1a', top='#1a1a3e')
        pl.enable_eye_dome_lighting()

        pl.add_mesh(cloud, scalars='Flood Depth', cmap='turbo', point_size=3,
                    render_points_as_spheres=True, opacity=0.9,
                    scalar_bar_args={'title': 'Predicted Flood Depth',
                                     'color': 'white', 'vertical': True})

        pl.add_text("GOD-TIER V2: DUALFloodGNN Prediction", position='upper_left',
                     color='white', font_size=14)

        pl.camera.azimuth = 45
        pl.camera.elevation = 30
        pl.camera.zoom(1.2)
        pl.screenshot(OUTPUT_DIR / "pyvista_cinematic.png")

        # Orbital GIF
        print("  Generating orbital flyover GIF...")
        path = pl.generate_orbital_path(n_points=90, shift=0.0)
        pl.open_gif(str(OUTPUT_DIR / "flood_flyover_orbit.gif"))
        pl.orbit_on_path(path, write_frames=True)
        pl.close()
        print(f"  [OK] PyVista -> pyvista_cinematic.png, flood_flyover_orbit.gif")

    except ImportError:
        print("  [SKIP] PyVista not available, using matplotlib renders only")
    except Exception as e:
        print(f"  [WARN] PyVista render failed: {e}")
        traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"  VISUALIZATION COMPLETE — Check {OUTPUT_DIR}")
    print(f"{'='*70}")

if __name__ == "__main__":
    vis()
