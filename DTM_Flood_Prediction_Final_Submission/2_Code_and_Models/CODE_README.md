# Code Structure

## Training Pipeline
1. `generate_flood_labels_QUICK_FIXED.py` - Generate HAND labels
2. `train_MEMORY_EFFICIENT.py` - Train the GNN model
3. `confusion_matrix_analysis.py` - Evaluate performance

## Visualization
- `visualize_god_tier_FIXED.py` - Python 4K visualizations
- `visualize_with_orthophoto.py` - Orthophoto overlays
- Java files - Professional desktop application

## Models
- `best_model.pt` - Use this for production
- `model_final.pt` - Final epoch checkpoint

## Setup
```bash
pip install -r requirements.txt
python train_MEMORY_EFFICIENT.py
```

See individual file headers for detailed documentation.
