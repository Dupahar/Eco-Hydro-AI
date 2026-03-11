import torch
import pandas as pd
import numpy as np
from pathlib import Path
from flood_gnn_model import get_gnn_model
import traceback

GRAPH_DIR = Path("Processed_Data/Graphs_Full")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def export_predictions():
    print("🚀 Exporting Inference Results...")
    try:
        model = get_gnn_model(DEVICE)
        model.load_state_dict(torch.load("flood_gnn_godtier_final.pth"))
        model.eval()
        
        all_pos = []
        all_depth = []
        
        for f in list(GRAPH_DIR.glob("*.pt")):
            try:
                data = torch.load(f).to(DEVICE)
                with torch.no_grad():
                    pred, _ = model(data.x, data.edge_index, data.edge_attr)
                
                all_pos.append(data.pos.cpu().numpy())
                all_depth.append(pred.cpu().numpy())
                print(f"Processed {f.name}")
            except Exception as e: 
                print(f"Error file {f}: {e}")

        if not all_pos: return

        full_pos = np.vstack(all_pos)
        full_depth = np.vstack(all_depth)
        
        df = pd.DataFrame(full_pos, columns=['X', 'Y', 'Z'])
        df['Depth'] = full_depth
        df.to_csv("Niagara_Manifest.csv", index=False)
        print("✅ Exported Niagara_Manifest.csv")
    except Exception as e:
        print("❌ Export Fail:")
        traceback.print_exc()

if __name__ == "__main__":
    export_predictions()
