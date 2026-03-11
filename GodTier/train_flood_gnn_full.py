import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from flood_gnn_model import get_gnn_model
import glob
from pathlib import Path
import traceback

EPOCHS = 50
LR = 0.005
GRAPH_DIR = Path("Processed_Data/Graphs_Full")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_god_tier():
    print(f"🚀 Training on {DEVICE}...")
    try:
        files = list(GRAPH_DIR.glob("*.pt"))
        dataset = []
        for f in files:
            try:
                d = torch.load(f)
                if not hasattr(d, 'y'):
                    d.y = (d.pos[:, 2].max() - d.pos[:, 2]) / d.pos[:, 2].max()
                    d.y = d.y.unsqueeze(1)
                dataset.append(d)
            except: pass
            
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        model = get_gnn_model(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = torch.nn.MSELoss()
        
        for epoch in range(1, EPOCHS+1):
            model.train()
            total_loss = 0
            for data in loader:
                data = data.to(DEVICE)
                optimizer.zero_grad()
                pred, _ = model(data.x, data.edge_index, data.edge_attr)
                loss = criterion(pred, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f}")
            
        torch.save(model.state_dict(), "flood_gnn_godtier_final.pth")
        print("✅ Training Complete.")
    except Exception as e:
        print("❌ Training Crash:")
        traceback.print_exc()

if __name__ == "__main__":
    train_god_tier()
