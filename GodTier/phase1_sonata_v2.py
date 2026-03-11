import sys
import os
from pathlib import Path

def run_sonata_phase():
    print("[SONATA] Checking for Semantic Terrain Understanding...")
    model_path = Path("sonata_epoch_15.pth")
    if model_path.exists():
        print(f"[SONATA] Found pre-trained model: {model_path.name}")
        print("[SONATA] Using this model to filter vegetation in Ingestion phase.")
    else:
        print("[SONATA] No model found. Skipping semantic enhancement (using raw geometric filtering).")

if __name__ == "__main__":
    run_sonata_phase()
