"""
Quick Diagnostic: Run this on the WORKSTATION (E:\DTM\Data\GodTier)
to check what phase the mission is stuck on.
"""
import json
from pathlib import Path
import os
import time

print("=" * 60)
print("  WORKSTATION DIAGNOSTIC PROBE")
print("=" * 60)

# 1. Check mission state
state_file = Path("mission_state_v2.json")
if state_file.exists():
    with open(state_file) as f:
        state = json.load(f)
    print(f"\nMission State: {json.dumps(state, indent=2)}")
else:
    print("\n[!] No mission_state_v2.json found")

# 2. Check mission log (last 20 lines)
log_file = Path("mission_control_v2.log")
if log_file.exists():
    lines = log_file.read_text(encoding='utf-8', errors='replace').strip().split('\n')
    print(f"\nMission Log (last 20 lines of {len(lines)} total):")
    for line in lines[-20:]:
        print(f"  {line}")
else:
    print("\n[!] No mission_control_v2.log found")

# 3. Check data directories
print("\n--- DATA DIRECTORIES ---")
dirs_to_check = [
    ("Input Point Clouds", Path("../Point-Cloud")),
    ("V2 Ingested Clouds", Path("Processed_Data/GodTier_V2/Clouds")),
    ("V2 Graphs", Path("Processed_Data/GodTier_V2/Graphs")),
    ("V2 Models", Path("Models/GodTier_V2")),
    ("V2 Visuals", Path("Visuals/GodTier_V2")),
]

for label, d in dirs_to_check:
    if d.exists():
        files = list(d.iterdir())
        total_size = sum(f.stat().st_size for f in files if f.is_file()) / (1024*1024)
        print(f"  {label}: {d} -> {len(files)} items, {total_size:.1f} MB")
        # Show first 3 files
        for f in sorted(files)[:3]:
            if f.is_file():
                sz = f.stat().st_size / (1024*1024)
                mod = time.strftime('%H:%M:%S', time.localtime(f.stat().st_mtime))
                print(f"    {f.name} ({sz:.1f} MB, modified {mod})")
        if len(files) > 3:
            print(f"    ... and {len(files)-3} more")
    else:
        print(f"  {label}: {d} -> DOES NOT EXIST")

# 4. Check running processes
print("\n--- RUNNING PYTHON PROCESSES ---")
try:
    import subprocess
    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                          capture_output=True, text=True)
    print(result.stdout)
except:
    print("  Could not check processes")

print("\n" + "=" * 60)
print("  DIAGNOSIS:")
print("=" * 60)

# Auto-diagnose
clouds_dir = Path("Processed_Data/GodTier_V2/Clouds")
graphs_dir = Path("Processed_Data/GodTier_V2/Graphs")

if not clouds_dir.exists() or not list(clouds_dir.glob("*.laz")):
    print("  -> Phase 2 (Ingestion) has NOT completed yet.")
    print("     The script is still extracting and filtering point clouds.")
    print("     This is the SLOW part — SOR filtering on raw data.")
    print("     With large ZIP files, this can take 4-8 hours.")
    print("\n  RECOMMENDATION: Let it finish. Check if python is still running.")
elif not graphs_dir.exists() or not list(graphs_dir.glob("*.pt")):
    print("  -> Phase 2 done, Phase 3 (Graph) is running or hasn't started.")
    print("     Graph construction with cKDTree should be fast (~minutes).")
else:
    n_clouds = len(list(clouds_dir.glob("*.laz")))
    n_graphs = len(list(graphs_dir.glob("*.pt")))
    print(f"  -> Clouds: {n_clouds} | Graphs: {n_graphs}")
    if n_graphs < n_clouds:
        print("     Phase 3 (Graph) is in progress.")
    else:
        print("     Phase 3 done. Check if training (Phase 4) started.")
