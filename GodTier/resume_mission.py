"""
RESUME HELPER — Run this ONCE before relaunching the mission.
Ensures mission_state_v2.json correctly marks completed phases.
"""
import json
from pathlib import Path

STATE_FILE = "mission_state_v2.json"

# These phases are CONFIRMED complete from directory inspection
COMPLETED_PHASES = [
    "Phase 1: Sonata Logic",
    "Phase 2: High-Perf Ingestion"
]

state = {"completed_steps": COMPLETED_PHASES}

with open(STATE_FILE, 'w') as f:
    json.dump(state, f, indent=2)

print("Mission state reset. Completed phases:")
for p in COMPLETED_PHASES:
    print(f"  [x] {p}")
print("\nPhase 3 (Graph) will RESUME — skipping already-built .pt files.")
print("Phase 4 (Training) and Phase 5 (Vis) will run fresh.")
print("\nNow run:  python run_god_tier_mission_v2.py")
