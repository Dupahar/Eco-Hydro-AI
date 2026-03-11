# God-Tier Mission: Migration & Deployment Guide

This guide details how to move the God-Tier pipeline to your new high-spec PC and execute it successfully.

## 1. Prerequisites on New PC
Ensure the new machine has:
*   **OS:** Windows 10/11 (or Linux)
*   **Python:** 3.9 or higher installed.
*   **CUDA (Optional but Recommended):** if you have an NVIDIA GPU, install the CUDA Toolkit to accelerate PyTorch training.

## 2. Transfer Files
Copy the entire `DTM` folder (or just the `Data/GodTier` folder and data) to the new PC.
**Key Structure Required:**
```
DTM/
├── Data/
│   ├── Point-Cloud/          <-- Put .zip or .las files here
│   └── GodTier/
│       ├── run_god_tier_mission.py
│       ├── requirements.txt
│       └── ...
```

## 3. Clean Setup (Important)
On the new PC, it is best to start with a fresh environment to avoid conflicts.

### Create Environment
```powershell
# If using Conda (Recommended for Data Science)
conda create -n godtier python=3.10
conda activate godtier

# OR if using standard Python
python -m venv venv
.\venv\Scripts\activate
```

### Install Dependencies
We have removed the problematic `pdal` dependency. The new pipeline uses `laspy` + `scipy` which is much easier to install.
```powershell
cd DTM\Data\GodTier
pip install -r requirements.txt
```
*Note: If you have a GPU, install the specific PyTorch-CUDA version from [pytorch.org](https://pytorch.org/) manually if `requirements.txt` installs the CPU version.*

## 4. Run the Mission
Execute the main script. It will auto-generate all necessary sub-scripts.
```powershell
python run_god_tier_mission.py
```

## 5. Debugging & What to Expect
### "It seems stuck at Ingestion"
*   **Cause:** The script is unzipping massive files (1.5GB - 6GB) and processing millions of points.
*   **Fix:** Open `Task Manager` and check if `python.exe` is using CPU/RAM. If it's moving, let it run. It can take 10-20 minutes per file depending on the CPU.

### "Out of Memory"
*   **Cause:** The default `scipy.cKDTree` is memory-intensive.
*   **Fix:** The script currently sets `pool.map` to use `cpu_count() // 4` processes. If you crash, edit `run_god_tier_mission.py` inside `generate_ingest_script`:
    ```python
    # Reduce this number to 1 or 2 if running out of RAM
    with multiprocessing.Pool(processes=2) as pool:
    ```

## 6. Verification
After the mission completes, check:
1.  **Ingestion:** `Processed_Data/FullScale/` should have `.laz` files.
2.  **Graphs:** `Processed_Data/Graphs_Full/` should have `.pt` files.
3.  **Visualization:** `DigitalTwin_GodTier_Heatmap.png` should be created.
