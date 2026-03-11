import pyvista as pv
import pandas as pd
import numpy as np
from pathlib import Path
import imageio
import traceback

OUTPUT_IMAGE = "DigitalTwin_GodTier_Heatmap.png"
OUTPUT_GIF = "DigitalTwin_GodTier_RisingFlood.gif"
OUTPUT_SPLIT = "DigitalTwin_GodTier_SplitScreen.gif"

def run_vis():
    print("🚀 Rendering Hollywood-Grade Digital Twin...")
    try:
        # Load Data
        df = pd.read_csv("Niagara_Manifest.csv")

        # 1. Create Terrain Mesh (The Village)
        las_files = list(Path("Processed_Data/FullScale").glob("*.laz"))
        terrain_mesh = None
        if las_files:
            try:
                reader = pv.get_reader(str(las_files[0]))
                terrain_mesh = reader.read()
            except: pass

        # 2. Create Flood Mesh (The Water)
        water_df = df[df['Depth'] > 0.1]
        water_points = water_df[['X', 'Y', 'Z']].values
        water_cloud = pv.PolyData(water_points)
        water_cloud['Depth'] = water_df['Depth'].values

        # --- SCENE 1: The High-Res Screenshot (4K) ---
        pl = pv.Plotter(off_screen=True, window_size=[3840, 2160])
        pl.set_background('black', top='midnightblue')
        pl.enable_eye_dome_lighting() 
        
        if terrain_mesh is not None:
            if 'Red' in terrain_mesh.array_names:
                pl.add_mesh(terrain_mesh, rgb=True, point_size=2)
            else:
                pl.add_mesh(terrain_mesh, color='#8B4513', point_size=2)

        pl.add_mesh(water_cloud, scalars='Depth', cmap='turbo', 
                    point_size=5, opacity=0.8, render_points_as_spheres=True, clim=[0, 5])
        
        pl.view_isometric()
        pl.camera.zoom(1.3)
        pl.screenshot(OUTPUT_IMAGE)
        print(f"✅ Saved 4K Still: {OUTPUT_IMAGE}")

        # --- SCENE 2: The Rising Flood Animation ---
        print("🎥 Rendering Rising Flood Animation...")
        pl = pv.Plotter(off_screen=True, window_size=[1920, 1080])
        pl.set_background('black', top='#1a1a2e')
        pl.enable_eye_dome_lighting()
        
        if terrain_mesh is not None:
             if 'Red' in terrain_mesh.array_names:
                pl.add_mesh(terrain_mesh, rgb=True, point_size=2)
             else:
                pl.add_mesh(terrain_mesh, color='#555555', point_size=2)
        
        pl.add_mesh(water_cloud, scalars='Depth', cmap='turbo', point_size=3, opacity=0.8, clim=[0,5])

        pl.view_isometric()
        pl.camera.zoom(1.2)
        
        pl.open_gif(OUTPUT_GIF)
        path = pl.generate_orbital_path(n_points=60, shift=100)
        for pos in path.points:
            pl.camera.position = pos
            pl.write_frame()
        pl.close()
        print(f"✅ Saved Animation: {OUTPUT_GIF}")

        # --- SCENE 3: The Split-Screen Comparison (Before vs After) ---
        print("🎥 Rendering Split-Screen Comparison...")
        pl = pv.Plotter(off_screen=True, result_layout='1x2', window_size=[3840, 1080])
        
        # Left Viewport (Before - Dry)
        pl.subplot(0, 0)
        pl.add_text("BEFORE: Current State", position='upper_left', font_size=20, color='white')
        pl.set_background('black')
        pl.enable_eye_dome_lighting()
        if terrain_mesh is not None:
             if 'Red' in terrain_mesh.array_names:
                pl.add_mesh(terrain_mesh, rgb=True, point_size=2)
             else:
                pl.add_mesh(terrain_mesh, color='#8B4513', point_size=2)
        pl.view_isometric()
        pl.camera.zoom(1.2)

        # Right Viewport (After - Flooded)
        pl.subplot(0, 1)
        pl.add_text("AFTER: AI Prediction (God Tier)", position='upper_left', font_size=20, color='red')
        pl.set_background('black', top='midnightblue')
        pl.enable_eye_dome_lighting()
        if terrain_mesh is not None:
             if 'Red' in terrain_mesh.array_names:
                pl.add_mesh(terrain_mesh, rgb=True, point_size=2)
             else:
                pl.add_mesh(terrain_mesh, color='#333333', point_size=2) # Darker terrain to make water pop
        pl.add_mesh(water_cloud, scalars='Depth', cmap='turbo', point_size=4, opacity=0.9, clim=[0, 5])
        pl.view_isometric()
        pl.camera.zoom(1.2)

        pl.link_views()
        pl.open_gif(OUTPUT_SPLIT)
        # But moving camera of subplot(0,1) should move (0,0) if linked.
        
        path = pl.generate_orbital_path(n_points=60, shift=100)
        for i, pos in enumerate(path.points):
            # For linked views, we only need to move one camera if correctly linked, 
            # but PyVista sometimes needs explicit updates.
            pl.subplot(0, 0)
            pl.camera.position = pos
            pl.subplot(0, 1)
            pl.camera.position = pos
            
            pl.write_frame()
        pl.close()
        print(f"✅ Saved Split-Screen: {OUTPUT_SPLIT}")

    except Exception as e:
        print("❌ Visualization Crash:")
        traceback.print_exc()

if __name__ == "__main__":
    run_vis()
