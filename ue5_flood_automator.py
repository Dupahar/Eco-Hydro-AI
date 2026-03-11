import unreal
import csv
import os

# --- CONFIGURATION (AUTO-DETECTED FROM YOUR TREE) ---
# 1. The CSV is in the ROOT 'DTM' folder
CSV_PATH = r"C:\Users\mahaj\Downloads\DTM\Niagara_Manifest.csv"

# 2. The Point Cloud is inside Processed_Data/UP
LAS_PATH = r"C:\Users\mahaj\Downloads\DTM\Processed_Data\UP\118129_Manjhola Khurd_POINT CLOUD_tile_13.laz"

# 3. GLOBAL OFFSET (From your export_inference.py output)
# These are the values needed to align the Point Cloud to the Water
OFFSET_X = -34.34
OFFSET_Y = 38.74  # Inverted Y for UE5
OFFSET_Z = -6.21

def create_flood_material():
    """Creates a simple glowing water material programmatically."""
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    
    mat_name = "M_AutoFlood"
    if unreal.EditorAssetLibrary.does_asset_exist(f"/Game/{mat_name}"):
        return unreal.EditorAssetLibrary.load_asset(f"/Game/{mat_name}")

    material = asset_tools.create_asset(mat_name, "/Game/", unreal.Material, unreal.MaterialFactoryNew())
    material.set_editor_property("blend_mode", unreal.BlendMode.BLEND_TRANSLUCENT)
    
    base_color = unreal.MaterialEditingLibrary.create_material_expression(material, unreal.MaterialExpressionConstant3Vector)
    base_color.constant = unreal.LinearColor(0.0, 0.2, 1.0, 1.0) 
    
    opacity = unreal.MaterialEditingLibrary.create_material_expression(material, unreal.MaterialExpressionScalarParameter)
    opacity.parameter_name = "Opacity"
    opacity.default_value = 0.8
    
    unreal.MaterialEditingLibrary.connect_to_material_property(base_color, "Constant", unreal.MaterialProperty.MP_BASE_COLOR)
    unreal.MaterialEditingLibrary.connect_to_material_property(opacity, "Output", unreal.MaterialProperty.MP_OPACITY)
    
    unreal.MaterialEditingLibrary.recompile_material(material)
    return material

def spawn_flood_actor(material, csv_file):
    """Spawns visualizer spheres from CSV data."""
    editor_actor_sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    
    # Clean up old actors
    found_actors = unreal.GameplayStatics.get_all_actors_with_tag(unreal.EditorLevelLibrary.get_editor_world(), "FloodVis")
    for old_actor in found_actors:
        unreal.EditorLevelLibrary.destroy_actor(old_actor)

    # Spawn new actor
    actor = editor_actor_sub.spawn_actor_from_class(unreal.Actor, unreal.Vector(0,0,0))
    actor.set_actor_label("AI_Flood_Visualizer")
    actor.tags = ["FloodVis"]
    
    ism_comp = actor.add_component_by_class(unreal.InstancedStaticMeshComponent, "FloodParticles")
    sphere_mesh = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Sphere")
    ism_comp.set_static_mesh(sphere_mesh)
    ism_comp.set_material(0, material)
    
    transforms = []
    print(f"Reading Flood Manifest from {csv_file}...")
    
    if not os.path.exists(csv_file):
        print(f"CRITICAL: CSV not found at {csv_file}")
        return

    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    depth = float(row['Depth'])
                    if depth > 0.05: # Optimization threshold
                        transforms.append(unreal.Transform(
                            location=unreal.Vector(float(row['X']), float(row['Y']), float(row['Z'])),
                            rotation=unreal.Rotator(0,0,0),
                            scale=unreal.Vector(depth * 0.5, depth * 0.5, depth * 0.5)
                        ))
                except ValueError:
                    continue
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if transforms:
        print(f"Spawning {len(transforms)} water particles...")
        chunk_size = 50000
        for i in range(0, len(transforms), chunk_size):
            ism_comp.add_instances(transforms[i:i+chunk_size], False)
        print("✅ Flood Visualization Complete!")
    else:
        print("Warning: CSV empty or no flood depth > 5cm found.")

def import_lidar(las_file):
    """Imports Point Cloud."""
    lidar_factory = unreal.LidarPointCloudFactory()
    task = unreal.AssetImportTask()
    task.filename = las_file
    task.destination_path = "/Game/"
    task.automated = True
    task.replace_existing = True
    task.factory = lidar_factory
    
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    
    asset_name = os.path.splitext(os.path.basename(las_file))[0]
    cloud_asset = unreal.EditorAssetLibrary.load_asset(f"/Game/{asset_name}")
    
    if cloud_asset:
        editor_actor_sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        
        # Avoid duplicate spawns
        found_clouds = unreal.GameplayStatics.get_all_actors_with_tag(unreal.EditorLevelLibrary.get_editor_world(), "VillageCloud")
        if not found_clouds:
            lidar_actor = editor_actor_sub.spawn_actor_from_object(cloud_asset, unreal.Vector(0,0,0))
            lidar_actor.set_actor_label("Village_PointCloud")
            lidar_actor.tags = ["VillageCloud"]
            print(f"✅ Imported Village: {asset_name}")
            
            # Print Alignment Instructions
            print("\n" + "="*50)
            print("🚨 ALIGNMENT REQUIRED 🚨")
            print("Select 'Village_PointCloud' in the Outliner.")
            print("Set its Location Transform to:")
            print(f"  X: {OFFSET_X}")
            print(f"  Y: {OFFSET_Y}")
            print(f"  Z: {OFFSET_Z}")
            print("="*50 + "\n")

if __name__ == "__main__":
    print("--- Starting Eco-Hydro-AI Visualization ---")
    flood_mat = create_flood_material()
    if os.path.exists(LAS_PATH): import_lidar(LAS_PATH)
    else: print(f"ERROR: LAS missing at {LAS_PATH}")
    spawn_flood_actor(flood_mat, CSV_PATH)