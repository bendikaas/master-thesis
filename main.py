import Floe as f
import FloePlot as fplt
import os

intrinsics_path = r"YOUR OWN PATH TO THE INTRINSICS FILE CALLED intrinsics.yaml"
workspace = r"YOUR OWN PATH TO THE FOLDER CONTAINING THE REAL DATA SET"

intrinsics_path = r"C:\Users\bendi\Documents\Master\Bendik dataset\IceDrift\calib\intrinsics.yaml"
workspace = r"C:\Users\bendi\Documents\Master\data set"

# SCENARIO 1. to predict the path, create a larger end_id. Remember that the end_id must be a valid folder
start_id = 1726069689 # the same timestep that the satelite image was captured 
end_id = 1726069764

# creates a list of of the folders from start_id to end_id
paths = [
    os.path.join(workspace, str(folder_id))
    for folder_id in range(start_id, end_id + 1, 5)
    if os.path.isdir(os.path.join(workspace, str(folder_id)))
]

floe_manager = f.Floe_Manager()
w, h, intrinsic_matrix, r_cam = floe_manager.get_intrinsics(intrinsics_path)

for path in paths:
    print("curr path", path)
    curr_raster = floe_manager.create_main_raster(path, w, h, intrinsic_matrix, r_cam)
    # fplt.show_raster(curr_raster) # vizualize the image of the main_raster
    floe_manager.predict_floes()
    floe_manager.associate(curr_raster)
    floe_manager.correct_floes()

fplt.plot_paths(floe_manager)
fplt.plot_floe_paths_with_uncertainties(floe_manager)
