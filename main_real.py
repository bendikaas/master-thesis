import Floe as f
import FloePlot as fplt
import os

intrinsics_path = r"C:\Users\bendi\Documents\Master\Bendik dataset\IceDrift\calib\int.yaml"
workspace = r"C:\Users\bendi\Documents\Master\final data set"

start_id = 1726069689
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
    curr_raster = floe_manager.create_main_raster(path, w, h, intrinsic_matrix, r_cam)
    # fplt.show_raster(curr_raster)
    floe_manager.predict_floes()
    floe_manager.associate(curr_raster)
    floe_manager.correct_floes()
