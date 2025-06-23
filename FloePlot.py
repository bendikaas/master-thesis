import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from pyproj import Transformer
from shapely.geometry import box, mapping
from rasterio.mask import mask
import seaborn as sns
from matplotlib import patches
from matplotlib.patches import Ellipse
import matplotlib.lines as mlines
from scipy.stats import chi2
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ------------ REAL DATA ------------

def show_raster(raster):
    plt.figure(figsize=(10, 10))
    plt.imshow(raster.image, cmap='gray')
    plt.show()

def visualize_optical_flow(prev_image, curr_image, avg_du, avg_dv, good_prev_keypoints, good_curr_keypoints):
    # Normalize and convert grayscale images to float [0, 1]
    prev_norm = cv2.normalize(prev_image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    curr_norm = cv2.normalize(curr_image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    prev_mask = prev_norm > 0.1
    curr_mask = curr_norm > 0.1
    overlap_mask = prev_mask & curr_mask

    overlay = np.zeros((*prev_norm.shape, 3), dtype=np.float32)

    palette = sns.color_palette("muted")
    color_prev = np.array(palette[0])  # blue
    color_curr = np.array(palette[1])  # orange
    color_overlap = (color_prev + color_curr) / 2  # perceptual mix

    # prev only
    for c in range(3):
        overlay[..., c][prev_mask & ~curr_mask] = color_prev[c] * prev_norm[prev_mask & ~curr_mask]

    # curr only
    for c in range(3):
        overlay[..., c][curr_mask & ~prev_mask] = color_curr[c] * curr_norm[curr_mask & ~prev_mask]

    # overlap
    overlap_intensity = ((prev_norm + curr_norm) / 2)[overlap_mask]
    for c in range(3):
        overlay[..., c][overlap_mask] = color_overlap[c] * overlap_intensity

    # Display image
    plt.figure(figsize=(8, 6))
    plt.imshow(overlay)

    for i in range(len(good_prev_keypoints)):
        x1, y1 = good_prev_keypoints[i].ravel()
        x2, y2 = good_curr_keypoints[i].ravel()

        plt.scatter(x1, y1, color='white', marker='o', s=10, label="Prev Keypoint" if i == 0 else "")
        plt.arrow(x1, y1, x2 - x1, y2 - y1, color='white', head_width=2, length_includes_head=True)

    center_x, center_y = np.mean(good_prev_keypoints, axis=0).ravel()
    plt.arrow(center_x, center_y, avg_du, avg_dv, color='red', head_width=5, width=1.5, length_includes_head=True, label="Avg Motion")

    # plt.legend()
    plt.axis("off")
    plt.show()

def plot_paths(floe_manager):
    dataset = rasterio.open("satelite_image.tif")  # EPSG:32630

    # Read image and transform
    image = dataset.read()
    transform = dataset.transform

    # Plot the image
    fig, ax = plt.subplots(figsize=(10, 10))
    show(image, transform=transform, ax=ax)

    # Coordinate transformer (if needed)
    transformer = Transformer.from_crs("EPSG:6052", "EPSG:32630", always_xy=True)
    palette = sns.color_palette("colorblind", n_colors=100)

    legends = []

    all_floes = list(floe_manager.active_floes.values()) + list(floe_manager.lost_floes.values())
    for floe_idx, floe in enumerate(all_floes):
        color = palette[floe_idx % len(palette) + 1] # color for that floe
        floe_label = f'Floe {floe_idx}'

        # Label for floe ID
        floe_line = plt.Line2D([], [], color=color, linestyle='-', marker='o', label=floe_label)
        legends.append(floe_line)

        # Plot predicted path
        if len(floe.predicted_states) > 0:
            predicted_path = np.array([(s[0, 0], s[1, 0]) for s in floe.predicted_states]) # EPSG:6052
            predicted_trans = [transformer.transform(x, y) for x, y in predicted_path] # EPSG:32630
            x_pred, y_pred = zip(*predicted_trans)
            ax.plot(x_pred, y_pred, linestyle='--', color=color, marker='s', label=None)

        # plot backtracked path
        if len(floe.backtracked_states) > 0:
            backtracked_path = np.array([(s[0, 0], s[1, 0]) for s in floe.backtracked_states]) # EPSG:6052
            backtraced_trans = [transformer.transform(x, y) for x, y in backtracked_path] # EPSG:32630
            x_back, y_back = zip(*backtraced_trans)
            ax.plot(x_back, y_back, linestyle='--', color=color, marker='P', label=None)

        # Plot corrected path
        estimated_path = np.array([(s[0, 0], s[1, 0]) for s in floe.states])
        if len(floe.states) > 0:
            estimated_path = np.array([(s[0, 0], s[1, 0]) for s in floe.states])
            estimated_trans = [transformer.transform(x, y) for x, y in estimated_path]
            x, y = zip(*estimated_trans)
            ax.plot(x, y, marker='o', color=color, label=None)
            
    corrected_line = plt.Line2D([], [], color=color, linestyle='-', marker='o', label='Estimated Path')
    forward_pred_line = plt.Line2D([], [], color=color, linestyle='--', marker='s', label='Forward Prediction')
    backward_pred_line = plt.Line2D([], [], color=color, linestyle='--', marker='P', label='Backward Prediction')
    style_legend = ax.legend(handles=[corrected_line, forward_pred_line, backward_pred_line], loc='upper left')
    ax.add_artist(style_legend)
    ax.legend(handles=legends, loc='upper right', title="Floe ID")

    ax.set_xlim(floe_manager.grid.east_min, floe_manager.grid.east_max)
    ax.set_ylim(floe_manager.grid.north_min, floe_manager.grid.north_max)

    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

# to plot we need to convert the floe paths in "EPSG:6052" to "EPSG:32630"
def plot_floe_paths_with_uncertainties(floe_manager):
    chi2_val = 5.991
    scale = np.sqrt(chi2_val)

    dataset = rasterio.open("satelite_image.tif")  # EPSG:32630

    # Read image and transform
    image = dataset.read()
    transform = dataset.transform

    # Plot the image
    fig, ax = plt.subplots(figsize=(10, 10))
    show(image, transform=transform, ax=ax)

    # Coordinate transformer (if needed)
    transformer = Transformer.from_crs("EPSG:6052", "EPSG:32630", always_xy=True)
    palette = sns.color_palette("colorblind", n_colors=100)

    legends = []

    all_floes = list(floe_manager.active_floes.values()) + list(floe_manager.lost_floes.values())
    for floe_idx, floe in enumerate(all_floes):
        color = palette[floe_idx % len(palette) + 1]
        floe_label = f'Floe {floe_idx}'

        # Label for floe ID
        floe_line = plt.Line2D([], [], color=color, linestyle='-', marker='o', label=floe_label)
        legends.append(floe_line)

        # Plot predicted path
        if len(floe.predicted_states) > 0:
            predicted_path = np.array([(s[0, 0], s[1, 0]) for s in floe.predicted_states])
            predicted_trans = [transformer.transform(x, y) for x, y in predicted_path]
            x_pred, y_pred = zip(*predicted_trans)
            ax.plot(x_pred, y_pred, linestyle='--', color=color, marker='s', label=None)

            for state, P in zip(floe.predicted_states, floe.predicted_P_matrices):
                x_state, y_state = state[0, 0], state[1, 0]
                x_utm, y_utm = transformer.transform(x_state, y_state)
                std_x = np.sqrt(P[0, 0])
                std_y = np.sqrt(P[1, 1])

                ellipse = Ellipse(
                    (x_utm, y_utm),
                    width=2 * std_x * scale,
                    height=2 * std_y * scale,
                    angle=0,
                    edgecolor=color,
                    facecolor='none',
                    linestyle='--',
                    linewidth=1,
                    alpha=0.3
                )
                ax.add_patch(ellipse)

        # Plot bactracked path
        if len(floe.backtracked_states) > 0:
            backtracked_path = np.array([(s[0, 0], s[1, 0]) for s in floe.backtracked_states]) # EPSG:6052
            backtraced_trans = [transformer.transform(x, y) for x, y in backtracked_path] # EPSG:32630
            x_back, y_back = zip(*backtraced_trans)
            ax.plot(x_back, y_back, linestyle='--', color=color, marker='P', label=None)

            for state, P in zip(floe.backtracked_states, floe.backtracked_P_matrices):
                x_state, y_state = state[0, 0], state[1, 0]
                x_utm, y_utm = transformer.transform(x_state, y_state)
                std_x = np.sqrt(P[0, 0])
                std_y = np.sqrt(P[1, 1])

                ellipse = Ellipse(
                    (x_utm, y_utm),
                    width=2 * std_x * scale,
                    height=2 * std_y * scale,
                    angle=0,
                    edgecolor=color,
                    facecolor='none',
                    linestyle='--',
                    linewidth=1,
                    alpha=0.3
                )
                ax.add_patch(ellipse)

        # Plot corrected path
        if len(floe.states) > 0:
            corrected_path = np.array([(s[0, 0], s[1, 0]) for s in floe.states])
            corrected_trans = [transformer.transform(x, y) for x, y in corrected_path]
            x, y = zip(*corrected_trans)
            ax.plot(x, y, marker='o', color=color, label=None)

            for state, P in zip(floe.states, floe.P_matrices):
                x_state, y_state = state[0, 0], state[1, 0]
                x_utm, y_utm = transformer.transform(x_state, y_state)
                std_x = np.sqrt(P[0, 0])
                std_y = np.sqrt(P[1, 1])

                ellipse = Ellipse(
                    (x_utm, y_utm),
                    width=2 * std_x * scale,
                    height=2 * std_y * scale,
                    angle=0,
                    edgecolor=color,
                    facecolor='none',
                    linestyle=':',
                    linewidth=1,
                    alpha=0.5
                )
                ax.add_patch(ellipse)

    corrected_line = plt.Line2D([], [], color=color, linestyle='-', marker='o', label='Estimated Path')
    predicted_line = plt.Line2D([], [], color=color, linestyle='--', marker='s', label='Forward Prediction')
    backtrack_line = plt.Line2D([], [], color=color, linestyle='--', marker='P', label='Backward Prediction')
    style_legend = ax.legend(handles=[corrected_line, predicted_line, backtrack_line], loc='upper left')
    ax.add_artist(style_legend)
    ax.legend(handles=legends, loc='upper right', title="Floe ID")

    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

def plot_P_evolution(floe_manager, floe_id):
    floe = floe_manager.active_floes.get(floe_id) or floe_manager.lost_floes.get(floe_id)

    if floe is None:
        print(f"Floe ID {floe_id} not found.")
        return

    plt.figure(figsize=(12, 6))
    
    est_color = 'navy'
    pred_color = 'orange'
    back_color = 'green'

    # Estimated
    std_x_estimated = [np.sqrt(P[0, 0]) for P in floe.P_matrices]
    n_estimated = len(std_x_estimated)
    t_estimated = np.arange(0, n_estimated)
    plt.plot(t_estimated, std_x_estimated, color = est_color, label="Estimated")

    # Backward estimated: spans -M to N (same length as backtracked_P_matrices)
    if len(floe.backtracked_P_matrices) > 0:
        std_x_back = [np.sqrt(P[0, 0]) for P in reversed(floe.backtracked_P_matrices)]
        n_back = len(std_x_back)
        t_back = np.arange(-n_back + n_estimated, n_estimated)
        plt.plot(t_back, std_x_back, color = back_color, label="Backward Estimated")

    # Forward predicted (optional): starts after estimated
    if len(floe.predicted_P_matrices) > 0:
        std_x_pred = [np.sqrt(P[0, 0]) for P in floe.predicted_P_matrices]
        n_pred = len(std_x_pred)
        t_pred = np.arange(n_estimated, n_estimated + n_pred)
        plt.plot(t_pred, std_x_pred, color = pred_color, label="Forward Predicted")

    plt.xlabel("Time [s]")
    plt.ylabel("Standard Deviation [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# plots the GT and noisy pos together with KF estimated pos and 95% CI
def plot_stats_of_floe(floe_manager, floe_id):
    # colors
    measurement_color = 'red'
    kalman_color = 'navy'
    estimated_backtracked_color = 'green'
    forward_predicted_color = 'orange'

    floe = floe_manager.active_floes.get(floe_id) or floe_manager.lost_floes.get(floe_id)

    meas_pos =  floe.measurements
    est_pos = [(s[0, 0], s[1, 0]) for s in floe.states]
    est_std = [(np.sqrt(P[0, 0]), np.sqrt(P[1, 1])) for P in floe.P_matrices]
    pred_pos = [(s[0, 0], s[1, 0]) for s in floe.predicted_states]
    pred_std = [(np.sqrt(P[0, 0]), np.sqrt(P[1, 1])) for P in floe.predicted_P_matrices]
    back_pos = [(s[0, 0], s[1, 0]) for s in floe.backtracked_states]
    back_std = [(np.sqrt(P[0, 0]), np.sqrt(P[1, 1])) for P in floe.backtracked_P_matrices]


    est_x = [p[0] for p in est_pos]
    est_y = [p[1] for p in est_pos]
    std_x = [s[0] for s in est_std]
    std_y = [s[1] for s in est_std]
    pred_x = [p[0] for p in pred_pos]
    pred_y = [p[1] for p in pred_pos]
    pred_std_x = [s[0] for s in pred_std]
    pred_std_y = [s[1] for s in pred_std]
    back_x = [p[0] for p in back_pos]
    back_y = [p[1] for p in back_pos]
    back_std_x = [s[0] for s in back_std]
    back_std_y = [s[1] for s in back_std]

    meas_x = [p[0] for p in meas_pos]
    meas_y = [p[1] for p in meas_pos]

    t_est = np.arange(len(est_x))
    t_pred = np.arange(len(est_x), len(est_x) + len(pred_x))
    t_back = np.arange(-len(back_x) + len(est_x), len(est_x))
    t_meas = np.arange(len(meas_x))

    # X direction
    plt.figure(figsize=(10, 5))
    if len(pred_x) > 0:
        plt.plot(t_pred, pred_x, '-o', color=forward_predicted_color)
        plt.fill_between(t_pred, np.array(pred_x) - 1.96 * np.array(pred_std_x),
                        np.array(pred_x) + 1.96 * np.array(pred_std_x),
                        color=forward_predicted_color, alpha=0.2)
    if len(back_x) > 0:
        plt.plot(t_back[::-1], back_x, '-o', color=estimated_backtracked_color, label="Backtracked")
        plt.fill_between(t_back[::-1], np.array(back_x) - 1.96 * np.array(back_std_x),
                        np.array(back_x) + 1.96 * np.array(back_std_x),
                        color=estimated_backtracked_color, alpha=0.2)
        
    plt.plot(t_est, est_x, '-o', color=kalman_color, label="Estimation")
    plt.fill_between(t_est, np.array(est_x) - 1.96 * np.array(std_x), np.array(est_x) + 1.96 * np.array(std_x), color=kalman_color, alpha=0.2)
    plt.plot(t_meas, meas_x, 'x', color=measurement_color, linestyle='dotted', label="Measurements")

    plt.xlabel("Time [s]")
    plt.ylabel("East [m]")
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # Y direction
    plt.figure(figsize=(10, 5))    
    if len(pred_y) > 0:
        plt.plot(t_pred, pred_y, '-o', color=forward_predicted_color)
        plt.fill_between(t_pred, np.array(pred_y) - 1.96 * np.array(pred_std_y),
                        np.array(pred_y) + 1.96 * np.array(pred_std_y),
                        color=forward_predicted_color, alpha=0.2)
    if len(back_y) > 0:
        plt.plot(t_back[::-1], back_y, '-o', color=estimated_backtracked_color, label="Backtracked")
        plt.fill_between(t_back[::-1], np.array(back_y) - 1.96 * np.array(back_std_y),
                        np.array(back_y) + 1.96 * np.array(back_std_y),
                        color=estimated_backtracked_color, alpha=0.2)
    
    plt.plot(t_est, est_y, '-o', color=kalman_color, label="Estimation")
    plt.fill_between(t_est, np.array(est_y) - 1.96 * np.array(std_y), np.array(est_y) + 1.96 * np.array(std_y), color=kalman_color, alpha=0.2)
    plt.plot(t_meas, meas_y, 'x', color=measurement_color, linestyle='dotted', label="Measurements")

    plt.xlabel("Time [s]")
    plt.ylabel("North [m]")
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# ------------ SIMULATOR ------------

def show_GT_map(simulator):
    plt.figure(figsize=(10, 10))
    plt.imshow(simulator.GT_map, cmap='gray')
    plt.title('GT map')
    plt.axis('on')
    plt.show()

# plots GT_paths, noisy measurements and Kalman filter estimate on top of GT_image
def plot_paths_sim(simulator, floe_manager=None):
    plt.figure(figsize=(10, 10))

    if simulator.GT_map is not None:
        h, w = simulator.GT_map.shape

        plt.imshow(simulator.GT_map, cmap='gray', origin='upper')
        plt.xlim(0, w)
        plt.ylim(h, 0)  # top down Y-axis

    # colors
    gt_color = 'limegreen'
    measurement_color = 'red'
    GT_backtracked_color = 'orange'
    kalman_color = 'deepskyblue'
    estimated_backtracked_color = 'violet'
    forward_predicted_color = 'yellow'

    legends = []

    for floe in list(simulator.active_floes.values()) + list(simulator.lost_floes.values()):
        gt_path = np.array([(state[0, 0], state[1, 0]) for state in floe.GT_states])
        noisy_path = np.array([(state[0, 0], state[1, 0]) for state in floe.noisy_states])

        plt.plot(gt_path[:, 0], gt_path[:, 1], '-o', color=gt_color, linewidth=1.5)
        plt.plot(noisy_path[:, 0], noisy_path[:, 1], linestyle='dotted', marker='x',
                color=measurement_color, markersize=10, markeredgewidth=2)

        legends.append(mlines.Line2D([], [], color=gt_color, linestyle='-', marker='o', label='GT Path'))
        legends.append(mlines.Line2D([], [], color=measurement_color, linestyle='dotted', marker='x', label='Measurements'))

        if len(floe.GT_backtracked_states) > 0:
            GT_backtrack = np.array([(state[0, 0], state[1, 0]) for state in floe.GT_backtracked_states])
            plt.plot(GT_backtrack[:, 0], GT_backtrack[:, 1], '-o', color=GT_backtracked_color, linewidth=1.5)
            legends.append(mlines.Line2D([], [], color=GT_backtracked_color, linestyle='-', marker='o', label='GT Backtrack'))

    if floe_manager is not None:
        for floe in list(floe_manager.active_floes.values()) + list(floe_manager.lost_floes.values()):
            path = np.array([(state[0, 0], state[1, 0]) for state in floe.states])
            path_in_pixels = [simulator.meters_to_pixels(x, y) for (x, y) in path]
            path_in_pixels = np.array(path_in_pixels)

            plt.plot(path_in_pixels[:, 0], path_in_pixels[:, 1], '-o', color=kalman_color, linewidth=1.5)
            legends.append(mlines.Line2D([], [], color=kalman_color, linestyle='-', marker='o', label='Estimated Path'))

            if len(floe.backtracked_states) > 0:
                back_path = np.array([(state[0, 0], state[1, 0]) for state in floe.backtracked_states])
                back_path_in_pixels = [simulator.meters_to_pixels(x, y) for (x, y) in back_path]
                back_path_in_pixels = np.array(back_path_in_pixels)

                plt.plot(back_path_in_pixels[:, 0], back_path_in_pixels[:, 1], '-o', color=estimated_backtracked_color, linewidth=1.5)
                legends.append(mlines.Line2D([], [], color=estimated_backtracked_color, linestyle='-', marker='o', label='Estimated Backtrack'))

            if len(floe.predicted_states) > 0:
                predicted_path = np.array([(state[0, 0], state[1, 0]) for state in floe.predicted_states])
                predicted_path_in_pixels = [simulator.meters_to_pixels(x, y) for (x, y) in predicted_path]
                predicted_path_in_pixels = np.array(predicted_path_in_pixels)

                plt.plot(predicted_path_in_pixels[:, 0], predicted_path_in_pixels[:, 1], '-o', color=forward_predicted_color, linewidth=1.5)
                legends.append(mlines.Line2D([], [], color=forward_predicted_color, linestyle='-', marker='o', label='Forward Prediction'))

    plt.legend(handles=legends, loc='upper right')
    plt.xlabel("East [pixels]")
    plt.ylabel("North [pixels]")
    plt.tight_layout(pad=0)
    plt.show()

# plots 2D path with 95% CI
def plot_map_with_uncertainties_sim(simulator, floe_manager):
    chi2_val = 5.991
    scale = np.sqrt(chi2_val)

    plt.figure(figsize=(10, 10))

    if simulator.GT_map is not None:
        h, w = simulator.GT_map.shape
        plt.imshow(simulator.GT_map, cmap='gray', origin='upper')
        plt.xlim(0, w)
        plt.ylim(h, 0)

    # Colors
    gt_color = 'limegreen'
    measurement_color = 'red'
    GT_backtracked_color = 'orange'
    kalman_color = 'deepskyblue'
    estimated_backtracked_color = 'violet'
    forward_predicted_color = 'yellow'

    legends = []

    for floe in list(simulator.active_floes.values()) + list(simulator.lost_floes.values()):
        gt_path = np.array([(s[0, 0], s[1, 0]) for s in floe.GT_states])
        noisy_path = np.array([(s[0, 0], s[1, 0]) for s in floe.noisy_states])

        plt.plot(gt_path[:, 0], gt_path[:, 1], linestyle='-', marker='o', color=gt_color, linewidth=1.5)
        legends.append(mlines.Line2D([], [], color=gt_color, linestyle='-', marker='o', label='GT Path'))
        
        plt.plot(noisy_path[:, 0], noisy_path[:, 1], linestyle='dotted', marker='x', color=measurement_color, markersize=10, markeredgewidth=2)        
        legends.append(mlines.Line2D([], [], color=measurement_color, linestyle='dotted', marker='x', label='Measurements'))

        if len(floe.GT_backtracked_states) > 0:
            GT_backtrack = np.array([(s[0, 0], s[1, 0]) for s in floe.GT_backtracked_states])
            plt.plot(GT_backtrack[:, 0], GT_backtrack[:, 1], '-o', color=GT_backtracked_color, linewidth=1.5)
            legends.append(mlines.Line2D([], [], color=GT_backtracked_color, linestyle='dotted', marker='o', label='GT Backtrack'))

    if floe_manager is not None:
        for floe in list(floe_manager.active_floes.values()) + list(floe_manager.lost_floes.values()):
            path = np.array([(s[0, 0], s[1, 0]) for s in floe.states])
            path_pixels = [simulator.meters_to_pixels(x, y) for (x, y) in path]
            path_pixels = np.array(path_pixels)

            plt.plot(path_pixels[:, 0], path_pixels[:, 1], linestyle='-', marker='o', color=kalman_color, linewidth=1.5)
            legends.append(mlines.Line2D([], [], color=kalman_color, linestyle='-', marker='o', label='Estimated Path'))

            for state, P in zip(floe.states, floe.P_matrices):
                x, y = state[0, 0], state[1, 0]
                u, v = simulator.meters_to_pixels(x, y)

                std_x = np.sqrt(P[0, 0])
                std_y = np.sqrt(P[1, 1])

                ellipse = patches.Ellipse(
                    (u, v),
                    width=2 * std_x * scale / simulator.resolution,
                    height=2 * std_y * scale / simulator.resolution,
                    angle=0,
                    edgecolor=kalman_color,
                    facecolor='none',
                    linewidth=1,
                    alpha=0.5
                )
                
                plt.gca().add_patch(ellipse)

            if len(floe.backtracked_states) > 0:
                back_path = np.array([(s[0, 0], s[1, 0]) for s in floe.backtracked_states])
                back_path_pixels = [simulator.meters_to_pixels(x, y) for (x, y) in back_path]
                back_path_pixels = np.array(back_path_pixels)

                plt.plot(back_path_pixels[:, 0], back_path_pixels[:, 1], linestyle='-', marker='o', color=estimated_backtracked_color, linewidth=1.5)
                legends.append(mlines.Line2D([], [], color=estimated_backtracked_color, linestyle='-', marker='o', label='Estimated backtrack'))

                for state, P in zip(floe.backtracked_states, floe.backtracked_P_matrices):
                    x, y = state[0, 0], state[1, 0]
                    u, v = simulator.meters_to_pixels(x, y)

                    std_x = np.sqrt(P[0, 0])
                    std_y = np.sqrt(P[1, 1])

                    ellipse = patches.Ellipse(
                        (u, v),
                        width=2 * std_x * scale / simulator.resolution,
                        height=2 * std_y * scale / simulator.resolution,
                        angle=0,
                        edgecolor=estimated_backtracked_color,
                        facecolor='none',
                        linewidth=1,
                        alpha=0.5
                    )
                    plt.gca().add_patch(ellipse)

            if len(floe.predicted_states) > 0:
                pred_path = np.array([(s[0, 0], s[1, 0]) for s in floe.predicted_states])
                pred_path_pixels = [simulator.meters_to_pixels(x, y) for (x, y) in pred_path]
                pred_path_pixels = np.array(pred_path_pixels)

                plt.plot(pred_path_pixels[:, 0], pred_path_pixels[:, 1], linestyle='-', marker='o', color=forward_predicted_color, linewidth=1.5)
                legends.append(mlines.Line2D([], [], color=forward_predicted_color, linestyle='-', marker='o', label='Forward Prediction'))

                for state, P in zip(floe.predicted_states, floe.predicted_P_matrices):
                    x, y = state[0, 0], state[1, 0]
                    u, v = simulator.meters_to_pixels(x, y)

                    std_x = np.sqrt(P[0, 0])
                    std_y = np.sqrt(P[1, 1])

                    ellipse = patches.Ellipse(
                        (u, v),
                        width=2 * std_x * scale / simulator.resolution,
                        height=2 * std_y * scale / simulator.resolution,
                        angle=0,
                        edgecolor=forward_predicted_color,
                        facecolor='none',
                        linewidth=1,
                        alpha=0.5
                    )
                    plt.gca().add_patch(ellipse)

    plt.legend(handles=legends, loc='upper right')
    plt.xlabel("East [pixels]")
    plt.ylabel("North [pixels]")

    plt.tight_layout(pad=0)
    plt.show()

# plots the GT and noisy pos together with KF estimated pos and 95% CI
def plot_stats_of_floe_sim(simulator, floe_manager, floe_id):
    floe = floe_manager.active_floes.get(floe_id) or floe_manager.lost_floes.get(floe_id)
    sim_floe = simulator.active_floes.get(floe_id) or simulator.lost_floes.get(floe_id)

    if floe is None or sim_floe is None:
        print(f"Floe ID {floe_id} not found.")
        return

    est_pos = [(s[0, 0], s[1, 0]) for s in floe.states]
    est_std = [(np.sqrt(P[0, 0]), np.sqrt(P[1, 1])) for P in floe.P_matrices]
    pred_pos = [(s[0, 0], s[1, 0]) for s in floe.predicted_states]
    pred_std = [(np.sqrt(P[0, 0]), np.sqrt(P[1, 1])) for P in floe.predicted_P_matrices]
    back_pos = [(s[0, 0], s[1, 0]) for s in floe.backtracked_states]
    back_std = [(np.sqrt(P[0, 0]), np.sqrt(P[1, 1])) for P in floe.backtracked_P_matrices]

    gt_pos = [simulator.pixels_to_meters(s[0, 0], s[1, 0]) for s in sim_floe.GT_states]
    meas_pos = [simulator.pixels_to_meters(s[0, 0], s[1, 0]) for s in sim_floe.noisy_states]

    est_x = [p[0] for p in est_pos]
    est_y = [p[1] for p in est_pos]
    std_x = [s[0] for s in est_std]
    std_y = [s[1] for s in est_std]
    pred_x = [p[0] for p in pred_pos]
    pred_y = [p[1] for p in pred_pos]
    pred_std_x = [s[0] for s in pred_std]
    pred_std_y = [s[1] for s in pred_std]
    back_x = [p[0] for p in back_pos]
    back_y = [p[1] for p in back_pos]
    back_std_x = [s[0] for s in back_std]
    back_std_y = [s[1] for s in back_std]
    gt_x = [p[0] for p in gt_pos]
    gt_y = [p[1] for p in gt_pos]
    meas_x = [p[0] for p in meas_pos]
    meas_y = [p[1] for p in meas_pos]

    t_est = np.arange(len(est_x))
    t_pred = np.arange(len(est_x), len(est_x) + len(pred_x))
    t_back = np.arange(-len(back_x) + len(est_x), len(est_x))
    t_gt = np.arange(len(gt_x))
    t_meas = np.arange(len(meas_x))

    # X direction
    plt.figure(figsize=(10, 5))
    plt.plot(t_gt, gt_x, '-o', color="limegreen", label="Ground Truth")
    plt.plot(t_meas, meas_x, 'x', color="red", linestyle='dotted', label="Measurements")
    plt.plot(t_est, est_x, '-o', color="navy", label="Estimation")
    plt.fill_between(t_est, np.array(est_x) - 1.96 * np.array(std_x), np.array(est_x) + 1.96 * np.array(std_x), color="navy", alpha=0.2)
    
    if len(pred_x) > 0:
        plt.plot(t_pred, pred_x, '-o', color="navy")
        plt.fill_between(t_pred, np.array(pred_x) - 1.96 * np.array(pred_std_x),
                        np.array(pred_x) + 1.96 * np.array(pred_std_x),
                        color="navy", alpha=0.2)
    if len(back_x) > 0:
        plt.plot(t_back[::-1], back_x, '-o', color="violet", label="Backtracked")
        plt.fill_between(t_back[::-1], np.array(back_x) - 1.96 * np.array(back_std_x),
                        np.array(back_x) + 1.96 * np.array(back_std_x),
                        color="violet", alpha=0.2)

    plt.xlabel("Time [s]")
    plt.ylabel("East [m]")
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # Y direction
    plt.figure(figsize=(10, 5))
    plt.plot(t_gt, gt_y, '-o', color="limegreen", label="Ground Truth")
    plt.plot(t_meas, meas_y, 'x', color="red", linestyle='dotted', label="Measurements")
    plt.plot(t_est, est_y, '-o', color="navy", label="Estimation")
    plt.fill_between(t_est, np.array(est_y) - 1.96 * np.array(std_y), np.array(est_y) + 1.96 * np.array(std_y), color="navy", alpha=0.2)
    
    if len(pred_y) > 0:
        plt.plot(t_pred, pred_y, '-o', color="navy")
        plt.fill_between(t_pred, np.array(pred_y) - 1.96 * np.array(pred_std_y),
                        np.array(pred_y) + 1.96 * np.array(pred_std_y),
                        color="navy", alpha=0.2)
    if len(back_y) > 0:
        plt.plot(t_back[::-1], back_y, '-o', color="violet", label="Backtracked")
        plt.fill_between(t_back[::-1], np.array(back_y) - 1.96 * np.array(back_std_y),
                        np.array(back_y) + 1.96 * np.array(back_std_y),
                        color="violet", alpha=0.2)

    plt.xlabel("Time [s]")
    plt.ylabel("North [m]")
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# Compute and plot the NEES (Normalized Estimation Error Squared) for a floe with id floe_id
def plot_NEES(simulator, floe_manager, floe_id = 0):
    nees_values = []
    state_dim = 4

    floe = floe_manager.active_floes.get(floe_id) or floe_manager.lost_floes.get(floe_id)
    sim_floe = simulator.active_floes.get(floe_id) or simulator.lost_floes.get(floe_id)

    est_states = floe.states
    P_matrices = floe.P_matrices
    gt_states = [simulator.gt_state_pixels_to_meters(s) for s in sim_floe.GT_states]

    for k in range(len(est_states)):
        x_est = est_states[k]  # shape (4, 1)
        x_gt = gt_states[k]    # shape (4, 1)
        P = P_matrices[k]      # shape (4, 4)

        err = x_est - x_gt

        try:
            nees = err.T @ np.linalg.inv(P) @ err
            nees_values.append(nees.item())
        except np.linalg.LinAlgError:
            nees_values.append(np.nan)

    lower = chi2.ppf(0.025, df=state_dim)
    upper = chi2.ppf(0.975, df=state_dim)

    plt.figure(figsize=(10, 4))
    plt.plot(nees_values, label='NEES')
    plt.hlines([lower, upper], 0, len(nees_values), colors='red', linestyles='dashed', label='95\% CI')
    plt.xlabel('Time [s]')
    plt.ylabel('NEES')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Compute and plot the NIS (Normalized Innovation Squared) for a floe with id floe_id
def plot_NIS(floe_manager, floe_id = 0):
    nis_values = []
    meas_dim = 2  # only measuring position (x, y)

    floe = floe_manager.active_floes.get(floe_id) or floe_manager.lost_floes.get(floe_id)

    est_states = floe.states
    P_matrices = floe.P_matrices
    measurements = floe.measurements
    C = floe.kalmanfilter.C
    R = floe.kalmanfilter.R

    for k in range(len(est_states)):
        if k >= len(measurements):
            nis_values.append(np.nan)
            continue

        meas = measurements[k]
        if meas is None:
            nis_values.append(np.nan)
            continue

        x_est = est_states[k]
        P = P_matrices[k]
        z = np.array([[meas[0]], [meas[1]]])

        # Innovation
        z_pred = C @ x_est
        innov = z - z_pred
        S = C @ P @ C.T + R

        try:
            nis = innov.T @ np.linalg.inv(S) @ innov
            nis_values.append(nis.item())
        except np.linalg.LinAlgError:
            nis_values.append(np.nan)

    # Confidence interval for Chi squared, with 2 DOF
    lower = chi2.ppf(0.025, df=meas_dim)
    upper = chi2.ppf(0.975, df=meas_dim)

    plt.figure(figsize=(10, 4))
    plt.plot(nis_values, label='NIS')
    plt.hlines([lower, upper], 0, len(nis_values), colors='red', linestyles='dashed', label='95\% CI')
    plt.xlabel('Time [s]')
    plt.ylabel('NIS')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

 # ------------ GRID ------------

def visualize(grid):
    arrow_scale = 25
    fig, ax = plt.subplots(figsize=(6, 6))

    distances_np = np.array(grid.distances)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=np.nanmin(distances_np), vmax=np.nanmax(distances_np))

    # Draw grid lines
    for i in range(grid.ni + 1):
        x = i * grid.cellsize
        ax.plot([x, x], [0, grid.nj * grid.cellsize], color='lightgray')

    for j in range(grid.nj + 1):
        y = j * grid.cellsize
        ax.plot([0, grid.ni * grid.cellsize], [y, y], color='lightgray')

    # Fill known cells with pink
    for i, j in grid.known_cells:
        x = i * grid.cellsize
        y = j * grid.cellsize
        ax.add_patch(plt.Rectangle((x, y), grid.cellsize, grid.cellsize,
                                color='#f8c8dc', zorder=0))

    # Draw motion vectors
    for j in range(grid.nj):
        for i in range(grid.ni):
            if grid.cells[j][i] is not None:
                dx, dy = grid.cells[j][i]
                dist = grid.distances[j][i]

                x = i * grid.cellsize + grid.cellsize / 2
                y = j * grid.cellsize + grid.cellsize / 2

                color = cmap(norm(dist)) if dist is not None else (0.7, 0.7, 0.7, 1)
                ax.arrow(x, y, dx * arrow_scale, dy * arrow_scale,
                        fc=color, ec=color,
                        head_width=2,
                        head_length=2,
                        length_includes_head=True)

    # Colorbar with spacing and smaller size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.5 )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label("Distance to nearest known cell [m]")

    ax.set_xlim(0, grid.ni * grid.cellsize)
    ax.set_ylim(0, grid.nj * grid.cellsize)

    # Set tick positions and labels to match real-world coordinates
    xticks = np.arange(0, grid.ni + 1, grid.ni)
    yticks = np.arange(0, grid.nj + 1, grid.nj)
    ax.set_xticks(xticks * grid.cellsize)
    ax.set_xticklabels((xticks * grid.cellsize + grid.east_min).astype(int))

    ax.set_yticks(yticks * grid.cellsize)
    ax.set_yticklabels((yticks * grid.cellsize + grid.north_min).astype(int))

    ax.set_aspect('equal')
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    plt.grid(False)
    plt.show()