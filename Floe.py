import Grid as gr
import FloePlot as fplt
import os, cv2, numpy as np, matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment # hungarian algorithm
import yaml # to load yaml files for intrinsics
from pyproj import Transformer # to convert between coord. reference systems

MAX_DIST = 200
MAX_LOST_FRAMES = 100 # number of frames I want to predict a lost Floe's path
MAX_COST = 100 # max cost a match can have in the cost matrix

class Floe_Manager:
    def __init__(self):
        self.active_floes = {} # Dictionary for Floes, with ID -> Floe
        self.lost_floes = {} # array for lost floes
        self.next_id = 0
        self.grid = gr.Grid(100) # grid for real world
        # self.grid = gr.Grid(10, 0, 100, 0, 100) # Simulator grid

    def generate_id(self):
        current_id = self.next_id
        self.next_id += 1
        return current_id

    def predict_floes(self):
        for floe in list(self.active_floes.values()) + list(self.lost_floes.values()):
            floe.predicted_state = floe.kalmanfilter.predict()

    def correct_floes(self):
        # lost floes
        for floe in self.lost_floes.values():
            if floe.lost_frames < MAX_LOST_FRAMES:
                floe.state = floe.predicted_state # set the state to the predicted state
                floe.predicted_states.append(floe.state.copy()) # store the predicted state
                floe.predicted_P_matrices.append(floe.kalmanfilter.P.copy()) # store the uncertainty matrix P

                floe.lost_frames += 1
                print("Using predicted state")

        # active floes
        for floe in self.active_floes.values():
            if floe.predicted_state is not None:
                # if Optical Flow has returned None
                if floe.measurement is None:
                    print("Using predicted state")
                    floe.state = floe.predicted_state
                    floe.predicted_states.append(floe.state.copy())
                    floe.P_matrices.append(floe.kalmanfilter.P.copy())
                    
                else:
                    x, y = floe.measurement
                    measurement_array = np.array([[x], [y]])
                    floe.state = floe.kalmanfilter.correct(measurement_array)

                    # adds the motion vector in floe_manager.grid
                    transformer = Transformer.from_crs("EPSG:6052", "EPSG:32630", always_xy=True)
                    px_6052, py_6052 = floe.state[0, 0], floe.state[1, 0] # in EPSG:6052
                    px_32630, py_32630 = transformer.transform(px_6052, py_6052) # in EPSG:32630
                    vx, vy = floe.state[2, 0], floe.state[3, 0] # vel is the same in EPSG:6052 and EPSG:32630?? TODO: sjekk dette
                    self.grid.add_velocity_vector((px_32630, py_32630), (vx, vy))

                    # Store the corrected state and P matrix
                    floe.states.append(floe.state.copy())
                    floe.P_matrices.append(floe.kalmanfilter.P.copy())

    def print_active_floes(self):
        for floe in self.active_floes.values():
            floe.print()
    
    def print_lost_floes(self):
        for floe in self.lost_floes.values():
            floe.print()

    # path: r"C:\Users\bendi\Documents\Master\Bendik dataset\IceDrift\calib\int.yaml"
    def get_intrinsics(self, path):
        # Camera calibration parameters
        with open(path, 'r') as int_yaml:
            cam_params = yaml.safe_load(int_yaml)
            
            #dist_coeffs = np.array(cam_params["distortion_coeffs"])
            fx, fy, cx, cy = cam_params["intrinsics"]
            intrinsic_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            w, h = tuple(cam_params["resolution"])

        # Some nice preprocessing    
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        uv1 = np.vstack((u.flatten(), v.flatten(), np.ones_like(u.flatten()))).T  # Pixel coordinates
        r_cam = (np.linalg.inv(intrinsic_matrix) @ uv1.T)  # Pixel vector in cam frame

        return w, h, intrinsic_matrix, r_cam

    # data path: r"C:\Users\bendi\Documents\mine python prosjekter\test\12345678"
    # intrinsics path: r"C:\Users\bendi\Documents\Master\Bendik dataset\IceDrift\calib\int.yaml"
    def create_main_raster(self, data_path, w, h, intrinsic_matrix, r_cam):
        # Load image and pose
        image_path = os.path.join(data_path, "binary.png")
        pose_path = os.path.join(data_path, "pose.npy")
        distance_path = os.path.join(data_path, "distance_map.npy")

        image = cv2.imread(image_path)
        pose = np.load(pose_path)  # 4x4 pose matrix
        pose_inv = np.linalg.inv(pose)

        # === Create distance_map ===
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        pixels = np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))  # shape (3, N)
        r_cam_full = np.linalg.inv(intrinsic_matrix) @ pixels                      # shape (3, N)

        R = pose[:3, :3]
        t = pose[:3, 3]

        r_world = R @ r_cam_full                                                   # shape (3, N)
        scale = -t[2] / r_world[2]
        r_world *= scale

        dist = np.where(r_world[2] <= 1e-9, np.inf, np.linalg.norm(r_world[:2], axis=0))
        distance_map = dist.reshape((h, w))
        np.save(distance_path, distance_map)

        # === Continue with raster projection ===
        valid_distance = distance_map < MAX_DIST
        r_world = R @ r_cam[:, valid_distance.flatten()]
        scale = -t[2] / r_world[2]
        r_world *= scale

        p_world = np.expand_dims(t, 1) + r_world

        north_min, north_max = p_world[0].min(), p_world[0].max()
        east_min, east_max = p_world[1].min(), p_world[1].max()

        resolution = 0.1
        decimation = int(1 / resolution)
        raster_dim = (int((north_max - north_min) * decimation),
                    int((east_max - east_min) * decimation), 3)

        x, y = np.meshgrid(
            np.linspace(north_max, north_min, raster_dim[0]),
            np.linspace(east_min, east_max, raster_dim[1])
        )

        p_raster_world = np.vstack((x.flatten(), y.flatten(), np.zeros(x.size), np.ones(x.size)))
        d_cam = np.linalg.norm(p_raster_world[:2] - np.expand_dims(t[:2], 1), axis=0)

        p_raster_cam = pose_inv @ p_raster_world

        uv_raster = intrinsic_matrix @ p_raster_cam[:3]
        uv_raster = np.rint(uv_raster[:2] / uv_raster[2]).astype(int)

        valid_uv = (uv_raster[0] >= 0) & (uv_raster[1] >= 0) & (uv_raster[1] < h) & (uv_raster[0] < w)
        valid_uv &= (d_cam < MAX_DIST)
        uv_raster_valid = uv_raster[:, valid_uv]

        raster_flat = np.zeros((valid_uv.size, raster_dim[2]))
        raster_flat[valid_uv] = image[uv_raster_valid[1], uv_raster_valid[0]]

        raster = raster_flat.reshape(raster_dim, order='F').astype(np.uint8)
        image = cv2.cvtColor(raster, cv2.COLOR_BGR2GRAY)

        return Raster(image, resolution, north_max, east_min)
    
    # gets the params to create so it only contains white pixels. padding in pixels
    # returnerer det som skal bli Floe's variabel. Lager nye Rasters fra dette i get_detections.
    def get_cropped_raster_params(self, isolated_image, main_raster, padding = 20):
        coords = np.column_stack(np.where(isolated_image > 0))  # (row, col) format

        # Get bounding box around the floe
        y_min, x_min = coords.min(axis=0)  # Top-left corner (in image space)
        y_max, x_max = coords.max(axis=0)  # Bottom-right corner

        # Apply padding (ensure within image bounds)
        y_min = max(y_min - padding, 0)
        x_min = max(x_min - padding, 0)
        y_max = min(y_max + padding, isolated_image.shape[0])
        x_max = min(x_max + padding, isolated_image.shape[1])

        # Crop the image
        cropped_image = isolated_image[y_min:y_max, x_min:x_max].astype(np.uint8)

        # Update real-world coordinates for top-left of the crop
        new_north_max = main_raster.north_max - y_min * main_raster.resolution
        new_east_min = main_raster.east_min + x_min * main_raster.resolution

        return cropped_image, new_north_max, new_east_min

    def get_common_raster_window_params(self, raster1, raster2):
        # Extract bounds from raster1
        north_min1 = raster1.north_max - raster1.image.shape[0] * raster1.resolution
        north_max1 = raster1.north_max
        east_min1 = raster1.east_min
        east_max1 = raster1.east_min + raster1.image.shape[1] * raster1.resolution

        # Extract bounds from raster2
        north_min2 = raster2.north_max - raster2.image.shape[0] * raster2.resolution
        north_max2 = raster2.north_max
        east_min2 = raster2.east_min
        east_max2 = raster2.east_min + raster2.image.shape[1] * raster2.resolution

        # Combine bounds
        new_north_max = max(north_max1, north_max2)  # Top edge
        new_north_min = min(north_min1, north_min2)  # Bottom edge
        new_east_min = min(east_min1, east_min2)     # Left
        new_east_max = max(east_max1, east_max2)     # Right

        # Compute new dimensions
        new_height = np.ceil((new_north_max - new_north_min) / raster1.resolution)
        new_width = np.ceil((new_east_max - new_east_min) / raster1.resolution)

        return new_north_max, new_east_min, new_height, new_width
    
    # pads image to the given common window. Doesnt change the input raster 
    def create_comparable_image(self, raster, new_north_max, new_east_min, new_height, new_width):
        row_offset = int((new_north_max - raster.north_max) / raster.resolution)
        col_offset = int((raster.east_min - new_east_min) / raster.resolution)

        if row_offset < 0 or col_offset < 0:
            raise ValueError("Raster does not fit within the new padded window.")
        
        new_height = int(new_height)
        new_width = int(new_width)

        # Create empty arrays to paste in the values
        image_in_common_frame = np.zeros((new_height, new_width), dtype=raster.image.dtype)

        h, w = raster.image.shape[:2]

        image_in_common_frame[row_offset:row_offset + h, col_offset:col_offset + w] = raster.image # inserting raster.image into the larger canvas
        
        return image_in_common_frame
    
    def get_kernel(self, image):
        # create a kernel for morphological operations based on the image size
        floe_area = cv2.countNonZero(image)

        if floe_area < 1000:
            size = 3
        elif floe_area < 10000:
            size = 7
        else:
            size = 11

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) # create the kernel

        return kernel, size

    def close_and_open(self, image, kernel):
        # given a krernel, perform closing and opening of an image
        closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, kernel)
        return opened_image

    def apply_blur(self, image, kernel):
        # apply Gaussian blur to the edges of an image
        edges = cv2.Canny(image, 50, 150)
        dilated_edges = cv2.dilate(edges, kernel) # making the border larger
        blur_mask = dilated_edges > 0 # Boolean mask for where we want the blurring

        img = image.copy()
        blurred_image = cv2.GaussianBlur(img, (11, 11), 0) # blur the whole image
        img[blur_mask] = blurred_image[blur_mask]

        return img

    # delta is set to 3 by default
    def huber_loss(self, residuals, delta):
        abs_r = np.abs(residuals)
        return np.where(
            abs_r <= delta,
            0.5 * abs_r**2,
            delta * (abs_r - 0.5 * delta)
        )

    def RANSAC(self, good_prev, good_curr, delta=3.0, loss_threshold=20.0):
        # RANSAC to filter outliers. Returns the inlier keypoints
        prev_pts = good_prev.reshape(-1, 2)
        curr_pts = good_curr.reshape(-1, 2)
        motions = curr_pts - prev_pts # motions are of a minimum length of 3

        N = len(motions) // 2 + 1 # number of iterations

        best_inliers = []
        best_translation = None

        for i in range(N):
            # Pick one random motion vector as candidate
            idx = np.random.randint(len(motions))
            candidate_translation = motions[idx]

            # Compare all motion vectors to this candidate
            diffs = motions - candidate_translation
            distances = np.linalg.norm(diffs, axis=1)
            losses = self.huber_loss(distances, delta)

            inlier_indices = np.where(losses < loss_threshold)[0]

            if len(inlier_indices) > len(best_inliers):
                best_inliers = inlier_indices
                best_translation = candidate_translation

        if best_translation is None or len(best_inliers) == 0:
            raise ValueError("RANSAC failed to find a valid translation.")
        
        # Average motion of all inliers
        prev_inliers = prev_pts[best_inliers]
        curr_inliers = curr_pts[best_inliers]

        return prev_inliers, curr_inliers

    def optical_flow(self, prev_image, curr_image):
        prev_keypoints = cv2.goodFeaturesToTrack(
            prev_image,
            maxCorners = 100,
            qualityLevel = 0.1,
            minDistance = 10,
            blockSize = 11,
            useHarrisDetector = False
        )

        if prev_keypoints is None:
            raise ValueError("No keypoints found.")

        # Lucas-Kanade Optical Flow parameters
        lk_params = dict(
            winSize=(21, 21),
            maxLevel = 3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        curr_keypoints, status, error = cv2.calcOpticalFlowPyrLK(
            prev_image, curr_image, prev_keypoints, None, **lk_params
        )

        if curr_keypoints is None or status is None:
            raise ValueError("No keypoints were tracked successfully.")
        
        # Keep only successfully tracked keypoints
        good_prev = prev_keypoints[status.flatten() == 1].reshape(-1, 2)
        good_curr = curr_keypoints[status.flatten() == 1].reshape(-1, 2)

        if len(good_prev) == 0:
            raise ValueError("No keypoints were successfully tracked.")
        
        # just take the avg if there are not enough points
        elif len(good_prev) == 1:
            motion = good_curr - good_prev
            avg_du, avg_dv = motion.ravel()

        elif len(good_prev) == 2:
            motion_vectors = good_curr - good_prev
            avg_du, avg_dv = np.mean(motion_vectors, axis=0)

        else:
            try:
                good_prev, good_curr = self.RANSAC(good_prev, good_curr, 3, 20)
                motion_vectors = good_curr - good_prev
            except ValueError as e:
                print(f"RANSAC failed: {e}")
                motion_vectors = good_curr - good_prev # Fallback to using the average of all points

            motion_vectors = good_curr - good_prev
            avg_du, avg_dv = np.mean(motion_vectors, axis=0)

        return avg_du, avg_dv, good_prev, good_curr

    def get_detections(self, main_raster, noise_size = 100):
        detections = [] # storage for Floe objects
        num_labels, connected_image = cv2.connectedComponents(main_raster.image)

        for label in range(1, num_labels):
            if label == 0: # the background
                continue

            if np.sum(connected_image == label) < noise_size: # too small for a Floe, ignore it
                continue

            else:
                isolated_image = (connected_image == label).astype(np.uint8) * 255 # contains only one floe
                cropped_image, new_north_max, new_east_min = self.get_cropped_raster_params(isolated_image, main_raster)
                sub_raster = Raster(cropped_image, main_raster.resolution, new_north_max, new_east_min) # new Raster object

                M = cv2.moments(cropped_image)
                if M["m00"] != 0:
                    u, v = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                else:
                    print("Warning: No mass center found. Using bounding rect as pos instead.")
                    u, v, _, _ = cv2.boundingRect(cropped_image)  # fallback

                measured_pos = (sub_raster.east_min + u * sub_raster.resolution, sub_raster.north_max - v * sub_raster.resolution)
                detection = Floe(measured_pos, sub_raster) # new Floe object
                detections.append(detection)

        return detections # list of newly created Floe objects that are detected in main_raster
    
    def get_iou(self, image1, image2):
        intersection = np.logical_and(image1, image2).sum()
        union = np.logical_or(image1, image2).sum()

        return intersection / union if union > 0 else 0
    
    def get_cost_matrix(self, prev_floes, curr_floes):
        # prev_floes: list of known floes in floe_manager.active_floes
        # curr_floes: list of newly detected floes, retruned from get_detections

        cost_matrix = np.full((len(prev_floes), len(curr_floes)), np.inf)  # Initialize with large values

        DIST = 20 # distance in meters
        MAX_IOU = 0.9  # the lower the better. from 0 to 1

        for i, prev_floe in enumerate(prev_floes):
            y_pred = prev_floe.predicted_state[:2].flatten()  # Predicted position [x y]

            for j, curr_floe in enumerate(curr_floes):
                y = np.array(curr_floe.measurement)  # measured position of new detection [x y]
                diff = y - y_pred
                dist_E = np.linalg.norm(diff) # Euclidean distance
                
                if dist_E > DIST: # if the distance is too far away, dont match
                    continue
                
                # need to make comparable images of prev and curr floe
                new_north_max, new_east_min, new_height, new_width = self.get_common_raster_window_params(prev_floe.raster, curr_floe.raster)
                prev_floe_image = self.create_comparable_image(prev_floe.raster, new_north_max, new_east_min, new_height, new_width)
                curr_floe_image = self.create_comparable_image(curr_floe.raster, new_north_max, new_east_min, new_height, new_width)

                iou_cost = 1 - self.get_iou(prev_floe_image, curr_floe_image)

                if iou_cost > MAX_IOU:
                    continue  # Leave cost as np.inf (won't be matched)
                
                # Combine costs (you can weight IoU if necessary)
                cost_matrix[i, j] = dist_E + 10 * iou_cost

        return cost_matrix
    
    # handle the matched floes
    def update_matched_floes(self, prev_match, curr_match, detections, floe_ids, cost_matrix):
        assigned_prev_floe_IDs = set()
        assigned_curr_floe_idxs = set()

        for prev_idx, curr_idx in zip(prev_match, curr_match):
            cost = cost_matrix[prev_idx, curr_idx]

            # skip the match if its cost is beyond max allowed cost
            if not np.isfinite(cost) or cost > MAX_COST:
                continue

            floe_id = floe_ids[prev_idx]
            # print(f"Matched Floe ID: {floe_id} with Detection Index: {curr_idx}")

            prev_floe = self.active_floes[floe_id] # Floe in prev image
            curr_floe = detections[curr_idx] # The same floe in the curr image

            assigned_prev_floe_IDs.add(floe_id) # set of Floe ID's that are assigned
            assigned_curr_floe_idxs.add(curr_idx) # set of indices in Detections that are matched

            try:
                new_north_max, new_east_min, new_height, new_width = self.get_common_raster_window_params(prev_floe.raster, curr_floe.raster)
                prev_image = self.create_comparable_image(prev_floe.raster, new_north_max, new_east_min, new_height, new_width)
                curr_image = self.create_comparable_image(curr_floe.raster, new_north_max, new_east_min, new_height, new_width)
                
                # morphological operations to remove noise
                kernel, _ = self.get_kernel(prev_image)

                # Opening and closing to remove sharp pixels
                prev_image = self.close_and_open(prev_image, kernel)
                curr_image = self.close_and_open(curr_image, kernel)

                # Blurring around the edges. Dialate to make the blur border larger
                prev_image = self.apply_blur(prev_image, kernel)
                curr_image = self.apply_blur(curr_image, kernel)


                # OPTION 1: Optical Flow
                avg_du, avg_dv, good_prev, good_curr = self.optical_flow(prev_image, curr_image)
                fplt.visualize_optical_flow(prev_image, curr_image, avg_du, avg_dv, good_prev, good_curr)

                prev_meas_x, prev_meas_y = prev_floe.measurements[-1]
                curr_x = prev_meas_x + avg_du * prev_floe.raster.resolution
                curr_y = prev_meas_y - avg_dv * prev_floe.raster.resolution

                # updating the variables for the matched floe
                prev_floe.measurement = (curr_x, curr_y)
                prev_floe.measurements.append((curr_x, curr_y)) # add the new measurement to the list of measurements

            # if optical flow fails
            except Exception as e:
                print("Error in finding measurement", e)
                prev_floe.measurement = None
            
            # update the raster
            prev_floe.raster = curr_floe.raster
            curr_floe.trash() # dont need the detected Floe object anymore
    
        return assigned_prev_floe_IDs, assigned_curr_floe_idxs # return the matched indexes as sets
    
    # Handles floes that were not matched. if that floe was an active, make it lost
    def handle_unmatched_floes(self, assigned_prev_floe_IDs):
        unmatched_prev_floe_IDs = set(self.active_floes.keys()) - assigned_prev_floe_IDs # the IDs that were not found in current image

        for unmatched_ID in unmatched_prev_floe_IDs:
            print("Floe lost, removing it")

            floe = self.active_floes[unmatched_ID]
            floe.measurement = None  # Remove old measurement

            self.lost_floes[floe.id] = floe
            del self.active_floes[floe.id]
            
    # Creates new floes for detections that were not matched
    def create_new_floes(self, assigned_curr_floe_idxs, detections):
        unmatched_detection_idxs = set(range(len(detections))) - assigned_curr_floe_idxs

        for idx in unmatched_detection_idxs:
            print("Created new floe")
            new_floe = detections[idx] # Floe object
            new_floe.id = self.generate_id()
            self.active_floes[new_floe.id] = new_floe # add it to active_floes
    
    def update_floes_old(self, curr_raster):
        detections = self.get_detections(curr_raster) # List of Floe objects

        # first time spotting floes, we add them all to self.floes
        if len(self.active_floes) == 0:
            for detection in detections:
                detection.id = self.generate_id()  # set the id
                self.active_floes[detection.id] = detection # add every floe to active_floes
            return
        
        # active floes
        floe_ids = list(self.active_floes.keys())
        floe_list = list(self.active_floes.values())

        cost_matrix = self.get_cost_matrix(floe_list, detections)
        prev_match, curr_match = linear_sum_assignment(cost_matrix) # match floes based on cost matrix

        assigned_prev_floe_IDs, assigned_curr_floe_idxs = self.update_matched_floes(prev_match, curr_match, detections, floe_ids, cost_matrix)
        self.handle_unmatched_floes(assigned_prev_floe_IDs)  # Handle floes that were not matched
        self.create_new_floes(assigned_curr_floe_idxs, detections) # Create new floes for unmatched detections

        return cost_matrix
    
    # TODO: remeber to comment
    def associate(self, curr_raster):
        detections = self.get_detections(curr_raster) # List of Floe objects

        # first time spotting floes, we add them all to self.floes
        if len(self.active_floes) == 0:
            for detection in detections:
                detection.id = self.generate_id()  # set the id
                self.active_floes[detection.id] = detection # add every floe to active_floes
            return
        
        # logic for understanding which floes are kept, which are lost and which are new
        floe_ids = list(self.active_floes.keys())
        floe_list = list(self.active_floes.values())

        cost_matrix = self.get_cost_matrix(floe_list, detections)

        # if all entry is float('inf') there are no matches, so mark all the old as lost, and all detected as new
        if not np.any(np.isfinite(cost_matrix)):
            print("All entries in cost matrix are inf, so skipping assignment")
            self.handle_unmatched_floes(set())  # noe old floes are assigned
            self.create_new_floes(set(range(len(detections))), detections)  # All detections are new
            return

        # Filter to only rows and cols that have finite values
        valid_rows = np.any(np.isfinite(cost_matrix), axis=1) # rows that have at least one valid entry
        valid_cols = np.any(np.isfinite(cost_matrix), axis=0) # columns that have at least one valid entry

        filtered_cost_matrix = cost_matrix[valid_rows][:, valid_cols] # keeps only valid rows and cols
        # print("Filtered cost matrix:", filtered_cost_matrix)
        valid_floe_ids = [floe_ids[i] for i, v in enumerate(valid_rows) if v]
        valid_detection_idxs = [j for j, v in enumerate(valid_cols) if v]

        # do the Hungarian algorithm to only feasible parts
        prev_match, curr_match = linear_sum_assignment(filtered_cost_matrix)

        # Map back to full indices
        assigned_prev_floe_IDs = {valid_floe_ids[i] for i in prev_match}
        assigned_curr_floe_idxs = {valid_detection_idxs[j] for j in curr_match}
        # ------------------------------

        self.update_matched_floes(prev_match, curr_match, detections, valid_floe_ids, cost_matrix)
        self.handle_unmatched_floes(assigned_prev_floe_IDs)
        self.create_new_floes(assigned_curr_floe_idxs, detections)

    def backtrack_floe(self, floe_ID, added_at_time, backtrack_steps=10):
        floe = self.active_floes.get(floe_ID) or self.lost_floes.get(floe_ID)
        if floe is None:
            print(f"Floe ID {floe_ID} not found.")
            return

        last_meas = floe.measurements[-1]
        x0 = np.array([[last_meas[0]],
                    [last_meas[1]],
                    [0.0],
                    [0.0]])

        # Use main Kalman filter in reverse
        reverse_kf = Kalman_Filter(x0)
        A_inv = np.linalg.inv(reverse_kf.A)
        floe.backtracked_states = [x0]
        floe.backtracked_P_matrices = [reverse_kf.P.copy()]

        # part 1, use pos measurements
        for t in reversed(range(1, len(floe.measurements))):
            reverse_kf.predict(A_inv)
            pos_meas = floe.measurements[t - 1]
            y = np.array([[pos_meas[0]], [pos_meas[1]]])
            reverse_kf.correct(y)

            floe.backtracked_states.append(reverse_kf.x.copy())
            floe.backtracked_P_matrices.append(reverse_kf.P.copy())

            if t == added_at_time:
                break

        # part 2, use vel meas
        transformer = Transformer.from_crs("EPSG:6052", "EPSG:32630", always_xy=True)
        for t in range(backtrack_steps):
            reverse_kf.predict(A_inv)

            px_6052, py_6052 = reverse_kf.x[0, 0], reverse_kf.x[1, 0] # in EPSG:6052
            px_32630, py_32630 = transformer.transform(px_6052, py_6052) # in EPSG:32630
            i, j = self.grid.get_grid_index(px_32630, py_32630)
            vel, dist = self.grid.get_velocity_and_distance(i, j)

            if vel is not None:
                R_vel = np.array([[0.2 * (1 + 0.01 * dist), 0],
                                [0, 0.2 * (1 + 0.01 * dist)]])
                C_vel = np.array([[0, 0, 1, 0],
                                [0, 0, 0, 1]])
                y_vel = np.array([[vel[0]], [vel[1]]])
                reverse_kf.correct(y_vel, C=C_vel, R=R_vel)

            floe.backtracked_states.append(reverse_kf.x.copy())
            floe.backtracked_P_matrices.append(reverse_kf.P.copy())


class Kalman_Filter:
    def __init__(self, x0):
        T = 5
        sigma_a = 0.0001
        self.x = x0
        self.T = T
        self.A = np.array([[1, 0, T, 0],
                           [0, 1, 0, T],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.A_inv = np.linalg.inv(self.A)
        self.B = np.zeros((4, 2))
        self.C = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = np.array([[T**4 / 4, 0, T**3 / 2, 0],
                           [0, T**4 / 4, 0, T**3 / 2],
                           [T**3 / 2, 0, T**2, 0],
                           [0, T**3 / 2, 0, T**2]]) * sigma_a**2
        self.R = np.array([[1, 0],
                           [0, 1]])
        self.P = np.array([[3, 0, 0, 0],
                            [0, 3, 0, 0],
                            [0, 0, 0.3**2, 0],
                            [0, 0, 0, 0.3**2]])

    def predict(self, A = None):
        A = A if A is not None else self.A
        self.x = A @ self.x
        self.P = A @ self.P @ A.T + self.Q
        return self.x

    def correct(self, y, C=None, R=None):
        C = C if C is not None else self.C
        R = R if R is not None else self.R

        S = C @ self.P @ C.T + R
        L = self.P @ C.T @ np.linalg.inv(S)

        self.x = self.x + L @ (y - C @ self.x)
        self.P = (np.eye(self.P.shape[0]) - L @ C) @ self.P

        return self.x

class Raster:
    def __init__(self, image, resolution, north_max, east_min): 
        self.image = image
        self.resolution = resolution
        self.north_max = north_max
        self.east_min = east_min

class Floe:
    def __init__(self, measured_pos, raster): # pos in world coord.
        self.id = None # ID of the floe to be set later
        self.raster = raster # Raster containing shape of floe
        
        # The first measurement is set as initial state
        x0 = np.array([[measured_pos[0]],   # x
                    [measured_pos[1]],      # y
                    [0],                    # vx
                    [0]])                   # vy
        
        self.state = x0
        self.predicted_state = None
        self.measurement = measured_pos # (x, y) in world coord.

        self.measurements = [measured_pos] # store all measurements for backtracking
        self.kalmanfilter = Kalman_Filter(x0)

        self.states = [x0.copy()] # store states for when the floe has been corrected
        self.predicted_states = [] # store states for when the floes has been predicted only
        self.P_matrices = [self.kalmanfilter.P.copy()]
        self.predicted_P_matrices = []
        self.backtracked_states = [] # used for storing backtracked states
        self.backtracked_P_matrices = [] # used for storing the P matrces as we backtrack the floe

        self.lost_frames = 0
    
    # trashing a floe for saving memory. Dont know if needed really
    def trash(self):
        self.id = None
        self.kalmanfilter = None
        self.raster = None
        self.measurement = None
        self.state = None
        self.predicted_state = None
        self.path = None