import cv2
import numpy as np

# Simulated Raster container
class Raster:
    def __init__(self, image, resolution, east_min, north_max):
        # in:  image [2D array], resolution [m/pixel], east_min [m], north_max [m]
        # out: Raster object
        self.image = image                   # 2D numpy array (binary image)
        self.resolution = resolution         # resolution in meters/pixel
        self.east_min = east_min             # EPSG:6052 [m]
        self.north_max = north_max           # EPSG:6052 [m]

# Simulated ice floe object
class Floe:
    def __init__(self, contour):
        # in:  contour [Nx2 int32 array]
        # out: Floe object
        self.id = None                      # will be assigned by Simulator
        self.contour = contour              # shape of the floe [pixels]

        self.GT_states = []                 # list of true states (position + velocity) [pixels]
        self.predicted_GT_states = []      # list of predicted GT states [pixels]
        self.noisy_states = []             # states as seen by sensor [pixels]
        self.GT_backtracked_states = []    # reconstructed earlier states [pixels]

    def get_center(self):
        # Calculates the mass center of the floe
        # out: (u, v) center [pixels] or None if invalid
        M = cv2.moments(self.contour)
        if M["m00"] != 0:
            cu = int(M["m10"] / M["m00"])
            cv = int(M["m01"] / M["m00"])
            return (cu, cv)
        return None

    def move(self, du, dv):
        # Moves the floe by du, dv pixels
        # in:  du, dv [int]
        self.contour = self.contour + np.array([du, dv])
        self.contour = self.contour.astype(np.int32)

    def add_GT_state(self, dt=5):
        # Adds the floe's current center + velocity to GT_states
        # in:  dt [seconds]
        center_u, center_v = self.get_center()

        if len(self.GT_states) == 0:
            vel_u, vel_v = 0.0, 0.0  # first state has 0 velocity
        else:
            last = self.GT_states[-1]
            prev_u = last[0, 0]
            prev_v = last[1, 0]
            vel_u = (center_u - prev_u) / dt
            vel_v = (center_v - prev_v) / dt

        state = np.array([[center_u], [center_v], [vel_u], [vel_v]])
        self.GT_states.append(state)

    def add_predicted_GT_state(self, du, dv, dt=5):
        # Adds a predicted GT state given motion
        # in:  du, dv [pixels], dt [seconds]
        if len(self.predicted_GT_states) == 0:
            last = self.GT_states[-1]
        else:
            last = self.predicted_GT_states[-1]

        prev_u = last[0, 0]
        prev_v = last[1, 0]
        vel_u = du / dt
        vel_v = dv / dt

        state = np.array([[prev_u + du], [prev_v + dv], [vel_u], [vel_v]])
        self.predicted_GT_states.append(state)

    def add_noisy_state(self, noisy_center, dt=5):
        # Adds a noisy (sensor) observation of the floe's center
        # in:  noisy_center [tuple], dt [seconds]
        center_u, center_v = noisy_center

        if len(self.noisy_states) == 0:
            vel_u, vel_v = 0.0, 0.0
        else:
            last = self.noisy_states[-1]
            prev_u = last[0, 0]
            prev_v = last[1, 0]
            vel_u = (center_u - prev_u) / dt
            vel_v = (center_v - prev_v) / dt

        state = np.array([[center_u], [center_v], [vel_u], [vel_v]])
        self.noisy_states.append(state)

    def print(self):
        # Debugging info
        print("Simulated Floe ID:", self.id)

# Simulator for generating and evolving floe scenes
class Simulator:
    def __init__(self):
        # out: Simulator object initialized
        self.resolution = 0.1      # [m/pixel]
        self.east_min = 0          # EPSG:6052 [m]
        self.east_max = 100        # EPSG:6052 [m]
        self.north_min = 0         # EPSG:6052 [m]
        self.north_max = 200       # EPSG:6052 [m]

        self.GT_map = self.create_empty_GT_map()
        self.active_floes = {}     # id -> Floe
        self.lost_floes = {}       # id -> Floe (removed)

        self.next_id = 0
        self.motion_history = []   # list of (du, dv) applied each step [pixels]

    def generate_id(self):
        # Returns unique floe ID
        # out: int
        current_id = self.next_id
        self.next_id += 1
        return current_id

    def custom_round(self, x):
        # Rounds away from zero
        # in:  float x
        # out: int
        return int(np.sign(x) * np.ceil(abs(x)))

    def meters_to_pixels(self, x, y):
        # Converts world coords [m] to image coords [pixels]
        # in:  x, y [m]
        # out: u, v [pixels]
        u = int(round((x - self.east_min) / self.resolution))
        v = int(round((self.north_max - y) / self.resolution))
        return u, v

    def pixels_to_meters(self, u, v):
        # Converts image coords to world coords
        # in:  u, v [pixels]
        # out: x, y [m]
        x = u * self.resolution + self.east_min
        y = self.north_max - v * self.resolution
        return x, y

    def gt_state_pixels_to_meters(self, state_pixel):
        # Converts GT state [pixels] to meters
        # in:  state_pixel [4x1]
        # out: state_meter [4x1]
        u, v = state_pixel[0, 0], state_pixel[1, 0]
        vu, vv = state_pixel[2, 0], state_pixel[3, 0]

        x = u * self.resolution + self.east_min
        y = self.north_max - v * self.resolution
        vx = vu * self.resolution
        vy = -vv * self.resolution

        return np.array([[x], [y], [vx], [vy]])

    def add_GT_states(self):
        # Updates GT states for all active floes
        for floe in self.active_floes.values():
            floe.add_GT_state()

    def get_GT_backtracked_states(self, floe_ID, added_at_time, T=5):
        # Reconstructs past GT states using motion history
        # in:  floe_ID, added_at_time T [s]
        floe = self.active_floes.get(floe_ID) or self.lost_floes.get(floe_ID)
        first = floe.GT_states[0]
        floe.GT_backtracked_states.append(first)

        for t in reversed(range(added_at_time)):
            du, dv = self.motion_history[t]
            u = floe.GT_backtracked_states[-1][0, 0] - du
            v = floe.GT_backtracked_states[-1][1, 0] - dv
            state = np.array([[u], [v], [du / T], [dv / T]])
            floe.GT_backtracked_states.append(state)

        floe.GT_backtracked_states.pop(0)

    def create_empty_GT_map(self):
        # Creates an empty binary GT image
        # out: GT map
        height = int((self.north_max - self.north_min) / self.resolution)
        width = int((self.east_max - self.east_min) / self.resolution)
        return np.zeros((height, width), dtype=np.uint8)

    def add_new_floe(self, u, v, w, h, num_points=5, max_attempts=20):
        # Tries to add a non-overlapping floe with random shape
        # in:  u,v,w,h [pixels], num_points
        # out: Floe or None
        MIN_AREA = w * h / 10  # filter out too small regions

        for _ in range(max_attempts):
            points = np.array([
                [u + np.random.randint(0, w), v + np.random.randint(0, h)]
                for _ in range(num_points)
            ], dtype=np.int32)

            center = np.mean(points, axis=0)
            angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
            sorted_points = points[np.argsort(angles)]

            if cv2.contourArea(sorted_points) < MIN_AREA:
                continue

            mask = np.zeros_like(self.GT_map)
            cv2.fillPoly(mask, [sorted_points], color=255)
            overlap = cv2.bitwise_and(self.GT_map, mask)

            if np.count_nonzero(overlap) == 0:
                cv2.fillPoly(self.GT_map, [sorted_points], color=255)
                floe = Floe(sorted_points)
                floe.id = self.generate_id()
                self.active_floes[floe.id] = floe
                return floe

        print("Warning: Could not place a new floe after", max_attempts, "attempts.")
        return None

    def remove_floe(self, floe_id):
        # Removes floe from GT map and adds it to lost floes
        # in:  floe_id
        floe = self.active_floes[floe_id]
        cv2.fillPoly(self.GT_map, [floe.contour], color=0)
        self.lost_floes[floe.id] = floe
        del self.active_floes[floe.id]

    def move_all_floes(self, du, dv, sigma_a, T=5):
        # Applies motion + acceleration noise to all floes
        # in:  du, dv [pixels], sigma_a [std], T [s]
        a_noise_u = np.random.normal(0, sigma_a)
        a_noise_v = np.random.normal(0, sigma_a)

        noise_u = 0.5 * a_noise_u * T**2
        noise_v = 0.5 * a_noise_v * T**2

        du_total = du + noise_u
        dv_total = dv + noise_v

        du_rounded = int(np.sign(du_total) * np.ceil(abs(du_total)))
        dv_rounded = int(np.sign(dv_total) * np.ceil(abs(dv_total)))

        self.motion_history.append((du_rounded, dv_rounded))
        self.GT_map = np.zeros_like(self.GT_map)

        for floe in self.active_floes.values():
            floe.move(du_rounded, dv_rounded)
            cv2.fillPoly(self.GT_map, [floe.contour], color=255)
            floe.add_GT_state()

    def create_raster(self, u_std, v_std, u_mean=0, v_mean=0):
        # Generates a raster image from GT with added pixel noise
        # in:  u_std, v_std, u_mean, v_mean [pixels]
        # out: Raster object
        noisy_image = np.zeros_like(self.GT_map)

        for floe in self.active_floes.values():
            du = int(np.random.normal(u_mean, u_std))
            dv = int(np.random.normal(v_mean, v_std))

            noise = np.array([du, dv])
            noisy_contour = (floe.contour + noise).astype(np.int32)
            noisy_center = floe.get_center() + noise

            floe.add_noisy_state(noisy_center)
            cv2.fillPoly(noisy_image, [noisy_contour], color=255)

        return Raster(noisy_image, self.resolution, self.east_min, self.north_max)
