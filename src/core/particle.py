
import numpy as np
import copy
import math

# ---------- helper functions ----------

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def sigmoid(l):
    return 1.0 / (1.0 + np.exp(-l))

# ---------- Particle class ----------
class FastSLAMParticle:
    def __init__(self, params):
        # pose
        x0 = params.get('x_initial', 30.0)
        y0 = params.get('y_initial', 30.0)
        theta0 = params.get('theta_initial', 0.0)
        xr = params.get('x_range', [0, 0.0])
        yr = params.get('y_range', [0, 0.0])
        tr = params.get('theta_range', [0, 0.0])

        self.x = x0 + np.random.uniform(xr[0], xr[1])
        self.y = y0 + np.random.uniform(yr[0], yr[1])
        self.theta = theta0 + np.random.uniform(tr[0], tr[1])

        # noise params
        self.motion_noise = params.get('motion_noise', 0.1)
        self.turn_noise = params.get('turn_noise', 0.05)
        self.measurement_noise = params.get('measurement_noise', 0.1)  # used for likelihood

        # local map size (square)
        self.local_map_size = params.get('local_map_size', 25)

        # map: store log-odds
        self.map_width = params.get('map_width', 500)
        self.map_height = params.get('map_height', 500)
        prior = params.get('prior_prob', 0.5)
        prior_logodds = logit(prior)
        self.prior_logodds = prior_logodds
        # initialize full map as prior
        self.occ_map = np.full((self.map_width, self.map_height), prior_logodds, dtype=float)

        # weight in log domain for numerical stability
        self.log_weight = math.log(params.get('initial_weight', 1.0))

    def copy(self):
        return copy.deepcopy(self)

    def sample_motion(self, command):
        # command: (distance, rotation_radians) - incremental control
        distance, rotation = command
        noisy_d = distance + np.random.normal(0, self.motion_noise)
        noisy_r = rotation + np.random.normal(0, self.turn_noise)

        # Update rotation (keep in radians internally)
        self.theta = (self.theta + noisy_r) % (2 * np.pi)
        
        # Motion model matching robot_simulator:
        # Robot simulator uses: x += sin(theta)*dx, y += cos(theta)*dx
        # This means theta=0 points in +Y direction
        self.x += np.sin(self.theta) * noisy_d
        self.y += np.cos(self.theta) * noisy_d

    def get_measurement_log_likelihood(self, local_map):
        """
        Compute log-likelihood of observing `local_map` given this particle's map and pose.
        - local_map: numpy array shape (S,S) of occupancy probabilities in robot frame (0..1)
        Returns: scalar log-likelihood
        Note: does NOT modify the particle map.
        """
        S = local_map.shape[0]
        assert S == self.local_map_size, "local_map size mismatch"

        half = S // 2
        sigma = max(self.measurement_noise, 1e-6)

        # The sensor extracts map[x-25:x+25, y-25:y+25] then rotates by (theta+90)
        # To go from sensor pixel to world: inverse rotate by -(theta+90), then offset
        theta_deg = np.degrees(self.theta)
        rotation_angle = -(theta_deg + 90)  # inverse rotation
        cos_r = np.cos(np.radians(rotation_angle))
        sin_r = np.sin(np.radians(rotation_angle))

        # Vectorized coordinate generation
        di_indices = np.arange(S)
        dj_indices = np.arange(S)
        di_grid, dj_grid = np.meshgrid(di_indices, dj_indices, indexing='ij')
        
        # Sensor coordinates relative to center
        dx_sensor = di_grid - half
        dy_sensor = dj_grid - half
        
        # Inverse rotate to get coordinates in robot-centered frame
        dx_robot = dx_sensor * cos_r - dy_sensor * sin_r
        dy_robot = dx_sensor * sin_r + dy_sensor * cos_r
        
        # Translate to world coordinates
        xw = self.x + dx_robot
        yw = self.y + dy_robot
        
        # Round to integer indices
        ix = np.round(xw).astype(int)
        iy = np.round(yw).astype(int)
        
        # Create mask for valid indices AND valid measurements (not unknown/artifact)
        valid_mask = (ix >= 0) & (ix < self.map_width) & (iy >= 0) & (iy < self.map_height)
        # Filter out measurements close to 0.5 (unknown/rotation artifacts)
        measurement_valid = np.abs(local_map - 0.5) > 0.15  # only use confident measurements
        valid_mask = valid_mask & measurement_valid
        
        # Extract valid measurements and map values
        meas_p_valid = local_map[valid_mask]
        map_logodds_valid = self.occ_map[ix[valid_mask], iy[valid_mask]]
        map_p_valid = sigmoid(map_logodds_valid)
        
        # Compute log-likelihood
        diff2 = (meas_p_valid - map_p_valid) ** 2
        log_like = -np.sum(diff2) / (2 * sigma * sigma)
        
        return log_like

    def fuse_map(self, local_map, alpha=1.0):
        """
        Fuse the local measurement into this particle's occupancy map (log-odds update).
        - local_map: occupancy probabilities (S,S)
        - alpha: trust factor (0..1+)
        """
        S = local_map.shape[0]
        half = S // 2
        
        # Same transformation as in likelihood
        theta_deg = np.degrees(self.theta)
        rotation_angle = -(theta_deg + 90)
        cos_r = np.cos(np.radians(rotation_angle))
        sin_r = np.sin(np.radians(rotation_angle))
        
        # Vectorized coordinate generation
        di_indices = np.arange(S)
        dj_indices = np.arange(S)
        di_grid, dj_grid = np.meshgrid(di_indices, dj_indices, indexing='ij')
        
        dx_sensor = di_grid - half
        dy_sensor = dj_grid - half
        
        # Inverse rotate
        dx_robot = dx_sensor * cos_r - dy_sensor * sin_r
        dy_robot = dx_sensor * sin_r + dy_sensor * cos_r
        
        # Translate to world
        xw = self.x + dx_robot
        yw = self.y + dy_robot
        
        ix = np.round(xw).astype(int)
        iy = np.round(yw).astype(int)
        
        # Mask for valid indices AND confident measurements
        valid_mask = (ix >= 0) & (ix < self.map_width) & (iy >= 0) & (iy < self.map_height)
        # Only fuse confident measurements (not rotation artifacts near 0.5)
        measurement_valid = np.abs(local_map - 0.5) > 0.15
        valid_mask = valid_mask & measurement_valid
        
        # Get valid measurements
        meas_p_valid = local_map[valid_mask]
        l_meas_valid = logit(meas_p_valid)
        
        # Compute updates - use proper log-odds fusion
        ix_valid = ix[valid_mask]
        iy_valid = iy[valid_mask]
        
        # Log-odds update: subtract prior first to get measurement evidence only
        current_logodds = self.occ_map[ix_valid, iy_valid]
        measurement_evidence = alpha * (l_meas_valid - self.prior_logodds)
        
        self.occ_map[ix_valid, iy_valid] = np.clip(
            current_logodds + measurement_evidence,
            -20.0, 20.0
        )


# # ---------- minimal example usage ----------
# if __name__ == '__main__':
#     # create a fake local measurement: a small 25x25 patch with a wall on the right side
#     S = 25
#     local = np.full((S, S), 0.5)
#     local[:, -5:] = 0.9  # high occupancy on the right

#     params = {
#         'x_initial': 250,
#         'y_initial': 250,
#         'theta_initial': 0.0,
#         'local_map_size': S,
#         'map_width': 500,
#         'map_height': 500,
#         'motion_noise': 0.2,
#         'turn_noise': 0.05,
#         'measurement_noise': 0.1,
#         'initial_weight': 1.0
#     }

#     pf = ParticleFilter(N=100, particle_cls=FastSLAMParticle, **params)
#     print('Initialized', len(pf.particles), 'particles')

#     # simulate: move forward 5, small turn
#     pf.predict((5.0, 0.1))
#     # update with local measurement
#     pf.update(local, resample_threshold=0.5, alpha=0.8)

#     best, weights = pf.get_best_particle()
#     # print('Best particle pose:', best.x, best.y, best.theta)
#     # print('Top 5 weights:', np.sort(weights)[-5:])
