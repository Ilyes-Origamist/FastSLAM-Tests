# fastslam_grid_particle.py
# Clean, corrected, minimal FastSLAM-like particle filter using occupancy grids (log-odds)
# - Particles store pose (x,y,theta) and an occupancy grid in log-odds
# - Motion update uses noisy odometry
# - Measurement update: 1) compute log-likelihood per particle (no map change)
#                      2) normalize weights, resample
#                      3) update maps (log-odds fusion) for resampled particles

import numpy as np
import copy
import math

# ---------- helper functions ----------

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(l):
    return 1.0 / (1.0 + np.exp(-l))


# systematic resampling
def systematic_resample(weights):
    N = len(weights)
    positions = (np.arange(N) + np.random.rand()) / N
    indexes = np.zeros(N, dtype=int)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


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
        
        # Create mask for valid indices
        valid_mask = (ix >= 0) & (ix < self.map_width) & (iy >= 0) & (iy < self.map_height)
        
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
        
        # Mask for valid indices
        valid_mask = (ix >= 0) & (ix < self.map_width) & (iy >= 0) & (iy < self.map_height)
        
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

# ---------- Particle Filter controller ----------
class ParticleFilter:
    def __init__(self, N, particle_cls, **params):
        self.N = N
        self.particles = [particle_cls(params) for _ in range(N)]
        # initialize equal weights in log domain
        for p in self.particles:
            p.log_weight = math.log(1.0 / N)

    def predict(self, command):
        for p in self.particles:
            p.sample_motion(command)

    def update(self, local_map, resample_threshold=0.5, alpha=1.0):
        # 1) compute log-likelihoods (without modifying maps)
        log_likes = np.array([p.get_measurement_log_likelihood(local_map) for p in self.particles])
        
        # 2) update log-weights: log w = log w + log_likelihood
        log_weights = np.array([p.log_weight for p in self.particles]) + log_likes
        # numeric stability: subtract max
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights_sum = np.sum(weights)
        if weights_sum <= 0:
            weights = np.ones(self.N) / self.N
        else:
            weights /= weights_sum
        
        # store normalized weights back into particles (log domain)
        for i, p in enumerate(self.particles):
            p.log_weight = math.log(max(weights[i], 1e-12))

        # 3) compute Neff and resample if needed
        Neff = 1.0 / np.sum(weights ** 2)
        if Neff < resample_threshold * self.N:
            indexes = systematic_resample(weights)
            # shallow copy particles, deep copy only maps
            new_particles = []
            for idx in indexes:
                p_new = FastSLAMParticle.__new__(FastSLAMParticle)
                p_old = self.particles[idx]
                p_new.x = p_old.x
                p_new.y = p_old.y
                p_new.theta = p_old.theta
                p_new.motion_noise = p_old.motion_noise
                p_new.turn_noise = p_old.turn_noise
                p_new.measurement_noise = p_old.measurement_noise
                p_new.local_map_size = p_old.local_map_size
                p_new.map_width = p_old.map_width
                p_new.map_height = p_old.map_height
                p_new.prior_logodds = p_old.prior_logodds
                p_new.occ_map = p_old.occ_map.copy()  # only deep copy the map
                p_new.log_weight = math.log(1.0 / self.N)
                new_particles.append(p_new)
            self.particles = new_particles
        
        # 4) map update (only after resampling decision)
        for p in self.particles:
            p.fuse_map(local_map, alpha=alpha)

    def update_optimized(self, local_map, resample_threshold=0.5, alpha=1.0):
        """
        Optimized update that computes likelihood and fuses map in a single pass.
        Note: This fuses maps BEFORE resampling, which may update maps of low-weight particles.
        """
        # 1) compute log-likelihoods AND fuse maps in single pass
        log_likes = np.array([p.get_measurement_log_likelihood_and_fuse(local_map, alpha) for p in self.particles])
        # 2) update log-weights
        log_weights = np.array([p.log_weight for p in self.particles]) + log_likes
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights_sum = np.sum(weights)
        if weights_sum <= 0:
            weights = np.ones(self.N) / self.N
        else:
            weights /= weights_sum
        for i, p in enumerate(self.particles):
            p.log_weight = math.log(max(weights[i], 1e-12))

        # 3) resample if needed
        Neff = 1.0 / np.sum(weights ** 2)
        if Neff < resample_threshold * self.N:
            indexes = systematic_resample(weights)
            new_particles = [copy.deepcopy(self.particles[i]) for i in indexes]
            self.particles = new_particles
            for p in self.particles:
                p.log_weight = math.log(1.0 / self.N)

    def get_best_particle(self):
        # return particle with highest weight
        weights = np.array([math.exp(p.log_weight) for p in self.particles])
        idx = np.argmax(weights)
        return self.particles[idx], weights


# ---------- minimal example usage ----------
if __name__ == '__main__':
    # create a fake local measurement: a small 25x25 patch with a wall on the right side
    S = 25
    local = np.full((S, S), 0.5)
    local[:, -5:] = 0.9  # high occupancy on the right

    params = {
        'x_initial': 250,
        'y_initial': 250,
        'theta_initial': 0.0,
        'local_map_size': S,
        'map_width': 500,
        'map_height': 500,
        'motion_noise': 0.2,
        'turn_noise': 0.05,
        'measurement_noise': 0.1,
        'initial_weight': 1.0
    }

    pf = ParticleFilter(N=100, particle_cls=FastSLAMParticle, **params)
    print('Initialized', len(pf.particles), 'particles')

    # simulate: move forward 5, small turn
    pf.predict((5.0, 0.1))
    # update with local measurement
    pf.update(local, resample_threshold=0.5, alpha=0.8)

    best, weights = pf.get_best_particle()
    # print('Best particle pose:', best.x, best.y, best.theta)
    # print('Top 5 weights:', np.sort(weights)[-5:])
