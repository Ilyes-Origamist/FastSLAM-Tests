# Minimal FastSLAM 1.0 particle filter using occupancy grids (log-odds)
# - Particles store pose (x,y,theta) and an occupancy grid in log-odds
# - Motion update uses noisy odometry
# - Measurement update: 1) compute log-likelihood per particle (no map change)
#                      2) normalize weights, resample
#                      3) update maps (log-odds fusion) for resampled particles

import numpy as np
import math
import copy
from .particle import FastSLAMParticle


# ---------- Particle Filter Class ----------
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
            indexes = self.systematic_resample(weights)
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
            indexes = self.systematic_resample(weights)
            new_particles = [copy.deepcopy(self.particles[i]) for i in indexes]
            self.particles = new_particles
            for p in self.particles:
                p.log_weight = math.log(1.0 / self.N)

    # systematic resampling
    def systematic_resample(self, weights):
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

    def get_best_particle(self):
        # return particle with highest weight
        weights = np.array([math.exp(p.log_weight) for p in self.particles])
        idx = np.argmax(weights)
        return self.particles[idx], weights

