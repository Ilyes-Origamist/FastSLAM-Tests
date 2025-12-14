"""
Integration tests for particle filter
"""

import unittest
import numpy as np
from src.core import ParticleFilter, FastSLAMParticle


class TestParticleFilter(unittest.TestCase):
    """Test complete particle filter pipeline"""
    
    def setUp(self):
        """Create particle filter before each test"""
        self.pf = ParticleFilter(
            N=10,
            particle_cls=FastSLAMParticle,
            x_initial=50,
            y_initial=50,
            theta_initial=0,
            map_width=100,
            map_height=100
        )
    
    def test_initialization(self):
        """Filter should create N particles"""
        self.assertEqual(len(self.pf.particles), 10)
    
    def test_weights_sum_to_one(self):
        """Particle weights should sum to 1"""
        weights = np.array([p.weight for p in self.pf.particles])
        self.assertAlmostEqual(np.sum(weights), 1.0)
    
    def test_predict_changes_poses(self):
        """Prediction step should update particle poses"""
        initial_x = self.pf.particles[0].x
        
        self.pf.predict((5, 0))  # Move forward
        
        # At least one particle should have moved
        moved = any(p.x != initial_x for p in self.pf.particles)
        self.assertTrue(moved)
    
    def test_resampling_preserves_count(self):
        """Resampling should maintain particle count"""
        # Create dummy sensor data
        sensor_data = np.random.rand(50, 50)
        
        self.pf.update(sensor_data, resample_threshold=0.5)
        
        self.assertEqual(len(self.pf.particles), 10)


if __name__ == '__main__':
    unittest.main()
