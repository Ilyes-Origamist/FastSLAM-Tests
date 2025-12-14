"""
Unit tests for FastSLAM particle operations
"""

import unittest
import numpy as np
from src.core import FastSLAMParticle, sigmoid, logit


class TestSigmoidLogit(unittest.TestCase):
    """Test probability-logodds conversions"""
    
    def test_sigmoid_neutral(self):
        """Sigmoid of 0 should be 0.5"""
        result = sigmoid(0)
        self.assertAlmostEqual(result, 0.5)
    
    def test_logit_neutral(self):
        """Logit of 0.5 should be 0"""
        result = logit(0.5)
        self.assertAlmostEqual(result, 0.0)
    
    def test_sigmoid_logit_inverse(self):
        """Sigmoid and logit should be inverses"""
        p = 0.7
        result = sigmoid(logit(p))
        self.assertAlmostEqual(result, p)


class TestParticleMotion(unittest.TestCase):
    """Test particle motion model"""
    
    def setUp(self):
        """Create particle before each test"""
        self.particle = FastSLAMParticle(
            x=10, y=10, theta=0,
            map_width=100, map_height=100,
            motion_noise=0.1, turn_noise=0.01
        )
    
    def test_initial_position(self):
        """Test particle initializes correctly"""
        self.assertEqual(self.particle.x, 10)
        self.assertEqual(self.particle.y, 10)
        self.assertEqual(self.particle.theta, 0)
    
    def test_forward_motion(self):
        """Test moving forward updates position"""
        initial_x = self.particle.x
        self.particle.sample_motion((5, 0))  # Move 5 forward, no rotation
        
        # Should have moved (noise makes it approximate)
        self.assertNotEqual(self.particle.x, initial_x)
    
    def test_rotation(self):
        """Test rotation updates theta"""
        initial_theta = self.particle.theta
        self.particle.sample_motion((0, np.pi/4))  # Rotate 45 degrees
        
        self.assertNotEqual(self.particle.theta, initial_theta)
    
    def test_theta_wrapping(self):
        """Test theta stays in [0, 2π]"""
        self.particle.sample_motion((0, 3*np.pi))  # Large rotation
        
        self.assertGreaterEqual(self.particle.theta, 0)
        self.assertLess(self.particle.theta, 2*np.pi)


class TestParticleMap(unittest.TestCase):
    """Test occupancy map operations"""
    
    def setUp(self):
        self.particle = FastSLAMParticle(
            x=50, y=50, theta=0,
            map_width=100, map_height=100
        )
    
    def test_map_initialization(self):
        """Map should start as neutral (log-odds = 0)"""
        self.assertTrue(np.all(self.particle.occ_map == 0))
    
    def test_map_bounds(self):
        """Map should clip to prevent overflow"""
        # Simulate many observations of occupied space
        for _ in range(100):
            self.particle.occ_map[50, 50] += 1
        
        self.assertLessEqual(self.particle.occ_map[50, 50], 20)


if __name__ == '__main__':
    unittest.main()
