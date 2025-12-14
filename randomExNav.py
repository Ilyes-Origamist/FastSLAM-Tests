#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random

# = RANDOMEX-LIKE NAVIGATION LOGIC =
class RandomExNavigator:
    """
    Implements a simple random navigation strategy inspired by RandomEx.
    The robot moves forward until an obstacle is detected, then chooses a new random direction.
    """
    def __init__(self, step_size=5, turn_speed_deg=20):
        self.step_size = step_size
        self.turn_speed_deg = turn_speed_deg
        # initial random heading
        self.current_heading = random.uniform(0, 2 * np.pi)  # in radians

    def random_direction(self):
        """Faithful to randomex: returns a random heading angle in radians."""
        theta = np.random.rand() * 2 * np.pi
        return theta

    def detect_obstacle(self, sensor_img, threshold=0.6):
        """
        RandomEx tests obstacles in a future 'patch'.
        Since RobotSim returns a rotated 50x50 local map:

        - Forward = TOP of image
        - Check a forward window for any strong obstacle pixel.

        Equivalent to randomex's: sense_local_obstacles(grid, nx, ny)
        """
        forward_region = sensor_img[0:15, 12:38]   # center-top slice
        return np.any(forward_region > threshold)

    def choose_new_direction(self):
        """RandomEx-style: pick a new global random heading (no bias)."""
        return self.random_direction()

    def compute_commands(self, sensor_data, robot_heading):

        obstacle = self.detect_obstacle(sensor_data)
        if obstacle:
            # Obstacle detected → choose a new random heading
            self.current_heading = self.choose_new_direction()
            forward_dx = 0.0
            turn_angle_deg = np.degrees(self.current_heading) - robot_heading
            # Normalize turn
            if turn_angle_deg > 180: turn_angle_deg -= 360
            if turn_angle_deg < -180: turn_angle_deg += 360
        else:
            # Safe path → continue in current random direction
            forward_dx = self.step_size
            turn_angle_deg = 0.0    # keep heading
        
        # Convert heading change into movement command
        # RobotSim expects dtheta in degrees
        # Limit rotation each step (so large heading changes take multiple steps)
        if abs(turn_angle_deg) > self.step_size:
            dtheta = np.sign(turn_angle_deg) * self.step_size
            forward_dx = 0.0   # rotate in place if large correction needed
        else:
            dtheta = turn_angle_deg

        dx = forward_dx
        return dx, dtheta
