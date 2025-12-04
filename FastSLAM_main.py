#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2
import random
import time
from FastSlamPf import ParticleFilter, FastSLAMParticle, sigmoid, logit
from robot_simulator import RobotSim
# from robot import RobotModel

random.seed(42)
np.random.seed(42)


if __name__ == "__main__":
    # Control commands
    dx = 3.0  # pixels
    dtheta = 1.0  # degrees
    # Noise parameters
    noise_dist = 0.5
    noise_rot = 0.1
    noise_sensor = 0.3
    
    robot_params = {
        'x': 30,
        'y': 30,
        'theta': 0,
        'sigmaDTheta': noise_rot,
        'sigmaDx': noise_dist,
        'sigmaSensor': noise_sensor
    }
    sim = RobotSim(**robot_params)

    # Initial coordinates : (x, y, theta) = (30, 30, 0)
    # x and y: pixels
    # theta: degrees

    # Initialize particle filter
    num_particles = 30  # Reduced for speed
    particle_filter = ParticleFilter(
        N=num_particles,
        particle_cls=FastSLAMParticle,
        x_initial=30,
        y_initial=30,
        theta_initial=0.0,
        x_range=[-1, 1],
        y_range=[-1, 1],
        theta_range=[-0.1, 0.1],
        local_map_size=50,
        map_width=500,
        map_height=500,
        motion_noise=noise_dist,
        turn_noise=np.radians(noise_rot*0.8),
        measurement_noise=noise_sensor*0.8,
        prior_prob=0.5
    )
    
    # Motion and sensor noise parameters
    resample_threshold = num_particles / 2

    plt.ion()
    i = 0
    
    # Pre-create figure for faster updates
    # 4 subplots
    fig = plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    
    best_particle_poses = []
    ground_truth_poses = []

    while True:
        
        # print('iteration ', i)
        start_time = time.time()
        try:
            data, coordGT = sim.commandAndGetData(dx, dtheta)
            # data: scan of the surrounding of the robot
            #       (50x50 image in robot frame)
            # coordGT: Ground Truth of the actual position
            #           of the robot
            # Parameters of the function: dx and dtheta, control
            #                               command of the robot
        except Exception as e:
            print(repr(e))
            break
        
        # FastSLAM Prediction Step: Apply motion model
        particle_filter.predict((dx, np.radians(dtheta)))
    
        # FastSLAM Update Step: Incorporate sensor measurement
        particle_filter.update(data, resample_threshold=0.5, alpha=1.0)
        
        # Get best particle for visualization
        best_particle, weights = particle_filter.get_best_particle()
        
        # print(f"Processing time: {time.time() - start_time:.3f}s")
        
        # print(f"Best particle pose: ({best_particle.x:.1f}, {best_particle.y:.1f}, {np.degrees(best_particle.theta):.1f}°)")
        # print(f"Ground truth pose: ({coordGT[0]:.1f}, {coordGT[1]:.1f}, {coordGT[2]:.1f}°)")
        
        best_particle_poses.append((best_particle.x, best_particle.y, best_particle.theta))
        ground_truth_poses.append(coordGT)
        
        # Update visualization only every N iterations for speed
        if i % 1 == 0:  # Can change to % 2 or % 3 for even faster updates
            sim.map[int(coordGT[0]), int(coordGT[1])] = 0.5
            
            ax1.clear()
            ax1.set_title('Ground Truth Map')
            ax1.imshow(sim.map, interpolation="None", vmin=0, vmax=1)
            ax1.plot(coordGT[1], coordGT[0], 'ro', markersize=5)
            
            ax2.clear()
            ax2.set_title('Sensor Data')
            ax2.imshow(data, interpolation="None", vmin=0, vmax=1)
            
            ax3.clear()
            ax3.set_title('Best Particle')
            estimated_map_prob = sigmoid(best_particle.occ_map)
            ax3.imshow(estimated_map_prob, interpolation="None", vmin=0, vmax=1)
            ax3.plot(best_particle.y, best_particle.x, 'go', markersize=5)
            
            # simplified: do not redraw per-iteration full theta history (avoid slowdown)
            ax4.clear()
            ax4.set_title('Theta (deg) latest')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Theta (deg)')
            # plot only latest values for lightweight live feedback
            ax4.plot(i, np.degrees(best_particle.theta), 'go')
            ax4.plot(i, coordGT[2], 'ro')
            ax4.set_xlim(0, 500)      # static x-axis (iterations) -- adjust if you expect more iterations
            ax4.set_ylim(-180, 180)   # static y-axis for theta in degrees
            
            plt.draw()
            plt.pause(0.01)
        
        i += 1

    plt.ioff()
    plt.show()

    # --- static plots after the run (full history, fixed axes) ---
    if len(best_particle_poses) > 0:
        bp = np.array(best_particle_poses)            # shape (T,3) -> (x,y,theta)
        gt = np.array(ground_truth_poses)             # shape (T,3) -> (x,y,theta_degrees)

        fig2, (ax_xy, ax_theta) = plt.subplots(1, 2, figsize=(12, 5))

        # Trajectory: x vs y (fixed axis to map size)
        ax_xy.plot(bp[:, 0], bp[:, 1], 'g-', label='Estimate (x,y)')
        ax_xy.plot(gt[:, 0], gt[:, 1], 'r--', label='Ground Truth (x,y)')
        ax_xy.set_title('Trajectory')
        ax_xy.set_xlabel('x')
        ax_xy.set_ylabel('y')
        ax_xy.set_xlim(0, 500)   # static limits matching map size
        ax_xy.set_ylim(0, 500)
        ax_xy.set_aspect('equal')
        ax_xy.legend()
        ax_xy.grid(True)

        # Theta vs iteration (fixed y-limits)
        iters = np.arange(bp.shape[0])
        ax_theta.plot(iters, np.degrees(bp[:, 2]), 'g-', label='Estimate theta (deg)')
        ax_theta.plot(iters, gt[:, 2], 'r--', label='GT theta (deg)')
        ax_theta.set_title('Theta over time')
        ax_theta.set_xlabel('Iteration')
        ax_theta.set_ylabel('Theta (deg)')
        ax_theta.set_xlim(0, max(50, bp.shape[0]))  # static-ish x limit, expands if few iters
        ax_theta.set_ylim(-180, 180)                # fixed y-limits
        ax_theta.legend()
        ax_theta.grid(True)

        plt.tight_layout()
        plt.show()