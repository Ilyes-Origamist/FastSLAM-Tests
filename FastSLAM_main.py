#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import random
import time
from FastSlamPf import ParticleFilter, FastSLAMParticle, sigmoid, logit
from teleop import TeleoperationController
from robot_simulator import RobotSim
# from robot import RobotModel

random.seed(42)
np.random.seed(42)


# wrapper for motion strategy


if __name__ == "__main__":
    # Toggle between teleoperation and predefined commands
    # use_teleop = True  # Set to False to use predefined dx and dtheta
    
    motion_strat = 'teleop' # or 'random_nav', or 'predefined'
    use_teleop = (motion_strat == 'teleop')
    use_random_nav = (motion_strat == 'random_nav')
    
    # Control commands
    dx = 2.0  # pixels (will be overridden by teleop)
    dtheta = 0.0  # degrees (will be overridden by teleop)
    
    # Noise parameters
    noise_dist = 0.5
    noise_rot = 0.1
    noise_sensor_angle = 0.3
    noise_sensor = 0.1
    # Initialize robot simulator
    robot_params = {
        'x': 30,
        'y': 30,
        'theta': 0,
        'sigmaDTheta': noise_rot,
        'sigmaDx': noise_dist,
        'sigmaSensorAngle': noise_sensor_angle,
        'sensorNoiseRatio': noise_sensor,
        'sizeSensor': 50
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
    fig = plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    
    # Connect keyboard events
    if use_teleop:
        # Initialize teleoperation controller
        teleop = TeleoperationController(default_speed=2.0, default_turn=5.0)
        fig.canvas.mpl_connect('key_press_event', teleop.on_key_press)
        fig.canvas.mpl_connect('key_release_event', teleop.on_key_release)
        print("Starting teleoperation mode. Use arrow keys to control the robot.")
        print("Click on the figure window to ensure it has focus for keyboard input.\n")
    else:
        print("Starting predefined control mode.")
        print(f"Using dx={dx}, dtheta={dtheta}\n")

    best_particle_poses = []
    ground_truth_poses = []

    while True:
        
        # Compute control commands

        # Get control command from teleoperation
        if use_teleop:
            # If paused, skip sending commands
            if dx == 0.0 and dtheta == 0.0 and teleop.paused:
                plt.pause(0.05)
                continue
            dx, dtheta = teleop.get_command()
        elif motion_strat == 'random_nav':
            pass

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
        
        best_particle_poses.append((best_particle.x, best_particle.y, best_particle.theta))
        ground_truth_poses.append(coordGT)
        
        # Update visualization
        if i % 1 == 0:
            sim.map[int(coordGT[0]), int(coordGT[1])] = 0.5
            
            ax1.clear()
            mode_label = teleop.mode.upper() if use_teleop else 'AUTO'
            ax1.set_title(f'Ground Truth Map [{mode_label}]')
            ax1.imshow(sim.map, interpolation="None", vmin=0, vmax=1)
            ax1.plot(coordGT[1], coordGT[0], 'ro', markersize=5)
            # Show control info
            ax1.text(10, 480, f'dx: {dx:.1f}  dθ: {dtheta:.1f}°', 
                    color='white', fontsize=10, bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            ax2.clear()
            ax2.set_title('Sensor Data')
            ax2.imshow(data, interpolation="None", vmin=0, vmax=1)
            
            ax3.clear()
            ax3.set_title('Best Particle Map')
            estimated_map_prob = sigmoid(best_particle.occ_map)
            ax3.imshow(estimated_map_prob, interpolation="None", vmin=0, vmax=1)
            ax3.plot(best_particle.y, best_particle.x, 'go', markersize=5)
            
            ax4.clear()
            ax4.set_title('Particle Distribution')
            ax4.set_xlabel('X (pixels)')
            ax4.set_ylabel('Y (pixels)')
            
            # Plot all particles
            particle_x = [p.x for p in particle_filter.particles]
            particle_y = [p.y for p in particle_filter.particles]
            ax4.scatter(particle_x, particle_y, c='blue', s=20, alpha=0.5, label='Particles')
            
            # Plot best particle
            ax4.plot(best_particle.x, best_particle.y, 'go', markersize=10, label='Best Est', markeredgecolor='black')
            
            # Plot ground truth
            ax4.plot(coordGT[0], coordGT[1], 'ro', markersize=10, label='GT', markeredgecolor='black')
            
            ax4.set_xlim(0, 500)
            ax4.set_ylim(0, 500)
            ax4.set_aspect('equal')
            if i == 0:
                ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.draw()
            plt.pause(0.01)
        
        i += 1

    plt.ioff()
    plt.show()

    # --- static plots after the run (full history, fixed axes) ---
    if len(best_particle_poses) > 0:
        bp = np.array(best_particle_poses)
        gt = np.array(ground_truth_poses)

        # Calculate errors
        pos_error = np.sqrt((bp[:, 0] - gt[:, 0])**2 + (bp[:, 1] - gt[:, 1])**2)
        theta_error = np.abs(np.degrees(bp[:, 2]) - gt[:, 2])
        theta_error = np.minimum(theta_error, 360 - theta_error)
        
        print(f"\n=== Tracking Performance ===")
        print(f"Mean position error: {np.mean(pos_error):.2f} pixels")
        print(f"Max position error: {np.max(pos_error):.2f} pixels")
        print(f"Mean theta error: {np.mean(theta_error):.2f}°")
        print(f"Max theta error: {np.max(theta_error):.2f}°")

        fig2, ((ax_xy, ax_theta), (ax_pos_err, ax_theta_err)) = plt.subplots(2, 2, figsize=(12, 10))

        # Trajectory
        ax_xy.plot(bp[:, 0], bp[:, 1], 'g-', label='Estimate (x,y)', linewidth=2)
        ax_xy.plot(gt[:, 0], gt[:, 1], 'r--', label='Ground Truth (x,y)', linewidth=1)
        ax_xy.scatter(bp[0, 0], bp[0, 1], c='green', s=100, marker='o', label='Start')
        ax_xy.scatter(bp[-1, 0], bp[-1, 1], c='blue', s=100, marker='*', label='End')
        ax_xy.set_title('Trajectory')
        ax_xy.set_xlabel('x')
        ax_xy.set_ylabel('y')
        ax_xy.set_xlim(0, 500)
        ax_xy.set_ylim(0, 500)
        ax_xy.set_aspect('equal')
        ax_xy.legend()
        ax_xy.grid(True)

        # Theta vs iteration
        iters = np.arange(bp.shape[0])
        ax_theta.plot(iters, np.degrees(bp[:, 2]), 'g-', label='Estimate theta (deg)', linewidth=2)
        ax_theta.plot(iters, gt[:, 2], 'r--', label='GT theta (deg)', linewidth=1)
        ax_theta.set_title('Theta over time')
        ax_theta.set_xlabel('Iteration')
        ax_theta.set_ylabel('Theta (deg)')
        ax_theta.set_xlim(0, max(50, bp.shape[0]))
        ax_theta.set_ylim(-180, 180)
        ax_theta.legend()
        ax_theta.grid(True)

        # Position error
        ax_pos_err.plot(iters, pos_error, 'b-', linewidth=2)
        ax_pos_err.set_title(f'Position Error (mean: {np.mean(pos_error):.2f} px)')
        ax_pos_err.set_xlabel('Iteration')
        ax_pos_err.set_ylabel('Error (pixels)')
        ax_pos_err.set_xlim(0, max(50, bp.shape[0]))
        ax_pos_err.grid(True)

        # Theta error
        ax_theta_err.plot(iters, theta_error, 'm-', linewidth=2)
        ax_theta_err.set_title(f'Theta Error (mean: {np.mean(theta_error):.2f}°)')
        ax_theta_err.set_xlabel('Iteration')
        ax_theta_err.set_ylabel('Error (degrees)')
        ax_theta_err.set_xlim(0, max(50, bp.shape[0]))
        ax_theta_err.set_ylim(0, 180)
        ax_theta_err.grid(True)

        plt.tight_layout()
        plt.show()