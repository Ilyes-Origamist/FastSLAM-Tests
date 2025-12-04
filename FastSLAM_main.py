#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2
import random
import time
from FastSlamPf import ParticleFilter, FastSLAMParticle, sigmoid, logit
from robot_simulator import RobotSim

random.seed(43)
np.random.seed(43)


# = RANDOMEX-LIKE NAVIGATION LOGIC =

def random_direction():
    """Faithful to randomex: returns a random heading angle in radians."""
    theta = np.random.rand() * 2 * np.pi
    return theta

def detect_obstacle(sensor_img, threshold=0.5):
    """
    RandomEx tests obstacles in a future 'patch'.
    Since RobotSim returns a rotated 50x50 local map:

    - Forward = TOP of image
    - Check a forward window for any strong obstacle pixel.

    Equivalent to randomex's: sense_local_obstacles(grid, nx, ny)
    """
    forward_region = sensor_img[0:12, 10:40]   # center-top slice
    return np.any(forward_region > threshold)

def choose_new_direction():
    """RandomEx-style: pick a new global random heading (no bias)."""
    return random_direction()


# = MAIN PROGRAM =

if __name__ == "__main__":
    # Control parameters (step sizes similar to randomex)
    STEP_SIZE = 6                # forward distance per move
    TURN_SPEED_DEG = 35         # how much a rotation step applies (scaled)
    
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

    # Initialize particle filter
    num_particles = 50
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
    
    resample_threshold = num_particles / 2

    plt.ion()
    i = 0
    
    fig = plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    
    best_particle_poses = []
    ground_truth_poses = []


    # RandomEx Navigation State 
    current_heading = random_direction()  # global heading (radians)



    # /// MAIN LOOP ////////////////////////////////////////////


    # We must read at least one sensor image before navigation decisions.
    # Perform a dummy call to get initial image:

    init_dx = 0.0
    init_dtheta = 0.0
    data, coordGT = sim.commandAndGetData(init_dx, init_dtheta)


    while True:
        
        start_time = time.time()

        
        # RANDOMEX-LIKE DECISION MAKING PRIOR TO MOVEMENT 
        

        obstacle = detect_obstacle(data)

        if obstacle:
            # Obstacle detected → choose a new random heading
            current_heading = choose_new_direction()
            forward_dx = 0.0
            turn_angle_deg = np.degrees(current_heading) - coordGT[2]
            # Normalize turn
            if turn_angle_deg > 180: turn_angle_deg -= 360
            if turn_angle_deg < -180: turn_angle_deg += 360
        else:
            # Safe path → continue in current random direction
            forward_dx = STEP_SIZE
            turn_angle_deg = 0.0    # keep heading
        
        # Convert heading change into movement command
        # RobotSim expects dtheta in degrees
        # Limit rotation each step (so large heading changes take multiple steps)
        if abs(turn_angle_deg) > TURN_SPEED_DEG:
            dtheta = np.sign(turn_angle_deg) * TURN_SPEED_DEG
            forward_dx = 0.0   # rotate in place if large correction needed
        else:
            dtheta = turn_angle_deg

        dx = forward_dx

        
        # APPLY MOVEMENT AND GET SENSOR DATA 
        

        try:
            data, coordGT = sim.commandAndGetData(dx, dtheta)
        except Exception as e:
            print(repr(e))
            break
        
        
        #  FASTSLAM PREDICTION / UPDATE 
        

        particle_filter.predict((dx, np.radians(dtheta)))
        particle_filter.update(data, resample_threshold=0.5, alpha=1.0)
        
        best_particle, weights = particle_filter.get_best_particle()
        
        best_particle_poses.append((best_particle.x, best_particle.y, best_particle.theta))
        ground_truth_poses.append(coordGT)
        
        
        # LIVE VISUALIZATION 
        

        if i % 1 == 0:
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
            
            ax4.clear()
            ax4.set_title('Theta (deg) latest')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Theta (deg)')
            ax4.plot(i, np.degrees(best_particle.theta), 'go')
            ax4.plot(i, coordGT[2], 'ro')
            ax4.set_xlim(0, 500)
            ax4.set_ylim(-180, 180)
            
            plt.draw()
            plt.pause(0.01)
        
        i += 1

    plt.ioff()
    plt.show()


    # === FINAL STATIC PLOTS 


    if len(best_particle_poses) > 0:
        bp = np.array(best_particle_poses)
        gt = np.array(ground_truth_poses)

        fig2, (ax_xy, ax_theta) = plt.subplots(1, 2, figsize=(12, 5))

        ax_xy.plot(bp[:, 0], bp[:, 1], 'g-', label='Estimate (x,y)')
        ax_xy.plot(gt[:, 0], gt[:, 1], 'r--', label='Ground Truth (x,y)')
        ax_xy.set_title('Trajectory')
        ax_xy.set_xlabel('x')
        ax_xy.set_ylabel('y')
        ax_xy.set_xlim(0, 500)
        ax_xy.set_ylim(0, 500)
        ax_xy.set_aspect('equal')
        ax_xy.legend()
        ax_xy.grid(True)

        iters = np.arange(bp.shape[0])
        ax_theta.plot(iters, np.degrees(bp[:, 2]), 'g-', label='Estimate theta (deg)')
        ax_theta.plot(iters, gt[:, 2], 'r--', label='GT theta (deg)')
        ax_theta.set_title('Theta over time')
        ax_theta.set_xlabel('Iteration')
        ax_theta.set_ylabel('Theta (deg)')
        ax_theta.set_xlim(0, max(50, bp.shape[0]))
        ax_theta.set_ylim(-180, 180)
        ax_theta.legend()
        ax_theta.grid(True)

        plt.tight_layout()
        plt.show()
