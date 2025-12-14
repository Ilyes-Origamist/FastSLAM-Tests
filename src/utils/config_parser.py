"""
Configuration file parser for FastSLAM parameters
"""

import yaml
import os
import numpy as np
from typing import Dict, Any, Optional


class Config:
    """Configuration container with dot notation access"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __repr__(self):
        return f"Config({self.__dict__})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config back to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object with dot notation access
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ['motion_strategy', 'noise', 'robot', 'particle_filter']
    for section in required_sections:
        if section not in config_dict:
            raise ValueError(f"Missing required configuration section: {section}")
    
    return Config(config_dict)


def get_robot_params(config: Config) -> Dict[str, Any]:
    """
    Extract robot simulator parameters from config
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary of robot parameters
    """
    return {
        'x': config.robot.initial_x,
        'y': config.robot.initial_y,
        'theta': config.robot.initial_theta,
        'sigmaDTheta': config.noise.rotation,
        'sigmaDx': config.noise.distance,
        'sigmaSensorAngle': config.noise.sensor_angle,
        'sensorNoiseRatio': config.noise.sensor_ratio,
        'sizeSensor': config.robot.sensor_size
    }


def get_particle_filter_params(config: Config) -> Dict[str, Any]:
    """
    Extract particle filter parameters from config
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary of particle filter parameters
    """
    pf = config.particle_filter
    
    return {
        'N': pf.num_particles,
        'x_initial': config.robot.initial_x,
        'y_initial': config.robot.initial_y,
        'theta_initial': np.radians(config.robot.initial_theta),
        'x_range': pf.x_range,
        'y_range': pf.y_range,
        'theta_range': pf.theta_range,
        'local_map_size': pf.local_map_size,
        'map_width': pf.map_width,
        'map_height': pf.map_height,
        'motion_noise': config.noise.distance * pf.motion_noise_scale,
        'turn_noise': np.radians(config.noise.rotation * pf.turn_noise_scale),
        'measurement_noise': config.noise.sensor_ratio * pf.measurement_noise_scale,
        'prior_prob': pf.prior_prob
    }


def print_config(config: Config, indent: int = 0):
    """
    Pretty print configuration
    
    Args:
        config: Configuration object
        indent: Indentation level
    """
    for key, value in config.__dict__.items():
        if isinstance(value, Config):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")
