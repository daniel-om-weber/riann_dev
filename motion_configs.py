import numpy as np
from ring.algorithms import MotionConfig

def create_motion_configs():
    """Create a list of diverse motion configurations"""
    configs = []
    
    # Define different motion speeds
    speeds = [
        {'name': 'very_slow', 't_min': 0.8, 't_max': 2.0, 'dang_min': 0.01, 'dang_max': 0.2, 'dpos_min': 0.005, 'dpos_max': 0.1},
        {'name': 'slow', 't_min': 0.4, 't_max': 1.0, 'dang_min': 0.05, 'dang_max': 0.5, 'dpos_min': 0.01, 'dpos_max': 0.2},
        {'name': 'medium', 't_min': 0.2, 't_max': 0.6, 'dang_min': 0.1, 'dang_max': 1.5, 'dpos_min': 0.05, 'dpos_max': 0.4},
        {'name': 'fast', 't_min': 0.1, 't_max': 0.4, 'dang_min': 0.3, 'dang_max': 3.0, 'dpos_min': 0.1, 'dpos_max': 0.6},
        {'name': 'very_fast', 't_min': 0.05, 't_max': 0.2, 'dang_min': 0.5, 'dang_max': 5.0, 'dpos_min': 0.2, 'dpos_max': 1.0}
    ]
    
    # Define different motion styles
    styles = [
        {'name': 'smooth', 'randomized_interpolation_angle': False, 'randomized_interpolation_position': False},
        {'name': 'random', 'randomized_interpolation_angle': True, 'randomized_interpolation_position': True, 'cdf_bins_min': 3, 'cdf_bins_max': 8}
    ]
    
    # Combine parameters to create configurations
    for speed in speeds:
        for style in styles:
            # Extract names
            speed_name = speed.pop('name')
            style_name = style.pop('name')
            
            # Copy dictionaries to avoid modifying the originals
            speed_params = speed.copy()
            style_params = style.copy()
            
            # Combine configuration parameters
            config_params = {**speed_params, **style_params}
            config_name = f"{speed_name}_{style_name}"
            
            # Create motion config
            configs.append({
                'name': config_name,
                'config': MotionConfig(T=60.0, **config_params)
            })
            
            # Restore names for next iteration
            speed['name'] = speed_name
            style['name'] = style_name
    
    return configs