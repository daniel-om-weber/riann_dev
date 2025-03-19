"""
Setup functions for RCMG simulations.

This module provides configurable setup functions for use with ring.RCMG to create
diverse IMU datasets by varying physical parameters like gravity, mass, and dynamics.
"""

import jax
import jax.numpy as jnp


def create_setup_fn(
    # Boolean switches to enable/disable randomization types
    randomize_gravity=True,
    randomize_dynamics=True,
    randomize_mass=True,
    
    # Gravity parameters
    gravity_mag_range=(9.0, 10.0),  # Default near Earth gravity
    
    # Dynamics parameters
    damping_range=(0.5, 3.0),
    stiffness_range=(0.2, 5.0),
    armature_range=(0.8, 2.5),
    
    # Mass parameters
    mass_range=(0.7, 2.5)
):
    """
    Creates a customized setup_fn for RCMG with specified parameter ranges.
    
    Parameters:
        randomize_gravity: Boolean to enable/disable gravity randomization
        randomize_dynamics: Boolean to enable/disable dynamics randomization
        randomize_mass: Boolean to enable/disable mass randomization
        gravity_mag_range: (min, max) for gravity magnitude in m/s²
        damping_range: (min, max) scaling factor for joint damping
        stiffness_range: (min, max) scaling factor for joint spring stiffness
        armature_range: (min, max) scaling factor for joint armature
        mass_range: (min, max) scaling factor for geometry masses
    
    Returns:
        setup_fn: Function with signature (key, sys) -> sys for use with RCMG
    """
    def setup_fn(key, sys):
        # Split the key for different random parameters
        keys = jax.random.split(key, 4)
        key_idx = 0
        
        # Apply requested randomizations
        if randomize_gravity:
            # Fixed downward direction, random magnitude
            mag = jax.random.uniform(
                keys[key_idx], 
                minval=gravity_mag_range[0], 
                maxval=gravity_mag_range[1]
            )
            key_idx += 1
            dir = jnp.array([0, 0, -1.0])  # Always down
            new_gravity = mag * dir
            sys = sys.replace(gravity=new_gravity)
        
        if randomize_dynamics:
            damp_scale = jax.random.uniform(
                keys[key_idx], 
                minval=damping_range[0], 
                maxval=damping_range[1]
            )
            key_idx += 1
            
            stiff_scale = jax.random.uniform(
                keys[key_idx], 
                minval=stiffness_range[0], 
                maxval=stiffness_range[1]
            )
            key_idx += 1
            
            arma_scale = jax.random.uniform(
                keys[key_idx], 
                minval=armature_range[0], 
                maxval=armature_range[1]
            )
            
            sys = sys.replace(
                link_damping=sys.link_damping * damp_scale,
                link_spring_stiffness=sys.link_spring_stiffness * stiff_scale,
                link_armature=sys.link_armature * arma_scale
            )
        
        if randomize_mass:
            new_geoms = []
            for geom in sys.geoms:
                # Create a reproducible subkey for each geometry
                geom_key = jax.random.fold_in(keys[0], hash(str(geom)))
                
                mass_multiplier = jax.random.uniform(
                    geom_key, 
                    minval=mass_range[0], 
                    maxval=mass_range[1]
                )
                new_geoms.append(geom.replace(mass=geom.mass * mass_multiplier))
            
            sys = sys.replace(geoms=new_geoms)
        
        return sys
    
    return setup_fn