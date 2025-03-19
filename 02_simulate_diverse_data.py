import os
import time
import json
import h5py
import multiprocessing as mp
import numpy as np
import ring
from ring.algorithms import MotionConfig
from ring.utils.path import parse_path
from functools import partial
import warnings

warnings.filterwarnings("ignore", message="The path .* already has an extension .*, but it gets replaced by the extension=.*")

# 1. Define various system configurations
def generate_box_system_xml(mass, size_scale, damping_scale=1.0):
    # Calculate dimensions based on scale
    dim_x = 0.1 * size_scale
    dim_y = 0.05 * size_scale
    dim_z = 0.02 * size_scale
    
    # Calculate damping values
    pos_damping = 2.0 * damping_scale
    rot_damping = 10.0 * damping_scale
    
    return f"""
    <x_xy model="box_with_imu">
        <options gravity="0 0 9.81" dt="0.01"/>
        <defaults>
            <geom edge_color="black" color="white"/>
        </defaults>
        <worldbody>
            <body name="object" joint="free" damping="{pos_damping} {pos_damping} {pos_damping} {rot_damping} {rot_damping} {rot_damping}">
                <geom type="box" mass="{mass}" pos="0 0 0" dim="{dim_x} {dim_y} {dim_z}"/>
                <body name="imu1" joint="frozen" pos="0 0 {dim_z}">
                    <geom type="box" mass="0.01" dim="0.02 0.02 0.01" color="orange"/>
                </body>
            </body>
        </worldbody>
    </x_xy>
    """

def generate_sphere_system_xml(mass, size_scale, damping_scale=1.0):
    # Calculate radius based on scale
    radius = 0.05 * size_scale
    
    # Calculate damping values
    pos_damping = 2.0 * damping_scale
    rot_damping = 10.0 * damping_scale
    
    return f"""
    <x_xy model="sphere_with_imu">
        <options gravity="0 0 9.81" dt="0.01"/>
        <defaults>
            <geom edge_color="black" color="white"/>
        </defaults>
        <worldbody>
            <body name="object" joint="free" damping="{pos_damping} {pos_damping} {pos_damping} {rot_damping} {rot_damping} {rot_damping}">
                <geom type="sphere" mass="{mass}" pos="0 0 0" dim="{radius}"/>
                <body name="imu1" joint="frozen" pos="0 0 {radius}">
                    <geom type="box" mass="0.01" dim="0.02 0.02 0.01" color="orange"/>
                </body>
            </body>
        </worldbody>
    </x_xy>
    """

def generate_cylinder_system_xml(mass, size_scale, damping_scale=1.0):
    # Calculate dimensions based on scale
    radius = 0.03 * size_scale
    length = 0.15 * size_scale
    
    # Calculate damping values
    pos_damping = 2.0 * damping_scale
    rot_damping = 10.0 * damping_scale
    
    return f"""
    <x_xy model="cylinder_with_imu">
        <options gravity="0 0 9.81" dt="0.01"/>
        <defaults>
            <geom edge_color="black" color="white"/>
        </defaults>
        <worldbody>
            <body name="object" joint="free" damping="{pos_damping} {pos_damping} {pos_damping} {rot_damping} {rot_damping} {rot_damping}">
                <geom type="cylinder" mass="{mass}" pos="0 0 0" dim="{radius} {length}"/>
                <body name="imu1" joint="frozen" pos="0 0 {radius}">
                    <geom type="box" mass="0.01" dim="0.02 0.02 0.01" color="orange"/>
                </body>
            </body>
        </worldbody>
    </x_xy>
    """

def generate_articulated_system_xml(num_segments, mass_per_segment=0.5, damping_scale=1.0):
    """Generate an articulated system with the specified number of segments"""
    
    # Base damping values
    pos_damping = 2.0 * damping_scale
    rot_damping = 3.0 * damping_scale
    
    # XML strings for segments
    segment_strings = []
    
    # Create segment XML recursively
    def create_segment(index, is_root=False):
        if index >= num_segments:
            return ""
        
        joint_type = "free" if is_root else "ry"
        damping_str = f'damping="{pos_damping} {pos_damping} {pos_damping} {rot_damping} {rot_damping} {rot_damping}"' if is_root else f'damping="{rot_damping}"'
        
        imu_element = '<body name="imu1" joint="frozen" pos="0.05 0 0.025"><geom type="box" mass="0.01" dim="0.02 0.02 0.01" color="orange"/></body>' if index == 0 else ""
        next_segment = create_segment(index + 1) if index < num_segments - 1 else ""
        
        segment = f"""
        <body name="segment{index}" joint="{joint_type}" {damping_str}>
            <geom type="box" mass="{mass_per_segment}" pos="0.05 0 0" dim="0.1 0.025 0.025"/>
            {imu_element}
            {next_segment}
        </body>
        """
        return segment
    
    # Create the full XML
    xml = f"""
    <x_xy model="articulated_{num_segments}_segments">
        <options gravity="0 0 9.81" dt="0.01"/>
        <defaults>
            <geom edge_color="black" color="white"/>
        </defaults>
        <worldbody>
            {create_segment(0, True)}
        </worldbody>
    </x_xy>
    """
    
    return xml

def create_system_configs():
    """Create a list of system configurations"""
    configs = []
    
    # Add box configurations (different masses and sizes)
    for mass in [0.2, 1.0, 5.0]:
        for size_scale in [0.8, 1.0, 1.5]:
            for damping in [0.5, 1.0, 2.0]:
                configs.append({
                    'name': f"box_m{mass}_s{size_scale}_d{damping}",
                    'xml': generate_box_system_xml(mass, size_scale, damping)
                })
    
    # Add sphere configurations
    for mass in [0.2, 1.0, 5.0]:
        for size_scale in [0.8, 1.0, 1.5]:
            configs.append({
                'name': f"sphere_m{mass}_s{size_scale}",
                'xml': generate_sphere_system_xml(mass, size_scale)
            })
    
    # Add cylinder configurations
    for mass in [0.2, 1.0, 5.0]:
        for size_scale in [0.8, 1.0, 1.5]:
            configs.append({
                'name': f"cylinder_m{mass}_s{size_scale}",
                'xml': generate_cylinder_system_xml(mass, size_scale)
            })
    
    # Add articulated configurations
    for segments in [2, 3, 4]:
        for mass in [0.2, 0.5, 1.0]:
            configs.append({
                'name': f"articulated_seg{segments}_m{mass}",
                'xml': generate_articulated_system_xml(segments, mass)
            })
    
    return configs

# 2. Define motion configurations
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
    
    # Define different motion constraints
    constraints = [
        {'name': 'constrained', 'delta_ang_min': 0.2, 'delta_ang_max': 0.8},
        {'name': 'free', 'delta_ang_min': 0.0, 'delta_ang_max': 2.0 * np.pi}
    ]
    
    # Combine parameters to create configurations
    for speed in speeds:
        for style in styles:
            for constraint in constraints:
                # Extract names
                speed_name = speed.pop('name')
                style_name = style.pop('name')
                constraint_name = constraint.pop('name')
                
                # Copy dictionaries to avoid modifying the originals
                speed_params = speed.copy()
                style_params = style.copy()
                constraint_params = constraint.copy()
                
                # Combine configuration parameters
                config_params = {**speed_params, **style_params, **constraint_params}
                config_name = f"{speed_name}_{style_name}_{constraint_name}"
                
                # Set consistent parameters for spherical joints
                config_params['dang_min_free_spherical'] = config_params['dang_min']
                config_params['dang_max_free_spherical'] = config_params['dang_max']
                config_params['delta_ang_min_free_spherical'] = config_params['delta_ang_min']
                config_params['delta_ang_max_free_spherical'] = config_params['delta_ang_max']
                
                # Create motion config
                configs.append({
                    'name': config_name,
                    'config': MotionConfig(T=60.0, **config_params)
                })
                
                # Restore names for next iteration
                speed['name'] = speed_name
                style['name'] = style_name
                constraint['name'] = constraint_name
    
    return configs

# 3. Save functions
def save_sequence_to_file(seq_data, path):
    """Save a sequence to HDF5 file"""
    h5_path = parse_path(path, extension="hdf5", file_exists_ok=True)
    X, y = seq_data
    
    try:
        with h5py.File(h5_path, 'w') as f:
            # Extract IMU data
            imu_data = X["object"]
            acc = imu_data["acc"] 
            gyr = imu_data["gyr"]
            mag = imu_data.get("mag", None)
            orientation = y.get("object", None)
            dt = X.get("dt", 0.01)
            
            # Store IMU data
            for i, axis in enumerate(['x', 'y', 'z']):
                f.create_dataset(f'acc_{axis}', data=acc[:, i], dtype='f4')
                f.create_dataset(f'gyr_{axis}', data=gyr[:, i], dtype='f4')
                
                if mag is not None:
                    f.create_dataset(f'mag_{axis}', data=mag[:, i], dtype='f4')
            
            # Store orientation if available
            if orientation is not None:
                for i, component in enumerate(['w', 'x', 'y', 'z']):
                    f.create_dataset(f'orientation_{component}', data=orientation[:, i], 
                                   dtype='f4')
            
            # Store timestep info
            f.create_dataset('dt', data=np.ones(len(acc), dtype=np.float32) * dt, dtype='f4')
            
        return True
    except Exception as e:
        print(f"Error saving to {h5_path}: {e}")
        return False

# 4. Generation function
def generate_for_configuration(system_config, motion_config, output_dir, num_sequences, seed=42):
    """Generate sequences for a specific system-motion configuration"""
    config_dir = os.path.join(output_dir, f"{system_config['name']}_{motion_config['name']}")
    os.makedirs(config_dir, exist_ok=True)
    
    # Initialize system
    sys = ring.System.from_str(system_config['xml'])
    
    # Create RCMG
    rcmg = ring.RCMG(
        sys,
        [motion_config['config']],
        add_X_imus=True,
        add_X_imus_kwargs={
            "noisy": True,
            "has_magnetometer": True,
            "low_pass_filter_pos_f_cutoff": 15.0,
            "low_pass_filter_rot_cutoff": 12.0,
        },
        add_y_relpose=True,
        add_y_rootfull=True,
        randomize_positions=True,
        imu_motion_artifacts=False,
        dynamic_simulation=True,
        disable_tqdm=True,
    )
    
    # Define custom save function
    def custom_save(seq_data, path):
        # Extract sequence index from path
        basename = os.path.basename(path)
        if 'seq' in basename:
            seq_idx = int(basename.split('seq')[1].split('.')[0])
        else:
            seq_idx = int(basename.split('_')[-1].split('.')[0])
        
        # Create new path in the configuration directory
        new_path = os.path.join(config_dir, f"sequence_{seq_idx}")
        
        # Save data
        result = save_sequence_to_file(seq_data, new_path)
        
        return result
    
    # Generate sequences
    start_time = time.time()
    print(f"Generating {num_sequences} sequences for {system_config['name']}_{motion_config['name']}...")
    
    rcmg.to_folder(
        path=config_dir,
        sizes=num_sequences,
        seed=seed,
        overwrite=True,
        save_fn=custom_save,
        verbose=False,
    )
    
    elapsed = time.time() - start_time
    print(f"Completed {system_config['name']}_{motion_config['name']} in {elapsed:.2f}s ({num_sequences/elapsed:.2f} seq/s)")
    
    return num_sequences

# 5. Worker function for parallel processing
def worker_function(args):
    """Worker function for parallel processing"""
    system_config, motion_config, output_dir, num_sequences, seed = args
    try:
        return generate_for_configuration(system_config, motion_config, output_dir, num_sequences, seed)
    except Exception as e:
        print(f"Error in worker for {system_config['name']}_{motion_config['name']}: {e}")
        return 0

# 6. Main generation function
def generate_diverse_imu_data(output_dir, sequences_per_config=10, n_processes=None, seed=42, debug=False):
    """Generate diverse IMU data with various system and motion configurations"""
    # Starting message
    print(f"Starting diverse IMU data generation...")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get system and motion configs
    system_configs = create_system_configs()
    motion_configs = create_motion_configs()
    
    # In debug mode, use only a small subset of configurations
    if debug:
        print("DEBUG MODE: Using minimal configuration set")
        system_configs = system_configs[:1]  # Just use the first system config
        motion_configs = motion_configs[:1]  # Just use the first motion config
        sequences_per_config = 1             # Generate only one sequence
    
    total_configs = len(system_configs) * len(motion_configs)
    total_sequences = total_configs * sequences_per_config
    
    print(f"Generated {len(system_configs)} system configurations and {len(motion_configs)} motion configurations")
    print(f"Total configurations: {total_configs}")
    print(f"Total sequences to generate: {total_sequences}")
    print(f"Generating {sequences_per_config} sequences per configuration")
    
    # Determine number of processes
    if n_processes is None:
        n_processes = mp.cpu_count()
    print(f"Using {n_processes} processes")
    
    # Prepare arguments for parallel processing
    args_list = []
    for i, system_config in enumerate(system_configs):
        for j, motion_config in enumerate(motion_configs):
            # Use different seeds for different configurations
            config_seed = seed + (i * len(motion_configs) + j) * 1000
            args_list.append((system_config, motion_config, output_dir, sequences_per_config, config_seed))
    
    # Start timing
    start_time = time.time()
    
    # Create process pool and run
    with mp.Pool(processes=n_processes) as pool:
        # Map worker function to argument list
        results = list(pool.map(worker_function, args_list))
    
    # Print completion stats
    total_sequences_generated = sum(results)
    total_time = time.time() - start_time
    print(f"\nGeneration complete!")
    print(f"Total configurations: {total_configs}")
    print(f"Total sequences generated: {total_sequences_generated}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average speed: {total_sequences_generated/total_time:.2f} sequences/second")

# Run if executed as script
if __name__ == "__main__":
    # Configuration
    output_directory = "diverse_imu_data"
    sequences_per_config = 20  # Adjust based on your needs
    num_processes = 8  # Adjust based on your machine's CPU count
    
    # Generate data
    generate_diverse_imu_data(
        output_directory,
        sequences_per_config=sequences_per_config,
        n_processes=num_processes,
        debug=False  # Set to True for debugging mode
    )