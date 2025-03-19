import os
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"

import time
import json
import h5py
import multiprocessing as mp
import numpy as np
import ring
from ring.utils.path import parse_path
import warnings
from system_configs import create_system_configs
from motion_configs import create_motion_configs
warnings.filterwarnings("ignore", message="The path .* already has an extension .*, but it gets replaced by the extension=.*")

# 1. Define various system configurations - moved to system_configs.py

# 2. Define motion configurations


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
    mp.set_start_method('spawn')
    # Configuration
    output_directory = "diverse_imu_data"
    sequences_per_config = 20  # Adjust based on your needs
    num_processes = 8  # Adjust based on your machine's CPU count
    
    # Generate data
    generate_diverse_imu_data(
        output_directory,
        sequences_per_config=sequences_per_config,
        n_processes=num_processes
    )