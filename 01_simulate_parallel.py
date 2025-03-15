import os
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"

import time
import numpy as np
import h5py
import ring
import multiprocessing as mp
from ring.algorithms import MotionConfig
from ring.utils.path import parse_path

import warnings
warnings.filterwarnings("ignore", message="The path .* already has an extension .*, but it gets replaced by the extension=.*")


# 2. Define the custom save function for HDF5 output
def save_sequence_to_file(seq_data, path):
    """Save a sequence directly to HDF5 file instead of pickle"""
    # Convert path from .pickle to .h5

    h5_path = parse_path(path, extension="hdf5", file_exists_ok=True)
    # Unpack data
    X, y = seq_data
    
    try:
        with h5py.File(h5_path, 'w') as f:
            # Extract IMU data
            imu_data = X["object"]
            acc = imu_data["acc"]
            gyr = imu_data["gyr"]
            mag = imu_data["mag"]
            orientation = y.get("object", None)
            dt = X.get("dt", 0.01)
            
            # Store IMU data
            for i, axis in enumerate(['x', 'y', 'z']):
                f.create_dataset(f'acc_{axis}', data=acc[:, i], dtype='f4',
                               compression="gzip", compression_opts=4)
                f.create_dataset(f'gyr_{axis}', data=gyr[:, i], dtype='f4',
                               compression="gzip", compression_opts=4)
                f.create_dataset(f'mag_{axis}', data=mag[:, i], dtype='f4',
                               compression="gzip", compression_opts=4)
            
            # Store orientation
            if orientation is not None:
                for i, component in enumerate(['w', 'x', 'y', 'z']):
                    f.create_dataset(f'orientation_{component}', data=orientation[:, i], 
                                   dtype='f4', compression="gzip", compression_opts=4)
            
            # Store timestep info
            f.create_dataset('dt', data=np.ones(len(acc), dtype=np.float32) * dt, dtype='f4')
            
        return True
    except Exception as e:
        print(f"Error saving to {h5_path}: {e}")
        return False

# Worker function for each process
def worker_generate_sequences(system_xml, configs, start_idx, num_sequences, output_dir, seed, base_filename="imu_sequence_"):
    """
    Worker function to generate a subset of sequences
    
    Args:
        system_xml: System XML string
        configs: List of MotionConfig objects
        start_idx: Start index for this worker's sequence numbering
        num_sequences: Number of sequences to generate in this worker
        output_dir: Directory to save HDF5 files
        seed: Random seed (will be adjusted per worker)
        base_filename: Base name for output files
    """
    # Set a unique seed for this worker to ensure different sequences
    worker_seed = seed + start_idx
    
    # Create system
    sys = ring.System.from_str(system_xml)
    
    # Create RCMG instance
    rcmg = ring.RCMG(
        sys,
        configs,
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
        disable_tqdm=True,  # Disable tqdm progress bar to reduce terminal clutter
    )
    
    # Custom function to rename files to make them sequential
    def custom_save_wrapper(seq_data, path):
        # Extract original index from path
        original_path = path
        filename = os.path.basename(original_path)
        
        # Create new path with global sequential index
        seq_idx = int(filename.split('_')[-1].split('.')[0])  # Extract sequence number
        new_filename = f"{base_filename}{start_idx + seq_idx}.pickle"
        new_path = os.path.join(os.path.dirname(original_path), new_filename)
        
        # Save using the original function
        return save_sequence_to_file(seq_data, new_path)
    
    # Generate and save sequences for this worker
    pid = mp.current_process().pid
    
    # Generate sequences silently
    rcmg.to_folder(
        path=output_dir,
        sizes=num_sequences,
        seed=worker_seed,
        overwrite=True,
        file_prefix="temp_",  # Temporary prefix, will be renamed by custom save wrapper
        save_fn=custom_save_wrapper,
        verbose=False,  # Disable verbose output
    )
    
    return num_sequences

# Main function to generate sequences using multiprocessing
def generate_sequences(system_xml, configs, total_sequences, output_dir, n_processes=None, seed=42):
    """
    Generate sequences using RCMG in parallel and save them directly using to_folder
    
    Args:
        system_xml: System XML string
        configs: List of MotionConfig objects
        total_sequences: Number of sequences to generate
        output_dir: Directory to save HDF5 files
        n_processes: Number of processes to use (defaults to CPU count)
        seed: Random seed
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine number of processes
    if n_processes is None:
        n_processes = mp.cpu_count()
    n_processes = min(n_processes, total_sequences)
    
    print(f"Generating {total_sequences} sequences using {n_processes} processes")
    print(f"Output directory: {output_dir}")
    
    # Start timing
    start_time = time.time()
    
    # Calculate sequences per process
    base_size = total_sequences // n_processes
    remainder = total_sequences % n_processes
    
    # Create process pool
    pool = mp.Pool(processes=n_processes)
    
    # Prepare arguments for each worker
    worker_args = []
    start_idx = 0
    
    for i in range(n_processes):
        # Distribute remainder among first processes
        chunk_size = base_size + (1 if i < remainder else 0)
        
        # Add arguments for this worker
        worker_args.append((
            system_xml,
            configs,
            start_idx,
            chunk_size,
            output_dir,
            seed
        ))
        
        start_idx += chunk_size
    
    # Create a progress counter
    completed = 0
    print(f"[{completed}/{total_sequences}] Starting simulations...")
    
    # Start workers
    results = pool.starmap(worker_generate_sequences, worker_args)
    
    # Close and join the pool
    pool.close()
    pool.join()
    
    # Count total sequences generated
    total_generated = sum(results)
    
    # Simulation complete
    total_time = time.time() - start_time
    print(f"Complete! {total_generated} sequences processed in {total_time:.2f}s ({total_generated/total_time:.2f} seq/s)")

# 4. Your configurations
system_xml = """
<x_xy model="object_with_imu">
    <options gravity="0 0 9.81" dt="0.01"/>
    <defaults>
        <geom edge_color="black" color="white"/>
    </defaults>
    <worldbody>
        <body name="object" joint="free" damping="2 2 2 10 10 10">
            <geom type="box" mass="1" pos="0 0 0" dim="0.1 0.05 0.02"/>
            <body name="imu1" joint="frozen" pos="0 0 0.02">
                <geom type="box" mass="0.01" dim="0.02 0.02 0.01" color="orange"/>
            </body>
        </body>
    </worldbody>
</x_xy>
"""

config1 = MotionConfig(
    T=10.0,
    t_min=0.1,
    t_max=0.5,
    dang_min=0.2,
    dang_max=2.0,
    dang_min_free_spherical=0.1,
    dang_max_free_spherical=1.5,
    dpos_min=0.01,
    dpos_max=0.5,
    randomized_interpolation_angle=True,
    randomized_interpolation_position=True,
)

config2 = MotionConfig(
    T=10.0,
    t_min=0.2,
    t_max=0.8,
    dang_min=0.1,
    dang_max=1.0,
    dang_min_free_spherical=0.05,
    dang_max_free_spherical=0.8,
    dpos_min=0.02,
    dpos_max=0.3,
    randomized_interpolation_angle=True,
    randomized_interpolation_position=True,
)

configs = [config1, config2]

# 5. Execute the generator
if __name__ == "__main__":
    # Set parameters
    output_dir = "imu_data"
    n_processes = 8  # Set to None to use all available CPU cores
    total_sequences = n_processes*1024
    
    # Run the parallel generator
    generate_sequences(
        system_xml, 
        configs, 
        total_sequences, 
        output_dir,
        n_processes
    )