import os
# # Set JAX compilation cache directory
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
# # Enable persistent compilation cache
# os.environ['JAX_DISABLE_JIT_CACHE_WARMING'] = '1'
# os.environ['JAX_ENABLE_X64'] = '1'
# os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # Or 'cpu' if not using GPU

# # Set cache directory (create this directory first)
# cache_dir = os.path.expanduser("~/.jax_cache")
# os.makedirs(cache_dir, exist_ok=True)
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# os.environ['JAX_CACHE_DIR'] = cache_dir

import time
import numpy as np
import h5py
import ring
from ring.algorithms import MotionConfig
from ring.utils.path import parse_path

import warnings
warnings.filterwarnings("ignore", message="The path .* already has an extension .*, but it gets replaced by the extension=.*")


def save_sequence_to_file(seq_data, path):
    """Save sequence data to HDF5 file"""
    h5_path = parse_path(path, extension="hdf5", file_exists_ok=True)
    X, y = seq_data
    
    try:
        with h5py.File(h5_path, 'w') as f:
            # Extract and store IMU data
            imu_data = X["object"]
            for data_type, values in {
                "acc": imu_data["acc"],
                "gyr": imu_data["gyr"],
                "mag": imu_data["mag"]
            }.items():
                for i, axis in enumerate(['x', 'y', 'z']):
                    f.create_dataset(f'{data_type}_{axis}', data=values[:, i], 
                                   dtype='f4', compression="gzip", compression_opts=4)
            
            # Store orientation if available
            orientation = y.get("object", None)
            if orientation is not None:
                for i, component in enumerate(['w', 'x', 'y', 'z']):
                    f.create_dataset(f'orientation_{component}', data=orientation[:, i], 
                                   dtype='f4', compression="gzip", compression_opts=4)
            
            # Store timestep info
            dt = X.get("dt", 0.01)
            f.create_dataset('dt', data=np.ones(len(imu_data["acc"]), dtype=np.float32) * dt, dtype='f4')
        return True
    except Exception as e:
        print(f"Error saving to {h5_path}: {e}")
        return False


def generate_sequences(system_xml, configs, total_sequences, output_dir, seed=42, base_filename="imu_sequence_"):
    """Generate IMU sequences and save as HDF5 files"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {total_sequences} sequences in {output_dir}")
    start_time = time.time()
    
    # Initialize system and RCMG
    sys = ring.System.from_str(system_xml)
    rcmg = ring.RCMG(
        sys, configs,
        add_X_imus=True,
        add_X_imus_kwargs={
            "noisy": True, "has_magnetometer": True,
            "low_pass_filter_pos_f_cutoff": 15.0, "low_pass_filter_rot_cutoff": 12.0,
        },
        add_y_relpose=True, add_y_rootfull=True,
        randomize_positions=True, imu_motion_artifacts=False,
        dynamic_simulation=True
    )
    
    # Generate and save sequences
    def custom_save(seq_data, path):
        # Extract sequence index from path, handle "seq" prefix if present
        basename = os.path.basename(path)
        if 'seq' in basename:
            # Handle case where it's in format 'seq0'
            seq_idx = int(basename.split('seq')[1].split('.')[0])
        else:
            # Handle original expected format
            seq_idx = int(basename.split('_')[-1].split('.')[0])
        
        new_path = os.path.join(output_dir, f"{base_filename}{seq_idx}.hdf5")
        return save_sequence_to_file(seq_data, new_path)
    
    rcmg.to_folder(path=output_dir, sizes=total_sequences, seed=seed, 
                  overwrite=True, save_fn=custom_save)
    
    # Report completion
    elapsed = time.time() - start_time
    print(f"Complete! {total_sequences} sequences in {elapsed:.2f}s ({total_sequences/elapsed:.2f} seq/s)")


# System configuration
system_xml = """
<x_xy model="object_with_imu">
    <options gravity="0 0 9.81" dt="0.01"/>
    <defaults><geom edge_color="black" color="white"/></defaults>
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

# Motion configurations
configs = [
    MotionConfig(
        T=10.0, t_min=0.1, t_max=0.5, 
        dang_min=0.2, dang_max=2.0,
        dang_min_free_spherical=0.1, dang_max_free_spherical=1.5,
        dpos_min=0.01, dpos_max=0.5,
        randomized_interpolation_angle=True, 
        randomized_interpolation_position=True
    ),
    MotionConfig(
        T=10.0, t_min=0.2, t_max=0.8, 
        dang_min=0.1, dang_max=1.0,
        dang_min_free_spherical=0.05, dang_max_free_spherical=0.8,
        dpos_min=0.02, dpos_max=0.3,
        randomized_interpolation_angle=True, 
        randomized_interpolation_position=True
    )
]

# Run if executed as script
if __name__ == "__main__":
    generate_sequences(
        system_xml, 
        configs, 
        total_sequences=8*1024, 
        output_dir="imu_data"
    )