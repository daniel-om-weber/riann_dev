# %%
import numpy as np
import matplotlib.pyplot as plt
import ring
from ring.algorithms import MotionConfig

# Step 1: Define our system with a single body and IMU
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

# Create the system from XML
sys = ring.System.from_str(system_xml)

# Step 2: Configure motion parameters
# We'll create a diverse set of motions by combining different configs
config1 = MotionConfig(
    T=10.0,                    # 10 seconds of motion
    t_min=0.1,                 # Min time between angle changes
    t_max=0.5,                 # Max time between angle changes
    dang_min=0.2,              # Min angular velocity (rad/s)
    dang_max=2.0,              # Max angular velocity (rad/s)
    dang_min_free_spherical=0.1,  # For free/spherical joints
    dang_max_free_spherical=1.5,  # For free/spherical joints
    dpos_min=0.01,             # Min positional velocity (m/s)
    dpos_max=0.5,              # Max positional velocity (m/s)
    randomized_interpolation_angle=True,  # Use random interpolation for angles
    randomized_interpolation_position=True,  # Use random interpolation for positions
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

# Join configs to create more diverse motions
configs = [config1, config2]

# %%
# PARALLELIZATION IMPROVEMENT 1: Set XLA thread count
from ring.utils.backend import set_host_device_count
import jax
print(f"Using {jax.default_backend()} backend with {jax.device_count()} devices")

# 1. Configure JAX to use multiple CPU cores (crucial for performance)
if jax.default_backend() == "cpu":
    num_cpus = max(1, os.cpu_count() - 1)  # Leave one core free
    print(f"Setting JAX to use {num_cpus} CPU cores as separate devices")
    set_host_device_count(num_cpus)
else:
    print(f"Using {jax.default_backend()} backend with {jax.device_count()} devices")

# Step 3: Set up RCMG and generate motion sequences
rcmg = ring.RCMG(
    sys,
    configs,
    add_X_imus=True,          # Generate IMU data
    add_X_imus_kwargs={       # Configuration for IMU data generation
        "noisy": True,        # Add realistic noise to IMU measurements
        "has_magnetometer": True,  # Generate magnetometer data (for 9D IMU)
        "low_pass_filter_pos_f_cutoff": 15.0,  # Low-pass filter cutoff for position
        "low_pass_filter_rot_cutoff": 12.0,    # Low-pass filter cutoff for rotation
    },
    add_y_relpose=True,       # Get relative orientations
    add_y_rootfull=True,      # Include full root state in ground truth
    randomize_positions=True,  # Randomize initial positions
    imu_motion_artifacts=False,  # No IMU motion artifacts for simplicity
    dynamic_simulation=True,   # Use dynamic simulation for realism
    disable_tqdm=True,        # Show progress bars
)

# Generate a batch of sequences
batch_size = 10  # Number of sequences to generate
data = rcmg.to_list(sizes=batch_size, seed=42)

# %%
data[0][1]["object"]

# %%
# Step 4: Process and analyze the data
def process_sequence(seq_data):
    X, y = seq_data
    
    # Extract IMU measurements
    imu_data = X["object"]
    acc = imu_data["acc"]  # Accelerometer data
    gyr = imu_data["gyr"]  # Gyroscope data
    mag = imu_data["mag"]  # Magnetometer data
    ground_truth = y.get("object", None)  # ground truth orientation

    return {
        "accelerometer": acc,
        "gyroscope": gyr,
        "magnetometer": mag,
        "orientation": ground_truth,
        "dt": X.get("dt", 0.01)  # Get timestep if available
    }

# Process the first sequence as an example
processed_data = process_sequence(data[0])
processed_data

# %%

import h5py
# Save the generated data
def save_data(seq_data, filename):
    """
    Save raw RCMG data to HDF5 format with each axis stored as a separate sequence.
    All data is stored as float32 ('f4') type.
    
    Parameters:
    seq_data (tuple): Tuple of (X, y) containing raw RCMG data
    filename (str): Output filename (without extension)
    """
    
    X, y = seq_data
    
    # Extract data directly from raw simulation output
    imu_data = X["object"]
    acc = imu_data["acc"]  # Accelerometer data
    gyr = imu_data["gyr"]  # Gyroscope data
    mag = imu_data["mag"]  # Magnetometer data
    orientation = y.get("object", None)  # Ground truth orientation
    dt = X.get("dt", 0.01)  # Get timestep if available
    
    # Create HDF5 file
    with h5py.File(filename + ".hdf5", 'w') as f:
        # Helper function to write dataset
        def write_dataset(group, ds_name, data, dtype='f4', chunks=None):
            group.create_dataset(ds_name, data=data, dtype=dtype, chunks=chunks)
            
        # Store each axis separately
        # Accelerometer
        for i, axis in enumerate(['x', 'y', 'z']):
            write_dataset(f, f'acc_{axis}', acc[:, i])
        
        # Gyroscope
        for i, axis in enumerate(['x', 'y', 'z']):
            write_dataset(f, f'gyr_{axis}', gyr[:, i])
        
        # Magnetometer
        for i, axis in enumerate(['x', 'y', 'z']):
            write_dataset(f, f'mag_{axis}', mag[:, i])
        
        # Orientation (quaternion - w, x, y, z)
        if orientation is not None:
            for i, component in enumerate(['w', 'x', 'y', 'z']):
                write_dataset(f, f'orientation_{component}', orientation[:, i])
        
        # Create dt as a sequence of the same length
        dt_array = np.ones(len(acc), dtype=np.float32) * dt
        write_dataset(f, 'dt', dt_array)

# Save all sequences
for i, seq in enumerate(data):
    save_data(seq, f"imu_sequence_{i}")
    print(f"Saved sequence {i}")

# %%
# Visualize the IMU data
def plot_imu_data(processed_data):
    fig, axs = plt.subplots(4, 1, figsize=(12, 10))
    
    # Plot accelerometer data
    axs[0].plot(processed_data["accelerometer"])
    axs[0].set_title("Accelerometer Data")
    axs[0].set_ylabel("Acceleration (m/s²)")
    axs[0].legend(["X", "Y", "Z"])
    
    # Plot gyroscope data
    axs[1].plot(processed_data["gyroscope"])
    axs[1].set_title("Gyroscope Data")
    axs[1].set_ylabel("Angular Velocity (rad/s)")
    axs[1].legend(["X", "Y", "Z"])
    
    # Plot magnetometer data
    axs[2].plot(processed_data["magnetometer"])
    axs[2].set_title("Magnetometer Data")
    axs[2].set_ylabel("Magnetic Field (a.u.)")
    axs[2].legend(["X", "Y", "Z"])
    
    # Plot orientation (quaternion)
    axs[3].plot(processed_data["orientation"])
    axs[3].set_title("Ground Truth Orientation (Quaternion)")
    axs[3].set_xlabel("Time Step")
    axs[3].legend(["w", "x", "y", "z"])
    
    plt.tight_layout()
    return fig


#Visualization example (uncomment to use)
fig = plot_imu_data(processed_data)
fig.savefig("imu_data_visualization.png")
plt.close(fig)


print(f"Generated {batch_size} sequences of IMU data with ground truth orientation.")
print("Example data shape:")
print(f"  Accelerometer: {processed_data['accelerometer'].shape}")
print(f"  Gyroscope: {processed_data['gyroscope'].shape}")
print(f"  Ground truth orientation: {processed_data['orientation'].shape}")

# %%
