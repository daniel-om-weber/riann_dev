"""
System configuration generation for RCMG simulations.

This module provides functions to generate XML configurations for different
physical systems to be used in ring.RCMG simulations, including boxes, spheres,
cylinders, and articulated systems.
"""

def generate_system_xml(system_type, name=None):
    """
    Generate XML configuration for a physical system.
    
    Parameters:
        system_type: Type of system ('box', 'sphere', 'cylinder', or 'articulated')
        name: Optional name for the system (defaults to system_type)
    
    Returns:
        XML string defining the system
    """
    name = name or system_type
    
    # System-specific parameters
    if system_type == 'box':
        params = {
            'geom_type': 'box',
            'dimensions': '0.1 0.05 0.02',
            'imu_pos': '0 0 0.02'
        }
    elif system_type == 'sphere':
        params = {
            'geom_type': 'sphere',
            'dimensions': '0.05',
            'imu_pos': '0 0 0.05'
        }
    elif system_type == 'cylinder':
        params = {
            'geom_type': 'cylinder',
            'dimensions': '0.03 0.15',
            'imu_pos': '0 0 0.03'
        }
    else:
        return generate_articulated_system_xml()
    
    return f"""
    <x_xy model="{name}_with_imu">
        <options gravity="0 0 9.81" dt="0.01"/>
        <defaults>
            <geom edge_color="black" color="white"/>
        </defaults>
        <worldbody>
            <body name="object" joint="free" damping="2.0 2.0 2.0 10.0 10.0 10.0">
                <geom type="{params['geom_type']}" mass="1.0" pos="0 0 0" dim="{params['dimensions']}"/>
                <body name="imu1" joint="frozen" pos="{params['imu_pos']}">
                    <geom type="box" mass="0.01" dim="0.02 0.02 0.01" color="orange"/>
                </body>
            </body>
        </worldbody>
    </x_xy>
    """


def generate_articulated_system_xml(num_segments=3):
    """
    Generate XML configuration for an articulated system.
    
    Parameters:
        num_segments: Number of linked segments in the system
    
    Returns:
        XML string defining the articulated system
    """
    # Create segment XML recursively
    def create_segment(index, is_root=False):
        if index >= num_segments:
            return ""
        
        joint_type = "free" if is_root else "ry"
        damping_str = 'damping="2.0 2.0 2.0 3.0 3.0 3.0"' if is_root else 'damping="3.0"'
        
        imu_element = '<body name="imu1" joint="frozen" pos="0.05 0 0.025"><geom type="box" mass="0.01" dim="0.02 0.02 0.01" color="orange"/></body>' if index == 0 else ""
        next_segment = create_segment(index + 1) if index < num_segments - 1 else ""
        
        return f"""
        <body name="segment{index}" joint="{joint_type}" {damping_str}>
            <geom type="box" mass="0.5" pos="0.05 0 0" dim="0.1 0.025 0.025"/>
            {imu_element}
            {next_segment}
        </body>
        """
    
    return f"""
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


def create_system_configs():
    """
    Create a list of system configurations.
    
    Returns:
        List of dictionaries containing system configuration names and XML definitions
    """
    return [
        {'name': 'box', 'xml': generate_system_xml('box')},
        {'name': 'sphere', 'xml': generate_system_xml('sphere')},
        {'name': 'cylinder', 'xml': generate_system_xml('cylinder')},
        {'name': 'articulated', 'xml': generate_articulated_system_xml()}
    ] 