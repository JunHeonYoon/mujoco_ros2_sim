import mujoco
import os
from ament_index_python.packages import get_package_share_directory
import time
import importlib

def load_mj_model(robot_name) -> mujoco.MjModel:
    """
    Loads a MuJoCo model from an XML file based on the provided robot name.

    This function constructs a path to the "mujoco_menagerie" directory within the package's share directory,
    checks if the specified robot exists, and then loads the corresponding MuJoCo model using its XML description.

    Parameters:
        robot_name (str): The name of the robot to load. This should match one of the subdirectories 
                          in the "mujoco_menagerie" directory.

    Returns:
        mujoco.MjModel: The MuJoCo model object loaded from the XML file corresponding to the given robot name.

    Raises:
        AssertionError: If the specified robot_name is not found in the available robot directories.
    """
    # Construct the path to the "mujoco_menagerie" directory in the package's share directory.
    mujoco_menagerie_path = os.path.join(get_package_share_directory(__package__), "mujoco_menagerie")
    
    # List all available robot directories in the mujoco_menagerie folder.
    available_robots = [name for name in os.listdir(mujoco_menagerie_path)
                        if os.path.isdir(os.path.join(mujoco_menagerie_path, name))]
    
    # Assert that the provided robot_name is among the available robots.
    assert robot_name in available_robots, f"{robot_name} is not included in {available_robots}!"
    
    # Build the full path to the XML file that describes the MuJoCo scene for the robot.
    xml_file_path = get_package_share_directory(__package__) + f"/mujoco_menagerie/{robot_name}/scene.xml"
    
    # Load and return the MuJoCo model from the XML file.
    return mujoco.MjModel.from_xml_path(xml_file_path)

def precise_sleep(duration):
    """
    Sleeps for a specified duration using a busy-wait loop with a high-resolution timer.

    This function uses a while-loop along with a high-resolution performance counter (perf_counter)
    to pause execution for the given duration. This approach is intended for use cases where
    precise timing is required, such as in simulation loops.

    Parameters:
        duration (float): The amount of time in seconds to sleep.

    Returns:
        None
    """
    # Record the start time using a high-resolution performance counter.
    start = time.perf_counter()
    
    # Busy-wait loop until the elapsed time reaches or exceeds the specified duration.
    while True:
        now = time.perf_counter()
        if (now - start) >= duration:
            break


def load_class(full_class_string: str):
    """
    Dynamically loads a class from a full class path string.

    This function handles cases where the input is None or an empty string,
    and verifies that the string contains at least one dot ('.') to separate
    the module and class names. If the input is valid, it imports the module and
    retrieves the class.

    Parameters:
        full_class_string (str): A string representing the full path of the class,
                                 in the format "package.module.ClassName". For example,
                                 "my_package.my_module.MyClass".

    Returns:
        type or None: The class specified by the input string if found; otherwise,
                      None if the input is None or an empty string.

    Raises:
        ValueError: If the input string does not contain a dot ('.'), indicating
                    that it is not in the expected "module.ClassName" format.
    """
    # Check if the provided string is None or empty.
    if not full_class_string:
        # Return None if there is no valid class string.
        return None

    # Ensure that the string includes a dot to separate the module and class names.
    if '.' not in full_class_string:
        # Raise an error if the format is incorrect.
        raise ValueError(f"Invalid class string: '{full_class_string}'. "
                         "Must be in the form 'package.module.ClassName'.")

    # Split the full class string into the module name and the class name.
    # rsplit is used with maxsplit=1 to split only at the last occurrence of '.'
    module_name, class_name = full_class_string.rsplit('.', 1)
    
    # Dynamically import the module using the module name.
    mod = importlib.import_module(module_name)
    
    # Retrieve the class attribute from the module using the class name.
    cls = getattr(mod, class_name)
    
    # Return the loaded class.
    return cls
