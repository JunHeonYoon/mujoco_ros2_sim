import numpy as np
from rclpy.node import Node

class RobotDataInterface:
    """
    Interface for handling and updating robot state data.
    
    This abstract class defines the interface for managing the robot's state,
    including joint positions, velocities, and torques obtained from a MuJoCo simulation.
    Subclasses must override the updateState() method to implement the desired behavior.
    """

    def __init__(self, node: Node, mj_joint_names: list):
        """
        Initializes the RobotDataInterface.

        Parameters:
            node (Node): A ROS2 node instance used for logging and ROS communications.
            mj_joint_names (list): A list of joint names (strings) corresponding to the MuJoCo robot's joints.

        Returns:
            None
        """
        self.node = node
        self.mj_joint_names = mj_joint_names
    
    def updateState(self, pos: np.ndarray, vel: np.ndarray, tau_ext: np.ndarray):
        """
        Updates the robot's state data with the latest simulation values.

        Parameters:
            pos (np.ndarray): An array containing the current joint positions of the robot.
            vel (np.ndarray): An array containing the current joint velocities of the robot.
            tau_ext (np.ndarray): An array containing the current external torques applied to the robot joints.

        Returns:
            None

        This method should be overridden by subclasses to process and store the robot's
        state information (e.g., for feedback control or state estimation purposes).

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError("updateState() must be overridden.")
