import numpy as np
from rclpy.node import Node

class ControllerInterface:
    """
    Default Controller Interface (Abstract Class).

    This abstract class defines the interface for controllers in the system.
    Subclasses must implement the following methods:
      - starting() -> None
      - updateState(pos, vel, tau, current_time) -> None
      - compute() -> None
      - getCtrlInput() -> tuple[np.ndarray, list]

    The updateState() method should use the provided robot state (positions, velocities, torques)
    to update any internal states, such as sensor readings or computed Jacobians.
    """

    def __init__(self, node: Node, dt: float, mj_joint_names: list):
        """
        Initializes the controller interface.

        Parameters:
            node (Node): A reference to the ROS2 node (used for logging and ROS communication).
            dt (float): The simulation time step in seconds.
            mj_joint_names (list): A list of joint names (strings) from the MuJoCo model.
            
        Returns:
            None
        """
        self.node = node
        self.dt = dt  # Simulation time step
        self.mj_joint_names = mj_joint_names

    def starting(self) -> None:
        """
        Called once at the start of the controller's execution.

        This method should be overridden by subclasses to perform any necessary 
        initialization routines, such as setting initial conditions.

        Returns:
            None

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError("starting() must be overridden)")

    def updateState(self, pos: np.ndarray, vel: np.ndarray, tau_ext: np.ndarray, current_time: float) -> None:
        """
        Updates the controller's internal state with the latest robot data.

        Parameters:
            pos (np.ndarray): The current joint positions of the robot.
            vel (np.ndarray): The current joint velocities of the robot.
            tau_ext (np.ndarray): The current  external torques applied to the robot joints.
            current_time (float): The current simulation time.

        Returns:
            None

        This method should be overridden to update any internal controller states,
        such as sensor data or computed quantities like Jacobians.
        
        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError("updateState() must be overridden.")

    def compute(self) -> None:
        """
        Computes the control commands based on the current state.

        Returns:
            None

        This method should be overridden by subclasses to compute control commands 
        (e.g., torques or position commands) based on the updated state. The computed
        commands should be stored internally to be later retrieved via getCtrlInput().
        
        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError("update() must be overridden.")

    def getCtrlInput(self) -> tuple[np.ndarray, list]:
        """
        Retrieves the computed control commands along with their corresponding actuator names.

        Returns:
            tuple:
                - np.ndarray: An array of computed control commands (e.g., torques, positions).
                - list: A list of actuator names (strings) corresponding to the control commands.

        This method should be implemented to return the control inputs that have been computed 
        by the compute() method.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError("getCtrlInput() must be overridden.")
        # Typically, you would return something like:
        # return self.ctrl_input, self.actuator_names
