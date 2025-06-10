import numpy as np
from rclpy.node import Node

class ControllerInterface:
    def __init__(self, 
                 node: Node, 
                 dt: float, 
                 mj_joint_dict: dict):
        self.node = node
        self.dt = dt
        self.mj_joint_dict = mj_joint_dict

    def starting(self) -> None:
        raise NotImplementedError("starting() must be overridden)")

    def updateState(self, 
                    pos_dict: dict, 
                    vel_dict: dict, 
                    tau_ext_dict: dict,
                    current_sensors: dict, 
                    current_time: float) -> None:
        raise NotImplementedError("updateState() must be overridden.")

    def compute(self) -> None:
        raise NotImplementedError("compute() must be overridden.")

    def getCtrlInput(self) -> dict:
        raise NotImplementedError("getCtrlInput() must be overridden.")