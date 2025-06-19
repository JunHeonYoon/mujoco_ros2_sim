from rclpy.node import Node
from typing import Dict, Any


class ControllerInterface:
    def __init__(self, node: Node, dt: float, mj_joint_dict: Dict[str, Any]):
        self.node = node
        self.dt   = dt
        self.mj_joint_dict = mj_joint_dict

    def starting(self) -> None:
        raise NotImplementedError

    def updateState(
        self,
        pos_dict: Dict[str, Any],
        vel_dict: Dict[str, Any],
        tau_ext_dict: Dict[str, Any],
        sensor_dict: Dict[str, Any],
        current_time: float
    ) -> None:
        raise NotImplementedError

    def compute(self) -> None:
        raise NotImplementedError

    def getCtrlInput(self) -> Dict[str, float]:
        raise NotImplementedError
