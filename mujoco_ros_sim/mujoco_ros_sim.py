import time
import numpy as np
import threading

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import JointState

import mujoco
import mujoco.viewer

from .utils import load_mj_model, precise_sleep, load_class, print_table


class MujocoSimNode(Node):
    def __init__(self):

        super().__init__('mujoco_sim_node')

        descriptor = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter(name='robot_name', descriptor=descriptor)
        self.declare_parameter(name='controller_class', descriptor=descriptor)
        robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        controller_class_str = self.get_parameter('controller_class').get_parameter_value().string_value
        
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        
        self.create_timer(0.01, self.pubJointStateCallback)

        self.mj_model = load_mj_model(robot_name)
        print_table(self, self.mj_model)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.dt = self.mj_model.opt.timestep
        self.viewer_fps = 60.0
        
        self.joint_dict = {}
        self.joint_dict["joint_names"] = []
        self.joint_dict["joint_ids"] = []
        for i in range(self.mj_model.njnt):
            name_adr = self.mj_model.name_jntadr[i]
            jname = self.mj_model.names[name_adr:].split(b'\x00', 1)[0].decode('utf-8')
            if not jname:
                continue
            self.joint_dict["joint_names"].append(jname)
            self.joint_dict["joint_ids"].append(i)
            
        self.joint_dict["actuator_names"] = []
        for i in range(self.mj_model.nu):
            name_adr = self.mj_model.name_actuatoradr[i]
            aname = self.mj_model.names[name_adr:].split(b'\x00', 1)[0].decode('utf-8')
            self.joint_dict["actuator_names"].append(aname)
                            
        self.sensor_names: list[str] = []
        self.sensor_adr:   list[int] = []
        self.sensor_dim:   list[int] = []

        for i in range(self.mj_model.nsensor):
            name_adr = self.mj_model.name_sensoradr[i]
            sname = self.mj_model.names[name_adr:].split(b'\x00', 1)[0].decode('utf-8')

            self.sensor_names.append(sname)
            self.sensor_adr.append(int(self.mj_model.sensor_adr[i]))
            self.sensor_dim.append(int(self.mj_model.sensor_dim[i]))
        
        Controller = load_class(controller_class_str)
        if Controller is not None:
            self.controller = Controller(self, self.dt, self.joint_dict)
            self.get_logger().info(f"Sim node with controller={controller_class_str}")
        else:
            self.controller = None

        self.thread = threading.Thread(target=lambda: rclpy.spin(self), daemon=True)
    
    def run(self):
        self.is_starting = True
        self.thread.start()
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data,
                                          show_left_ui=False, show_right_ui=False) as viewer:
            viewer.sync()
            last_view_time = 0.0

            while viewer.is_running():
                step_start = time.perf_counter()
                rclpy.spin_once(self, timeout_sec=0.0001)

                mujoco.mj_step(self.mj_model, self.mj_data)
                
                self.pos_dict, self.vel_dict, self.tau_ext_dict = {}, {}, {}
                for jid, jname in zip(self.joint_dict["joint_ids"], self.joint_dict["joint_names"]):
                    idx_q = self.mj_model.jnt_qposadr[jid]
                    idx_v = self.mj_model.jnt_dofadr[jid]

                    next_q = self.mj_model.jnt_qposadr[jid + 1] if jid + 1 < self.mj_model.njnt else self.mj_model.nq
                    next_v = self.mj_model.jnt_dofadr[jid + 1] if jid + 1 < self.mj_model.njnt else self.mj_model.nv
                    nq = next_q - idx_q
                    nv = next_v - idx_v

                    self.pos_dict[jname]      = np.copy(self.mj_data.qpos[idx_q : idx_q + nq])
                    self.vel_dict[jname]      = np.copy(self.mj_data.qvel[idx_v : idx_v + nv])
                    self.tau_ext_dict[jname]  = np.copy(self.mj_data.qfrc_applied[idx_v : idx_v + nv])
                
                current_sensors = {}
                for name, adr, dim in zip(self.sensor_names,
                                          self.sensor_adr,
                                          self.sensor_dim):
                    current_sensors[name] = np.copy(self.mj_data.sensordata[adr: adr + dim])
                
                if self.controller is not None:
                    self.controller.updateState(self.pos_dict, self.vel_dict, self.tau_ext_dict, current_sensors, self.mj_data.time)
                    
                    if self.is_starting:
                        self.controller.starting()
                        self.is_starting = False   
                    self.controller.compute()
                else:
                    self.is_starting = False   
                if self.controller is not None:
                    ctrl_dict = self.controller.getCtrlInput()
                    
                    for given_actuator_name, given_ctrl_cmd in ctrl_dict.items():
                        if given_actuator_name in self.joint_dict["actuator_names"]:
                            actuator_id = self.joint_dict["actuator_names"].index(given_actuator_name)
                            self.mj_data.ctrl[actuator_id] = given_ctrl_cmd
                            
                sim_time = self.mj_data.time
                if (sim_time - last_view_time) >= 1.0 / self.viewer_fps:
                    viewer.sync()
                    last_view_time = sim_time
                
                leftover = self.dt - (time.perf_counter() - step_start)
                if leftover > 0:
                    precise_sleep(leftover)
                    
    def pubJointStateCallback(self):
        # Publish joint states
        if not self.is_starting:
            joint_state_msg = JointState()
            joint_state_msg.header.stamp = self.get_clock().now().to_msg()
            joint_state_msg.name = self.joint_dict["joint_names"]
            positions = []
            velocities = []
            for jname in self.joint_dict["joint_names"]:
                positions.extend(self.pos_dict[jname].tolist())
                velocities.extend(self.vel_dict[jname].tolist())
            joint_state_msg.position = positions
            joint_state_msg.velocity = velocities
            self.joint_state_pub.publish(joint_state_msg)
            
def main(args=None):

    rclpy.init(args=args)  # Initialize the ROS2 communications.
    node = MujocoSimNode()  # Create an instance of the simulation node.
    try:
        node.run()  # Run the simulation loop.
    except KeyboardInterrupt:
        # Exit cleanly on Ctrl+C.
        pass
    finally:
        node.destroy_node()  # Clean up the node.
        rclpy.shutdown()  # Shutdown the ROS2 communications.
