#!/usr/bin/env python3
import time
import numpy as np
import threading
import inspect

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import JointState

import mujoco
import mujoco.viewer

from mujoco_ros_sim.utils import load_mj_model, precise_sleep, load_class, print_table


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

        self.cam_fps        = 15.0
        self._cam_rendered     = {}
        
        self.joint_dict = {}
        self.joint_dict["joint_names"] = []
        self.joint_dict["jname_to_jid"] = {}
        for i in range(self.mj_model.njnt):
            name_adr = self.mj_model.name_jntadr[i]
            jname = self.mj_model.names[name_adr:].split(b'\x00', 1)[0].decode('utf-8')
            if not jname:
                continue
            self.joint_dict["joint_names"].append(jname)
            self.joint_dict["jname_to_jid"][jname] = i
            
        self.joint_dict["actuator_names"] = []
        self.joint_dict["aname_to_aid"] = {}
        for i in range(self.mj_model.nu):
            name_adr = self.mj_model.name_actuatoradr[i]
            aname = self.mj_model.names[name_adr:].split(b'\x00', 1)[0].decode('utf-8')
            self.joint_dict["actuator_names"].append(aname)
            self.joint_dict["aname_to_aid"][aname] = i
                            
        self.sensor_names: list[str] = []
        self.sensor_adr:   list[int] = []
        self.sensor_dim:   list[int] = []

        for i in range(self.mj_model.nsensor):
            name_adr = self.mj_model.name_sensoradr[i]
            sname = self.mj_model.names[name_adr:].split(b'\x00', 1)[0].decode('utf-8')

            self.sensor_names.append(sname)
            self.sensor_adr.append(int(self.mj_model.sensor_adr[i]))
            self.sensor_dim.append(int(self.mj_model.sensor_dim[i]))
            
        self.state_lock = threading.Lock()
        self.pos_dict, self.vel_dict, self.tau_ext_dict = {}, {}, {}
        self.sensor_dict = {}

        # ────────────────────────────────INIT_CAM──────────────────────────────────────────
        cam_names, cam_ids, cam_res = [], {}, {}
        for cam_id in range(self.mj_model.ncam):
            cname = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
            if cname in ("", "free"):                         # skip the free cam
                continue
            w, h = self.mj_model.cam_resolution[cam_id]
            cam_names.append(cname)
            cam_ids[cname] = cam_id
            cam_res[cname] = (w, h)

        # allocate ONE renderer big enough for the largest requested resolution
        if cam_names:
            max_w = max(w for w, _ in cam_res.values())
            max_h = max(h for _, h in cam_res.values())
            self.mj_model.vis.global_.offwidth  = max(max_w,  self.mj_model.vis.global_.offwidth)
            self.mj_model.vis.global_.offheight = max(max_h,  self.mj_model.vis.global_.offheight)
            self.cam_renderer = mujoco.Renderer(self.mj_model, width=max_w, height=max_h)
        else:
            self.cam_renderer = None

        self.camera_dict = {
            "names": cam_names,
            "id":    cam_ids,
            "res":   cam_res
        }
        # ────────────────────────────────────────────────────────────────────────────────────────

        
        Controller = load_class(controller_class_str)
        if Controller is not None:
            if inspect.isclass(Controller):
                # Python controller
                self.controller = Controller(self, self.dt, self.joint_dict)
            else:
                # C++ Boost.Python factory 
                self.controller = Controller(self.dt, self.joint_dict)
            self.get_logger().info(f"Sim node with controller={controller_class_str}")
        else:
            self.controller = None

        self.is_starting = True
        
        self.thread = threading.Thread(target=lambda: rclpy.spin(self), daemon=True)
        self.thread.start()

    def _render_rgb(self, cam_name: str) -> np.ndarray:
        """Render *cam_name* and return uint8 RGB (H,W,3) at its native size."""
        cid      = self.camera_dict["id"][cam_name]
        w, h     = self.camera_dict["res"][cam_name]

        self.cam_renderer.disable_depth_rendering()
        self.cam_renderer.update_scene(self.mj_data, camera=cid)
        full_img = self.cam_renderer.render()             # (max_h, max_w, 3)

        return full_img[:h, :w, :]                        # crop & return
    
    def run(self):
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data,
                                          show_left_ui=False, show_right_ui=False) as viewer:
            viewer.sync()
            last_view_time = 0.0
            _last_cam_time = 0.0

            while viewer.is_running():
                step_start = time.perf_counter()

                mujoco.mj_step(self.mj_model, self.mj_data)

                sim_time = self.mj_data.time
                
                tmp_pos, tmp_vel, tmp_tau_ext = {}, {}, {}
                for jname in self.joint_dict["joint_names"]:
                    jid   = self.joint_dict["jname_to_jid"][jname]
                    idx_q = self.mj_model.jnt_qposadr[jid]
                    idx_v = self.mj_model.jnt_dofadr[jid]

                    next_q = (self.mj_model.jnt_qposadr[jid + 1]
                            if jid + 1 < self.mj_model.njnt else self.mj_model.nq)
                    next_v = (self.mj_model.jnt_dofadr[jid + 1]
                            if jid + 1 < self.mj_model.njnt else self.mj_model.nv)

                    nq = next_q - idx_q
                    nv = next_v - idx_v

                    tmp_pos[jname]     = self.mj_data.qpos[idx_q : idx_q + nq]
                    tmp_vel[jname]     = self.mj_data.qvel[idx_v : idx_v + nv]
                    tmp_tau_ext[jname] = self.mj_data.qfrc_applied[idx_v : idx_v + nv]

                tmp_sensor = {}
                for name, adr, dim in zip(self.sensor_names,
                                          self.sensor_adr,
                                          self.sensor_dim):
                    tmp_sensor[name] = self.mj_data.sensordata[adr: adr + dim]

                # ────────────────────────────────────────────────────────────────────────────────────────
                if self.cam_renderer is not None and sim_time - _last_cam_time >= 1.0/self.cam_fps:
                    for cname in self.camera_dict["names"]:
                        self._cam_rendered[cname] = self._render_rgb(cname)
                    _last_cam_time = sim_time

                for cname, img in self._cam_rendered.items():
                    tmp_sensor[cname] = img
                # ────────────────────────────────────────────────────────────────────────────────────────
                    
                with self.state_lock:
                    self.pos_dict = tmp_pos
                    self.vel_dict = tmp_vel
                    self.tau_ext_dict = tmp_tau_ext
                    self.sensor_dict = tmp_sensor
                if self.controller is not None:
                    self.controller.updateState(self.pos_dict, self.vel_dict, self.tau_ext_dict, self.sensor_dict, self.mj_data.time)
                if self.is_starting:
                    if self.controller is not None:
                        self.controller.starting()
                    self.is_starting = False
                if self.controller is not None:
                    self.controller.compute()
                    ctrl_dict = self.controller.getCtrlInput()
                    
                    for given_actuator_name, given_ctrl_cmd in ctrl_dict.items():
                        if given_actuator_name in self.joint_dict["actuator_names"]:
                            actuator_id = self.joint_dict["actuator_names"].index(given_actuator_name)
                            self.mj_data.ctrl[actuator_id] = given_ctrl_cmd
                
                if (sim_time - last_view_time) >= 1.0 / self.viewer_fps:
                    viewer.sync()
                    last_view_time = sim_time
                
                leftover = self.dt - (time.perf_counter() - step_start)
                if leftover > 0:
                    precise_sleep(leftover)
                    
    def pubJointStateCallback(self):
        # Publish joint states
        with self.state_lock:
            if not self.pos_dict:
                return
            positions = []
            velocities = []
            for jname in self.joint_dict["joint_names"]:
                positions.extend(self.pos_dict[jname].tolist())
                velocities.extend(self.vel_dict[jname].tolist())
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_dict["joint_names"]
        msg.position = positions
        msg.velocity = velocities
        self.joint_state_pub.publish(msg)
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
        
if __name__ == "__main__":
    main()