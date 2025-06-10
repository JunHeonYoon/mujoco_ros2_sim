import time
import numpy as np

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor

import mujoco
import mujoco.viewer

from .utils import load_mj_model, precise_sleep, load_class

class MujocoSimNode(Node):
    def __init__(self):

        super().__init__('mujoco_sim_node')

        descriptor = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter(name='robot_name', descriptor=descriptor)
        self.declare_parameter(name='controller_class', descriptor=descriptor)
        robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        controller_class_str = self.get_parameter('controller_class').get_parameter_value().string_value


        self.mj_model = load_mj_model(robot_name)
        self.print_table(self.mj_model)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.dt = self.mj_model.opt.timestep
        self.viewer_fps = 60.0
        
        self.joint_dict = {}
        self.joint_dict["joint_names"] = []
        for i in range(self.mj_model.njnt):
            name_adr = self.mj_model.name_jntadr[i]
            jname = self.mj_model.names[name_adr:].split(b'\x00', 1)[0].decode('utf-8')
            self.joint_dict["joint_names"].append(jname)

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

    
    def run(self):
        
        self.is_starting = True
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data,
                                          show_left_ui=False, show_right_ui=False) as viewer:
            viewer.sync()
            last_view_time = 0.0

            while viewer.is_running():
                step_start = time.perf_counter()
                rclpy.spin_once(self, timeout_sec=0.0001)

                mujoco.mj_step(self.mj_model, self.mj_data)
                
                pos_dict, vel_dict, tau_ext_dict = {}, {}, {}
                for jid, jname in enumerate(self.joint_dict["joint_names"]):
                    idx_q = self.mj_model.jnt_qposadr[jid]
                    idx_v = self.mj_model.jnt_dofadr[jid]

                    next_q = self.mj_model.jnt_qposadr[jid + 1] if jid + 1 < self.mj_model.njnt else self.mj_model.nq
                    next_v = self.mj_model.jnt_dofadr[jid + 1] if jid + 1 < self.mj_model.njnt else self.mj_model.nv
                    nq = next_q - idx_q
                    nv = next_v - idx_v

                    pos_dict[jname]      = np.copy(self.mj_data.qpos[idx_q : idx_q + nq])
                    vel_dict[jname]      = np.copy(self.mj_data.qvel[idx_v : idx_v + nv])
                    tau_ext_dict[jname]  = np.copy(self.mj_data.qfrc_applied[idx_v : idx_v + nv])
                
                current_sensors = {}
                for name, adr, dim in zip(self.sensor_names,
                                          self.sensor_adr,
                                          self.sensor_dim):
                    current_sensors[name] = np.copy(self.mj_data.sensordata[adr: adr + dim])
                
                if self.controller is not None:
                    self.controller.updateState(pos_dict, vel_dict, tau_ext_dict, current_sensors, self.mj_data.time)
                    
                    if self.is_starting:
                        self.controller.starting()
                        self.is_starting = False   
                    self.controller.compute()
                             
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

    def print_table(self, m):
        # ---------- Helper: build enum-value → name dict for joint types ----------
        jt_enum = mujoco.mjtJoint                 # enum container for joints
        enum2name = {}                            # maps int value → readable str
        for attr in dir(jt_enum):                 # iterate every attribute
            if attr.startswith("mjJNT_"):         # only keep joint type constants
                enum2name[getattr(jt_enum, attr)] = attr[5:].title()  # mjJNT_FREE -> Free

        # ---------- Joint table header ----------
        hdr = " id | name                 | type   | nq | nv | idx_q | idx_v"
        sep = "----+----------------------+--------+----+----+-------+------"
        self.get_logger().info(hdr)
        self.get_logger().info(sep)

        # ---------- Iterate over joints and print per-row ----------
        for jid in range(m.njnt):
            adr  = m.name_jntadr[jid]                         # byte offset in names buffer
            name = m.names[adr:].split(b'\x00', 1)[0].decode()  # null-terminated C-string

            jtype     = int(m.jnt_type[jid])                  # numeric enum value
            type_str  = enum2name.get(jtype, "Unk")           # human-readable

            idx_q = int(m.jnt_qposadr[jid])                   # first qpos index
            idx_v = int(m.jnt_dofadr[jid])                    # first qvel (dof) index

            next_q = m.jnt_qposadr[jid + 1] if jid + 1 < m.njnt else m.nq
            next_v = m.jnt_dofadr[jid + 1] if jid + 1 < m.njnt else m.nv

            nq = int(next_q - idx_q)                          # number of qpos entries
            nv = int(next_v - idx_v)                          # number of dof entries

            # Print one formatted line
            self.get_logger().info(
                f"{jid:3d} | {name:20s} | {type_str:6s} |"
                f" {nq:2d} | {nv:2d} | {idx_q:5d} | {idx_v:4d}"
            )

        # ---------- Actuator table (print after a blank line) ----------
        self.get_logger().info("")                            # readability spacer

        # Build enum-value → name dict for actuator transmission types
        trn_enum = mujoco.mjtTrn
        trn2name = {}
        for attr in dir(trn_enum):
            if attr.startswith("mjTRN_"):
                trn2name[getattr(trn_enum, attr)] = attr[5:].title()  # mjTRN_JOINT -> Joint

        # Pre-compute joint ID → name for quick lookup when actuator targets a joint
        joint_names = {
            jid: m.names[m.name_jntadr[jid]:].split(b'\x00', 1)[0].decode()
            for jid in range(m.njnt)
        }

        # Actuator table header
        ahdr = " id | name                 | trn     | target_joint"
        asep = "----+----------------------+---------+-------------"
        self.get_logger().info(ahdr)
        self.get_logger().info(asep)

        # Iterate over all actuators
        for aid in range(m.nu):                                # m.nu == number of actuators
            adr  = m.name_actuatoradr[aid]                     # byte offset to actuator name
            name = m.names[adr:].split(b'\x00', 1)[0].decode()

            trn_type   = int(m.actuator_trntype[aid])          # transmission type enum
            trn_str    = trn2name.get(trn_type, "Unk")         # readable transmission

            # Each actuator has up to two target IDs in actuator_trnid.
            # For JOINT / JOINTINPARENT the first entry is the joint index.
            target_joint = "-"
            if trn_type in (trn_enum.mjTRN_JOINT,
                            trn_enum.mjTRN_JOINTINPARENT):
                j_id = int(m.actuator_trnid[aid, 0])
                target_joint = joint_names.get(j_id, str(j_id))

            # Log actuator info
            self.get_logger().info(
                f"{aid:3d} | {name:20s} | {trn_str:7s} | {target_joint}"
            )
            
        # ---------- Sensor table (print after a blank line) ----------
        self.get_logger().info("")

        # enum → 문자열 매핑 --------------------
        sens_enum = mujoco.mjtSensor
        sens2name = {getattr(sens_enum, a): a[7:].title()
                    for a in dir(sens_enum) if a.startswith("mjSENS_")}

        obj_enum  = mujoco.mjtObj
        obj2name  = {getattr(obj_enum, a): a[6:].title()
                    for a in dir(obj_enum) if a.startswith("mjOBJ_")}

        # 미리 body / site / joint 등의 id→name 사전 생성
        body_names = {bid: m.names[m.name_bodyadr[bid]:].split(b'\0', 1)[0].decode()
                    for bid in range(m.nbody)}
        site_names = {sid: m.names[m.name_siteadr[sid]:].split(b'\0', 1)[0].decode()
                    for sid in range(m.nsite)}
        # joint_names는 위에서 이미 생성됨

        def obj_name(objtype, objid):
            """주요 object 타입별 id→name 변환(없으면 raw id 반환)"""
            if objtype == obj_enum.mjOBJ_BODY:
                return body_names.get(objid, str(objid))
            if objtype == obj_enum.mjOBJ_SITE:
                return site_names.get(objid, str(objid))
            if objtype == obj_enum.mjOBJ_JOINT:
                return joint_names.get(objid, str(objid))
            return str(objid)

        # 테이블 헤더 출력
        shdr = " id | name                        | type             | dim | adr | target (obj)"
        ssep = "----+-----------------------------+------------------+-----+-----+----------------"
        self.get_logger().info(shdr)
        self.get_logger().info(ssep)

        # 센서 루프
        for sid in range(m.nsensor):
            adr  = m.name_sensoradr[sid]
            name = m.names[adr:].split(b'\0', 1)[0].decode()

            stype = int(m.sensor_type[sid])
            tstr  = sens2name.get(stype, "Unk")

            dim   = int(m.sensor_dim[sid])
            sadr  = int(m.sensor_adr[sid])

            objtype = int(m.sensor_objtype[sid])
            objid   = int(m.sensor_objid[sid])
            target  = f"{obj2name.get(objtype,'-')}:{obj_name(objtype,objid)}" \
                    if objid >= 0 else "-"

            self.get_logger().info(
                f"{sid:3d} | {name:27s} | {tstr:16s} | {dim:3d} | {sadr:3d} | {target}"
            )

            
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
