#pragma once

#include <rclcpp/rclcpp.hpp>
#include <unordered_map>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "mujoco_ros_sim/JointDict.hpp"

using Vec          = Eigen::VectorXd;
using VecMap       = std::unordered_map<std::string, Vec>;
using CtrlInputMap = std::unordered_map<std::string, double>;
using JointDict    = mujoco_ros_sim::JointDict;

/**
 * @brief Base class for C++ controllers in the mujoco_ros_sim package.
 *
 * Instances are created (or wrapped) by the Python MujocoSimNode and
 * driven from its simulation loop.  At each mj_step, Python calls:
 *   1. updateState(...) with the latest qpos, qvel, external torques, sensor data, and sim time
 *   2. starting() once on the very first step
 *   3. compute() to form the control law
 *   4. getCtrlInput() to retrieve actuator commands, which Python then writes into mjData.ctrl[]
 */
class ControllerInterface
{
public:
   /**
   * @brief Construct a new ControllerInterface.
   *
   * The Python MujocoSimNode passes itself as the rclcpp::Node,
   * the Mujoco timestep, and the JointDict it built from the MJCF model.
   * This constructor will:
   *   - store the node handle for parameters or publishers,
   *   - record the simulation dt,
   *   - keep the joint metadata,
   *   - start a ROS 2 executor thread so callbacks (if any) run concurrently.
   *
   * @param[in] node Shared pointer to the ROS 2 node provided by Python.
   * @param[in] dt   Mujoco simulation time step (seconds).
   * @param[in] jd   JointDict with joint/actuator name‐to‐index maps.
   */
  ControllerInterface(const rclcpp::Node::SharedPtr& node,
                      double dt,
                      JointDict jd)
    : node_(node)
    , dt_(dt)
    , mj_joint_dict_(std::move(jd))
  {
    // Attach node to executor and start spinning in its own thread
    exec_.add_node(node_);
    spin_thread_ = std::thread([this]
    {
      // Spin at 5000 Hz to process ROS 2 callbacks promptly
      rclcpp::Rate rate(5000.0);
      while (running_)
      {
        exec_.spin_some();
        rate.sleep();
      }
    });
  }

  virtual ~ControllerInterface() = default;

  /**
   * @brief Called exactly once on the first simulation step.
   *
   * Invoked by the Python loop before the first compute() call,
   * allowing you to read ROS 2 parameters, allocate buffers,
   * or perform one‐time initialization.
   */
  virtual void starting() = 0;

  /**
   * @brief Update the controller’s internal state from simulation data.
   *
   * Called every mj_step by Python before compute().
   *
   * @param[in] pos       Map of joint name → current position vector (from mjData.qpos).
   * @param[in] vel       Map of joint name → current velocity vector (from mjData.qvel).
   * @param[in] tau_ext   Map of joint name → applied external torque (from mjData.qfrc_applied).
   * @param[in] sensors   Map of sensor name → raw sensor array (from mjData.sensordata).
   * @param[in] sim_time  Current simulation time (from mjData.time).
   */
  virtual void updateState(const VecMap& pos,
                           const VecMap& vel,
                           const VecMap& tau_ext,
                           const VecMap& sensors,
                           double sim_time) = 0;

  /**
   * @brief Compute the control commands based on the most recent state.
   *
   * Called every mj_step by Python after updateState().
   * Use your stored state plus any internal variables to formulate
   * the desired outputs (e.g., torques, velocity setpoints).
   */
  virtual void compute() = 0;

  /**
   * @brief Retrieve the control inputs computed in compute().
   *
   * Called every mj_step by Python after compute().
   * The returned map (actuator name → command value) is applied directly to
   * mjData.ctrl in the Python simulation loop.
   *
   * @return CtrlInputMap Map of actuator name → control command.
   */
  virtual CtrlInputMap getCtrlInput() const = 0;

protected:
  rclcpp::Node::SharedPtr node_;           // ROS 2 node for parameters, topics, services
  double                  dt_;             // Simulation time step (s)
  JointDict               mj_joint_dict_;  // Metadata for Mujoco joints

private:
  rclcpp::executors::SingleThreadedExecutor exec_;          // Executor spinning callbacks
  std::thread                               spin_thread_;   // Background spin thread
  std::atomic_bool                          running_{true}; // Flag to keep spin thread alive
};
