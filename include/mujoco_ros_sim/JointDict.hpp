#pragma once
#include <vector>
#include <string>
#include <unordered_map>

namespace mujoco_ros_sim
{
/**
 * @brief Lookup tables for Mujoco joint and actuator names to indices.
 *
 * This struct is populated by MujocoSimNode (Python) and passed to C++ controllers.
 * It provides mappings between joint/actuator names and their corresponding indices
 * in the Mujoco model arrays.
 */
struct JointDict
{
  /// Ordered list of Mujoco joint names (size == model->njnt)
  std::vector<std::string> joint_names;

  /// Map from joint name to its index in the `joint_names` vector and Mujoco model
  std::unordered_map<std::string,int> jname_to_jid;

  /// Ordered list of Mujoco actuator names (size == model->nu)
  std::vector<std::string> actuator_names;

  /// Map from actuator name to its index in the `actuator_names` vector and Mujoco model
  std::unordered_map<std::string,int> aname_to_aid;
};

}  // namespace mujoco_ros_sim
