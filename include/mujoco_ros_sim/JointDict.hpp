#pragma once
#include <vector>
#include <string>
#include <unordered_map>

namespace mujoco_ros_sim
{
struct JointDict
{
  std::vector<std::string> joint_names;
  std::unordered_map<std::string,int> jname_to_jid;

  std::vector<std::string> actuator_names;
  std::unordered_map<std::string,int> aname_to_aid;
};
}  // namespace mujoco_ros_sim
