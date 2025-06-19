#include "mujoco_ros_sim/ControllerRegistry.hpp"

ControllerRegistry& ControllerRegistry::instance()
{
  static ControllerRegistry inst;
  return inst;
}
