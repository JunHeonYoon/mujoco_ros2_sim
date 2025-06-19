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
 * Base class for *C++* controllers.
 * Each instance owns (or receives) an rclcpp::Node to interact with ROS 2.
 */
class ControllerInterface
{
public:
  /*
  node_
  dt_
  mj_joint_dict_
  */
  ControllerInterface(const rclcpp::Node::SharedPtr& node,
                      double dt,
                      JointDict jd)
  : node_(node), dt_(dt), mj_joint_dict_(std::move(jd)) 
  {
    exec_.add_node(node_);
    spin_thread_ = std::thread([this]
    {
      rclcpp::Rate r(5'000);                 // 5 kHz (충분히 빠름)
      while (running_)
      {
        exec_.spin_some();                   // pending 콜백 한 덩어리 처리
        r.sleep();
      }
    });
  }

  virtual ~ControllerInterface() = default;

  virtual void starting() = 0;

  virtual void updateState(const VecMap& pos,
                           const VecMap& vel,
                           const VecMap& tau_ext,
                           const VecMap& sensors,
                           double sim_time) = 0;

  virtual void compute() = 0;

  virtual CtrlInputMap getCtrlInput() const = 0;

protected:
  rclcpp::Node::SharedPtr node_;   ///< full ROS 2 API access
  double                  dt_;
  JointDict               mj_joint_dict_;
private:
  rclcpp::executors::SingleThreadedExecutor       exec_;
  std::thread                                     spin_thread_;
  std::atomic_bool                                running_{true};
};
