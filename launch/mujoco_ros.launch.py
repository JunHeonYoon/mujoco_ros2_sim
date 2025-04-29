from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='franka_fr3',
        description='Name of the robot model to be used in MuJoCo'
    )

    controller_class_arg = DeclareLaunchArgument(
        'controller_class',
        default_value='',
        description='Full Python path of the controller class to load (e.g. my_pkg.my_controller.MyController)'
    )

    robot_name = LaunchConfiguration('robot_name')
    controller_class = LaunchConfiguration('controller_class')

    sim_node = Node(
        package='mujoco_ros_sim',      
        executable='mujoco_ros_sim',   
        name='mujoco_sim_node',
        output='screen',
        parameters=[{
            'robot_name': robot_name,
            'controller_class': controller_class,
        }]
    )

    return LaunchDescription([
        robot_name_arg,
        controller_class_arg,
        sim_node
    ])