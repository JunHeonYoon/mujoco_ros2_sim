from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 런치 아규먼트들
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

    robot_data_class_arg = DeclareLaunchArgument(
        'robot_data_class',
        default_value='',
        description='Full Python path of the RobotData class to load (e.g. my_pkg.my_robot_data.MyRobotData)'
    )

    # LaunchConfiguration 객체
    robot_name = LaunchConfiguration('robot_name')
    controller_class = LaunchConfiguration('controller_class')
    robot_data_class = LaunchConfiguration('robot_data_class')

    # 실제 노드 실행 설정
    sim_node = Node(
        package='mujoco_ros_sim',      # 패키지 이름 (setup.py의 project_name 혹은 package.xml의 <name>)
        executable='mujoco_ros_sim',   # setup.py 혹은 entry_point에서 지정된 실행 가능 이름
        name='mujoco_sim_node',
        output='screen',
        parameters=[{
            'robot_name': robot_name,
            'controller_class': controller_class,
            'robot_data_class': robot_data_class
        }]
    )

    return LaunchDescription([
        robot_name_arg,
        controller_class_arg,
        robot_data_class_arg,
        sim_node
    ])