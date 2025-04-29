from setuptools import setup
from glob import glob
import os

package_name = 'mujoco_ros_sim'

data_files = [
    # ROS2 package resource index
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    # package.xml
    ('share/' + package_name, ['package.xml']),
    # launch/*.py
    ('share/' + package_name + '/launch', glob('launch/*.py')),
]

# Traverse the 'mujoco_menagerie' directory and add all files to data_files
robots_path = 'mujoco_menagerie'
for root, dirs, files in os.walk(robots_path):
    for file in files:
        relative_path = os.path.relpath(root, robots_path)
        install_path = os.path.join('share', package_name, robots_path, relative_path)
        data_files.append((
            install_path, 
            [os.path.join(root, file)]
        ))

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],  # Recognize the package directory (mujoco_ros_sim)
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='JunheonYoon',
    maintainer_email='yoonjh98@snu.ac.kr',
    description='Robot simulator package with MuJoCo & ROS2',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # ros2 run mujoco_ros_sim mujoco_ros_sim -> main function in mujoco_ros_sim.py
            'mujoco_ros_sim = mujoco_ros_sim.mujoco_ros_sim:main'
        ],
    },
)