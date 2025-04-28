from setuptools import setup
from glob import glob
import os

package_name = 'mujoco_ros_sim'

data_files = [
    # ROS2 패키지 리소스 인덱스
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    # package.xml
    ('share/' + package_name, ['package.xml']),
    # launch/*.py
    ('share/' + package_name + '/launch', glob('launch/*.py')),
]

# 'mujoco_menagerie' 디렉토리를 순회하며 모든 파일을 data_files에 추가
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
    packages=[package_name],  # 패키지 디렉토리(mujoco_ros_sim) 인식
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
            # ros2 run mujoco_ros_sim mujoco_ros_sim -> mujoco_ros_sim.py 의 main
            'mujoco_ros_sim = mujoco_ros_sim.mujoco_ros_sim:main'
        ],
    },
)