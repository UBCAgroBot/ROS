from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'python_workspace'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ('share/' + package_name + '/msg', ['msg/BoundingBox.msg']),
        # ('share/' + package_name + '/msg', ['msg/BoundingBoxes.msg']),
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
    ],
    install_requires=[
        'setuptools',
        'rclpy', 
        'cv_bridge', 
        'numpy', 
        'opencv-python', 
        'pycuda', 
        'tensorrt'
    ],
    zip_safe=True,
    maintainer='ishaan_datta',
    maintainer_email='ishaandatta737@gmail.com',
    description='Experimental ROS 2 Python Package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera = python_workspace.camera_node:main',
            'jetson = python_workspace.jetson_node:main',
            'extermination = python_workspace.extermination_node:main'
        ],
    },
)