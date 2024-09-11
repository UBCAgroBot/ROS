import os
from glob import glob
from setuptools import setup

package_name = 'python_workspace'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # Include our package.xml file
        (os.path.join('share', package_name), ['package.xml']),
        # Include all launch files.
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools',],
    zip_safe=True,
    maintainer='ishaan_datta',
    maintainer_email='ishaandatta737@gmail.com',
    description='Experimental ROS 2 Python Package',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'camera_node = python_workspace.camera_node:main',
            'jetson_node = python_workspace.jetson_node:main',
            'extermination_node = python_workspace.extermination_node:main'
        ],
    },
)