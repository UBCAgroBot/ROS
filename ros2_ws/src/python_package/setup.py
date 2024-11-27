import os
from glob import glob
from setuptools import setup

package_name = 'python_package'
subfolder = f'{package_name}/scripts'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name,subfolder],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
    ],
    zip_safe=True,
    maintainer='ishaan_datta',
    maintainer_email='ishaandatta737@gmail.com',
    description='Experimental ROS 2 Python Package',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'camera_node = python_package.camera_node:main',
            'inference_node = python_package.inference_node:main',
            'extermination_node = python_package.extermination_node:main',
            'proxy_node = python_package.proxy_node:main',
            'picture_node = python_package.picture_node:main' # remove later
        ],
    },

)