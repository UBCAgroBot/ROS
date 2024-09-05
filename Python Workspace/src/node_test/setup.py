from setuptools import setup

package_name = 'node_test'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/msg', ['msg/BoundingBox.msg']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ishaan_datta',
    maintainer_email='ishaandatta737@gmail.com',
    description='Experimental ROS 2 Python Package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = node_test.camera_node:main',
            'jetson_node = node_test.jetson_node:main',
            'display_node = node_test.display_node:main'
        ],
    },
)
