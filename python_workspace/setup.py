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

# from setuptools import setup

# package_name = 'node_test'

# setup(
#     name=package_name,
#     version='0.0.0',
#     packages=[package_name],
#     py_modules=[],
#     install_requires=['setuptools'],
#     zip_safe=True,
#     maintainer='your_name',
#     maintainer_email='your_email@example.com',
#     description='ROS 2 package with a TensorRT inference node.',
#     license='Apache License 2.0',
#     tests_require=['pytest'],
#     entry_points={
#         'console_scripts': [
#             'jetson_node = node_test.jetson_node:main',
#         ],
#     },
# )

# from setuptools import setup

# package_name = 'node_test'

# setup(
#     name=package_name,
#     version='0.0.0',
#     packages=[package_name],
#     py_modules=[],
#     install_requires=['setuptools'],
#     zip_safe=True,
#     maintainer='your_name',
#     maintainer_email='your_email@example.com',
#     description='ROS 2 package with a TensorRT inference node.',
#     license='Apache License 2.0',
#     tests_require=['pytest'],
#     entry_points={
#         'console_scripts': [
#             'jetson_node = node_test.jetson_node:main',
#             'bounding_box_publisher = node_test.bounding_box_publisher:main',
#         ],
#     },
# )

from setuptools import find_packages, setup

package_name = 'python_workspace'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)