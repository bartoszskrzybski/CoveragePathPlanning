from setuptools import setup, find_packages
import os
from glob import glob
package_name = 'my_gazebo_maps'
setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bartek',
    maintainer_email='bartek@todo.todo',
    description='Paczka z mapami Gazebo',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={},
)
