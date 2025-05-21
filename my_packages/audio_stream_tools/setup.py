import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'audio_stream_tools'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files.
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='appuser',
    maintainer_email='bancsimark02@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'playbacker = audio_stream_tools.audio_playback_node:main',
            'recorder = audio_stream_tools.audio_recorder_node:main',
            'input =  audio_stream_tools.audio_input_node:main',
            'output = audio_stream_tools.audio_output_node:main',
        ],
    },
)
