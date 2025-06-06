import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'lumencontrol'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*launch.[pxy][yma]*')), #launchfiles
        ('share/' + package_name + '/config', glob('config/*')), #config files
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
            'beat_detector = lumencontrol.beat_detector:main',
            'demixer = lumencontrol.demixer:main',
            'filtering = lumencontrol.filtering:main',
            'features = lumencontrol.feature_extractor:main',
            'gui = lumencontrol.gui:main',
            'spectral_analyzer = lumencontrol.spectral_analyzer:main',
        ],
    },
)
