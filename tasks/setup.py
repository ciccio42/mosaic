# setup.py
from setuptools import setup, find_packages

setup(
    name='robosuite_env',
    version='0.0.1',
    include_package_data=True,
    packages=find_packages(exclude=['test_models']),
    package_data={"robosuite_env.objects": ['*','*/*','*/*/*','*/*/*/*'],
                  "robosuite_env.arena": ['*','*/*'],
                  "robosuite_env.controllers": ['*','*/*'],
                  "robosuite_env.utils": ['*'],
                  "robosuite_env.config": ['*']}
)

if __name__ == "__main__":
    packages = find_packages(exclude=['test_models'])
    print(packages)
