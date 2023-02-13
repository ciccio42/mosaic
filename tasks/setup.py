
# setup.py
from setuptools import setup, find_packages

setup(
    name='robosuite_env',
    version='0.0.1',
    include_package_data=True,
    packages=find_packages(exclude=['test_models']),
    package_data={"robosuite_env.objects": ['*','*/*','*/*/*','*/*/*/*'],
<<<<<<< Updated upstream
                  "robosuite_env.arena": ['*','*/*']}
=======
                  "robosuite_env.arena": ['*','*/*'],
                  "robosuite_env.controllers": ['*','*/*'],
                  "robosuite_env.utils": ['*'],
                  "robosuite_env.config": ['*']}
>>>>>>> Stashed changes
)

if __name__ == "__main__":
    packages = find_packages(exclude=['test_models'])
<<<<<<< Updated upstream
    print(packages)
=======
    print(packages)
>>>>>>> Stashed changes
