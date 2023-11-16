from setuptools import find_packages, setup
from typing import List

HYPHE_E_DOT = '-e .'

def get_requiements(file_path:str)->List[str]:
    '''
    this function will return list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

    if HYPHE_E_DOT in requirements:
        requirements.remove(HYPHE_E_DOT)

    return requirements


setup(
    name='endmlproject',
    version='0.0.1',
    author='Jonah Uche',
    author_email='jonahuche600@gmail.com',
    description='End to end datascience implemetation',
    packages= find_packages(),
    install_requires=get_requiements('requirements.txt')
)