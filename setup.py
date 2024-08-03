from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT = '-e .'

def get_requiremnets(File_path:str) -> List[str]:
    '''
    This function will return the list of requirements 
    '''
    requirements = []
    with open(File_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='VisionBuddy',
    version='0.0.1',
    author='kuldeep Kadu',
    author_email='kuldeepkadu1210@gmail.com',
    packages=find_packages(),
    install_requires=get_requiremnets('requirements.txt')
)