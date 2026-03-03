from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    requirement_list: List[str] = []
    try:
        with open(file_path, 'r') as file:
            # Read lines and remove whitespace/newlines immediately
            lines = [line.strip() for line in file.readlines()]

            for requirement in lines:
                # Skip empty lines and comments
                if not requirement or requirement.startswith('#'):
                    continue
                
                # Use 'in' or 'startswith' to be safer against extra spaces
                if '-e' in requirement and '.' in requirement:
                    continue
                
                requirement_list.append(requirement)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")

    return requirement_list

setup(

    name='Credit Risk Modelling',
    version='0.1',
    author='Kenaz Jose',
    author_email='kenazjose007@ghmail.com',
    packages=find_packages(),
    install_requires=get_requirements('D:\\Credit Risk Modelling\\requirements.txt'),
)