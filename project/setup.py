from setuptools import setup, find_packages

# read the contents of your README file
from os import path
par_directory = path.abspath(path.dirname(path.dirname(__file__)))
with open(path.join(par_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="text-gan",
    description="GANs with Text - Master's Project course towards MSCS at Northeastern",
    long_description=long_description,
    url="https://github.ccs.neu.edu/pavanchhatpar/cs8674",
    packages=find_packages(),
    version="1.0.0",
    author='Pavan Chhatpar',
    author_email='chhatpar.p@husky.neu.edu',
    install_requires=[
        'tensorflow_datasets'
    ]
)