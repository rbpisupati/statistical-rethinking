from setuptools import setup, find_packages  # Always prefer setuptools over distutils
from codecs import open  # To use a consistent encoding
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='PyRethink',
    version='0.0.1',
    description='A simple python library for Statistical rethinking',
    long_description=long_description,
    long_description_content_type='text/markdown',  # This is important for PyPi packaging
    url='https://github.com/rbpisupati/',
    author=['Rahul Pisupati'],
    author_email='rahul.pisupati@gmi.oeaw.ac.at',
    license='GMI',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Programming Language :: Python :: 3.4',
    ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    include_package_data=True,
    install_requires=[
        "pyro-ppl",
        'torch',
        "networkx"
    ],
    entry_points={
        'console_scripts': [
            'pyrethink=pyrethink:main'
        ],
    },
)