#! /usr/bin/env python3
from setuptools import setup,find_packages
    
install_requires = [
'joblib>=1.1.0',
'numpy>=1.21.1',
'opencv_python',
'colorcet',
'hdbscan',
'scipy',
'tqdm',
'pandas',
'pytesseract',
'matplotlib>=3.3.0',
'seaborn>=0.11.1',
'Pillow',
'plateypus',
'scikit_learn',
]

classifiers = [
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Operating System :: Unix',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: Microsoft :: Windows',
    'Intended Audience :: Science/Research'
]

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="radianceQuantifier",
    version="0.4.17",
    author = "Sooraj Achar",
    author_email = "acharsr@nih.gov",
    description = "Automatically crops mice and quantifies their tumor luminescences from raw IVIS images",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/soorajachar/radianceQuantifier",
    classifiers = classifiers,
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "radianceQuantifier = radianceQuantifier.__main__:main"
        ]
    },
)
