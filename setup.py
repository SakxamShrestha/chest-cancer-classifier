# setup.py
from setuptools import setup, find_packages

setup(
    name="chest_cancer_classifier",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'pillow',
        'pyyaml',
        'scikit-learn', 
        'fastapi',
        'uvicorn',
        "python-multipart",
        "pillow"

    ]
)