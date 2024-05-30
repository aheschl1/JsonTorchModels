from setuptools import find_packages
from setuptools import setup

setup(
    name='json_torch_models',
    version='0.1.0',
    install_requires=[
        'numpy',
        'torch'
    ],
    packages=find_packages(),
    author='Andrew Heschl',
    author_email='ajheschl@gmail.com',
    url="https://github.com/aheschl1/JsonTorchModels",
    description='Package for defining PyTorch models in JSON, '
                'intended for quick iteration of experimentation, and config-like model defining.'
)