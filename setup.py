from setuptools import setup, find_packages

setup(
    name="air_hockey_agent",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
    ],
) 