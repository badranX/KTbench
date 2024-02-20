# setup.py

from setuptools import setup, find_packages

setup(
    name='ktbench',
    version='0.0.1',
    author='Yahya Badran',
    author_email='techtweaking@gmail.com',
    description='benchmark knowledge tracing models',
    url='https://github.com/badranx/KTbench',
    install_requires=['pandas >= 1.0.0'],
    packages = [package for package in find_packages() if package.startswith("ktbench")],
    license='MIT',
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
