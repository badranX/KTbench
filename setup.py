# setup.py

from setuptools import setup, find_packages

setup(
    name='ktbench',
    version='0.0.1',
    author='anonymous',
    author_email='anonymous@anonymous.anonymous',
    description='benchmark knowledge tracing models',
    install_requires=['pandas >= 1.0.0',
                      'datasets == 2.18.0',
                      'GitPython==3.1.42',
                      'yamld == 0.0.22',
                      'scikit-learn>=1.2.2',
                      'torch==2.2.1'
                      ],
    packages = [package for package in find_packages() if package.startswith("ktbench")],
    license='MIT',
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
