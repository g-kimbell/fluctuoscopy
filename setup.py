from setuptools import setup, find_packages

setup(
    name='fluctuoscopy',
    version='0.1.0',
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        'numpy',
        'multiprocess',
    ],
    extras_require={
        'dev': [
            'pytest',
        ],
    },
    package_data={
        'fluctuoscopy': ['bin/*'],
    },
    author='Graham Kimbell',
    description='A Python wrapper with additional helper functions for the FSCOPE program, written by Andreas Glatz',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/g-kimbell/fluctuoscopy',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Windows',
    ],
    python_requires='>=3.6',
)